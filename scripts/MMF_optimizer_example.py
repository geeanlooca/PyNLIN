import os
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.fiber
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.constants import lambda2nu, nu2lambda

from pynlin.raman.pytorch.gain_optimizer import GainOptimizer
from pynlin.raman.pytorch.solvers import MMFRamanAmplifier
from pynlin.raman.solvers import MMFRamanAmplifier as NumpyMMFRamanAmplifier
from pynlin.utils import dBm2watt, watt2dBm
import pynlin.constellations
import json
from matplotlib.cm import viridis

plt.rcParams.update({
	"text.usetex": False,
})

import logging
logging.basicConfig(filename='MMF_optimizer.log', encoding='utf-8', level=logging.INFO)
log = logging.getLogger(__name__)
log.debug("starting to load sim_config.json")

f = open("./scripts/sim_config.json")
data = json.load(f)
# print(data)
dispersion = data["dispersion"]
effective_area = data["effective_area"]
baud_rate = data["baud_rate"]
fiber_lengths = data["fiber_length"]
num_channels = data["num_channels"]
interfering_grid_index = data["interfering_grid_index"]
channel_spacing = data["channel_spacing"]
center_frequency = data["center_frequency"]
store_true = data["store_true"]
pulse_shape = data["pulse_shape"]
partial_collision_margin = data["partial_collision_margin"]
num_co = data["num_co"]
num_ct = data["num_ct"]
wavelength = data["wavelength"]
special = data["special"]
pump_direction = data["pump_direction"]
num_only_co_pumps = data['num_only_co_pumps']
num_only_ct_pumps = data['num_only_ct_pumps']
gain_dB_setup = data['gain_dB_list']
gain_dB_list = np.linspace(gain_dB_setup[0], gain_dB_setup[1], gain_dB_setup[2])
power_dBm_setup = data['power_dBm_list']
power_dBm_list = np.linspace(power_dBm_setup[0], power_dBm_setup[1], power_dBm_setup[2])
num_modes = data['num_modes']
oi_fit = np.load('oi_fit.npy')
oi_avg = np.load('oi_avg.npy')
use_avg_oi = True


num_original_modes = oi_avg[0].shape[0]
matrix_avg = oi_avg
matrix_zeros = np.tile(np.zeros((num_original_modes, num_original_modes))[None, :, :], (5, 1, 1))
oi_avg_complete = np.stack((*matrix_zeros, matrix_avg), axis=0)

# Manual configuration
power_per_channel_dBm_list = power_dBm_list
# Pumping scheme choice
pumping_schemes = ['ct']
# num_only_co_pumps = 4
# num_only_ct_pumps = 4
optimize = True
profiles = True


fiber_length = fiber_lengths[0]
gain_dB = -5
power_per_pump = dBm2watt(0)

log.warning("end loading of parameters")

beta2 = pynlin.utils.dispersion_to_beta2(
    dispersion, wavelength
)
ref_bandwidth = baud_rate

if use_avg_oi:
  oi_set = oi_avg_complete
else:
  oi_set = oi_fit
fiber = pynlin.fiber.MMFiber(
    effective_area=80e-12,
    beta2=beta2,
    modes=num_modes,
    overlap_integrals=oi_set,
)
# assert((oi_set == oi_fit).all())

actual_fiber = pynlin.fiber.MMFiber(
    effective_area=80e-12,
    beta2=beta2,
    modes=num_modes,
    overlap_integrals=oi_fit,
)

wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=190e12
)

# comute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
channel_spacing = interfering_frequency - frequency_of_interest
partial_collision_margin = 5
points_per_collision = 10

log.warning("select only first lenfth and gain")
length_setup = int(fiber_length * 1e-3)
optimization_result_path_ct = '../results_' + \
    str(length_setup) + '/optimization_gain_' + str(gain_dB) + \
    '_scheme__' + str(num_only_ct_pumps) + '_ct/'
results_path_ct = '../results_' + \
    str(length_setup) + '/' + str(num_only_ct_pumps) + '_ct/'

# PRECISION REQUIREMENTS ESTIMATION =================================
max_channel_spacing = wdm.frequency_grid(
)[num_channels - 1] - wdm.frequency_grid()[0]
max_num_collisions = len(pynlin.nlin.get_m_values(
    fiber,
    fiber_length,
    max_channel_spacing,
    1 / baud_rate,
    partial_collisions_start=partial_collision_margin,
    partial_collisions_end=partial_collision_margin)
)
integration_steps = max_num_collisions * points_per_collision
# Suggestion: 100m step is sufficient
dz = 100
integration_steps = int(np.ceil(fiber_length / dz))
z_max = np.linspace(0, fiber_length, integration_steps)

np.save("z_max.npy", z_max)
pbar_description = "Optimizing vs signal power"
pbar = tqdm.tqdm(power_per_channel_dBm_list, leave=False)
pbar.set_description(pbar_description)


def ct_solver(power_per_channel_dBm, use_precomputed=False):
    if use_precomputed and os.path.exists(results_path_ct + "pump_solution_ct_power" + str(power_per_channel_dBm) + "_opt_gain_" + str(gain_dB) + ".npy"):
        print("Result already computed for power: ",
              power_per_channel_dBm, " and gain: ", gain_dB)
        return
    else:
        print("Computing the power: ", power_per_channel_dBm, " and gain: ", gain_dB)
    # print("Power per channel: ", power_per_channel_dBm, "dBm")
    num_pumps = num_only_ct_pumps
    # initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)
    # BROMAGE
    initial_pump_frequencies = np.array(lambda2nu([1447e-9, 1467e-9, 1485e-9, 1515e-9]))
    power_per_channel = dBm2watt(power_per_channel_dBm)
    signal_wavelengths = wdm.wavelength_grid()
    initial_pump_wavelengths = nu2lambda(initial_pump_frequencies[:num_pumps])

    initial_pump_powers = np.ones_like(initial_pump_wavelengths) * power_per_pump
    initial_pump_powers = initial_pump_powers.repeat(num_modes, axis=0)
    torch_amplifier_ct = MMFRamanAmplifier(
        fiber_length,
        integration_steps,
        num_pumps,
        signal_wavelengths,
        power_per_channel,
        fiber,
        counterpumping=True
    )

    optimizer = GainOptimizer(
        torch_amplifier_ct,
        torch.from_numpy(initial_pump_wavelengths),
        torch.from_numpy(initial_pump_powers),
    )

    signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
    signal_powers = signal_powers[:, None].repeat(num_modes, axis=1)
    target_spectrum = watt2dBm(signal_powers)[None, :, :] + gain_dB
    learning_rate = 2e-4

    pump_wavelengths, pump_powers = optimizer.optimize(
        target_spectrum=target_spectrum,
        epochs=1,
        learning_rate=learning_rate,
        lock_wavelengths=200,
        )
    print(pump_powers)
    amplifier = NumpyMMFRamanAmplifier(fiber)

    print("\n=========== results ===================")
    print("Pump powers")
    print(watt2dBm(pump_powers.reshape((num_pumps, num_modes))))
    print("Initial pump powers")
    print(watt2dBm(initial_pump_powers.reshape((num_pumps, num_modes))))
    print("Pump frequency")
    print(lambda2nu(pump_wavelengths))
    print("Pump wavelenghts")
    print(pump_wavelengths)
    print("Initial pump wavelenghts")
    print(initial_pump_wavelengths)
    print("=========== end ===================\n\n")
    
    print("WARN: converting the inlined pump_powers into a matrix")
    pump_powers = pump_powers.reshape((num_pumps, num_modes))
    pump_solution, signal_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers,
        pump_wavelengths,
        z_max,
        actual_fiber,
        counterpumping=True,
        reference_bandwidth=ref_bandwidth
    )
    
    # fixed mode
    plt.clf()
    cmap = viridis
    z_plot = np.linspace(0, fiber_length, len(pump_solution[:, 0, 0])) * 1e-3
    for i in range(num_pumps):
      if i ==1:
        plt.plot(z_plot,
                 watt2dBm(pump_solution[:, i, :]), label="pump",  color=cmap(i/num_pumps),ls="--")
      else:
        plt.plot(z_plot, 
                 watt2dBm(pump_solution[:, i, :]), color=cmap(i/num_pumps),ls="--")
    for i in range(num_channels):
      if i==1:
        plt.plot(z_plot, watt2dBm(signal_solution[:, i, :]),  color=cmap(i/num_channels),label="signal")
      else:
        plt.plot(z_plot, 
                   watt2dBm(signal_solution[:, i, :]), color=cmap(i/num_channels))
    plt.legend()
    plt.grid()
    
    flatness = np.max(watt2dBm(signal_solution[-1, :, :])) - watt2dBm(np.min(signal_solution[-1, :, :]))
    print(f"final flatness: {flatness:.2f}")
    # plt.show()
    plt.savefig("media/optimized_profile.pdf")
    return

ct_solver(-30.0)