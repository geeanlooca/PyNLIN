import os
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from multiprocessing import Pool

import numpy as np
import torch
from scipy.constants import lambda2nu, nu2lambda

from pynlin.raman.pytorch.gain_optimizer import GainOptimizer
from pynlin.raman.pytorch.solvers import MMFRamanAmplifier
from pynlin.raman.solvers import RamanAmplifier as NumpyRamanAmplifier
from pynlin.utils import dBm2watt, watt2dBm
import pynlin.constellations
from random import shuffle
import json

f = open("./scripts/sim_config.json")
data = json.load(f)
print(data)
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
oi_fit = np.load('oi.npy')
oi_avg = np.load('oi_avg.npy')

# Manual configuration
power_per_channel_dBm_list = power_dBm_list
# Pumping scheme choice
pumping_schemes = ['ct']
num_only_co_pumps = 4
num_only_ct_pumps = 4
optimize = True
profiles = True

###########################################
#  COMPUTATION OF AMPLITUDE FUNCTIONS
###########################################
beta2 = pynlin.utils.dispersion_to_beta2(
    dispersion * 1e-12 / (1e-9 * 1e3), wavelength
)
ref_bandwidth = baud_rate
fiber = pynlin.fiber.MMFiber(
    effective_area=80e-12,
    beta2=beta2,
    modes=num_modes,
    overlap_integrals=oi_fit,
    overlap_integrals_avg=oi_avg,
)
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing * 1e-9,
    num_channels=num_channels,
    center_frequency=190
)

# comute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
channel_spacing = interfering_frequency - frequency_of_interest
partial_collision_margin = 5
points_per_collision = 10
print("Warning: select only first lenfth and gain")
fiber_length = fiber_lengths[0]
gain_dB = gain_dB_list[0]

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


def ct_solver(power_per_channel_dBm):
    if os.path.exists(results_path_ct + "pump_solution_ct_power" + str(power_per_channel_dBm) + "_opt_gain_" + str(gain_dB) + ".npy"):
        print("Result already computed for power: ",
              power_per_channel_dBm, " and gain: ", gain_dB)
        return
    else:
        print("Computing the power: ", power_per_channel_dBm, " and gain: ", gain_dB)
    # print("Power per channel: ", power_per_channel_dBm, "dBm")
    num_pumps = num_only_ct_pumps
    pump_band_b = lambda2nu(1480e-9)
    pump_band_a = lambda2nu(1400e-9)
    # initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)
    # BROMAGE
    initial_pump_frequencies = np.array(lambda2nu([1447e-9, 1467e-9, 1485e-9, 1515e-9]))
    power_per_channel = dBm2watt(power_per_channel_dBm)
    power_per_pump = dBm2watt(-10)
    signal_wavelengths = wdm.wavelength_grid()
    pump_wavelengths = nu2lambda(initial_pump_frequencies)
    num_pumps = len(pump_wavelengths)
    signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
    pump_powers = np.ones_like(pump_wavelengths) * power_per_pump
    torch_amplifier_ct = MMFRamanAmplifier(
        fiber_length,
        integration_steps,
        num_pumps,
        signal_wavelengths,
        power_per_channel,
        fiber,
    )

    optimizer = GainOptimizer(
        torch_amplifier_ct,
        torch.from_numpy(pump_wavelengths),
        torch.from_numpy(pump_powers),
    )

    target_spectrum = watt2dBm(signal_powers) + gain_dB
    target_spectrum = target_spectrum[:, None].repeat(num_modes, axis=1)
    print(np.shape(target_spectrum))
    if power_per_channel > -6.0:
        learning_rate = 1e-4
    else:
        learning_rate = 1e-3

    pump_wavelengths, pump_powers = optimizer.optimize(
        target_spectrum=target_spectrum,
        epochs=500,
        learning_rate=learning_rate,
        lock_wavelengths=200,
    )
    np.save(optimization_result_path_ct + "opt_wavelengths_ct" +
            str(power_per_channel_dBm) + "_opt_gain_" + str(gain_dB) + ".npy", pump_wavelengths)
    np.save(optimization_result_path_ct + "opt_powers_ct" +
            str(power_per_channel_dBm) + "_opt_gain_" + str(gain_dB) + ".npy", pump_powers)

    amplifier = NumpyRamanAmplifier(fiber)

    print("============ results ==============")
    print("Pump powers")
    print(pump_powers)
    print("Pump frequency")
    print(lambda2nu(pump_wavelengths))
    print("=========== end ===================")
    
    pump_solution, signal_solution, ase_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers,
        pump_wavelengths,
        z_max,
        pump_direction=-1,
        use_power_at_fiber_start=True,
        reference_bandwidth=ref_bandwidth
    )
    
    np.save(results_path_ct + "pump_solution_ct_power" + str(power_per_channel_dBm) +
            "_opt_gain_" + str(gain_dB) + ".npy", pump_solution)
    np.save(results_path_ct + "signal_solution_ct_" + str(power_per_channel_dBm) +
            "_opt_gain_" + str(gain_dB) + ".npy", signal_solution)
    np.save(results_path_ct + "ase_solution_ct_" + str(power_per_channel_dBm) +
            "_opt_gain_" + str(gain_dB) + ".npy", ase_solution)
    print(pump_solution[-1])
    return

ct_solver(-30.0)