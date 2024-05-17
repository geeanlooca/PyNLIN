import json
import pynlin.constellations
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.raman.solvers import MMFRamanAmplifier
from scipy.constants import lambda2nu, nu2lambda
import numpy as np
import os
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.fiber
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

plt.rcParams.update({
    "text.usetex": False,
})

# from multiprocessing import Pool

# import torch

# from pynlin.raman.pytorch.gain_optimizer import GainOptimizer
# from pynlin.raman.solvers import RamanAmplifier
# from random import shuffle


def get_final_signals(
    max_oi=True
):

    f = open("./scripts/sim_config.json")
    data = json.load(f)
    dispersion = data["dispersion"]
    baud_rate = data["baud_rate"]
    fiber_lengths = data["fiber_length"]
    num_channels = data["num_channels"]
    interfering_grid_index = data["interfering_grid_index"]
    channel_spacing = data["channel_spacing"]
    wavelength = data["wavelength"]
    num_only_ct_pumps = data['num_only_ct_pumps']
    gain_dB_setup = data['gain_dB_list']
    gain_dB_list = np.linspace(gain_dB_setup[0], gain_dB_setup[1], gain_dB_setup[2])
    power_dBm_setup = data['power_dBm_list']
    num_modes = data['num_modes']
    power_dBm_list = np.linspace(
        power_dBm_setup[0], power_dBm_setup[1], power_dBm_setup[2])
    oi_fit = np.load('oi_fit.npy')
    oi_max = np.load('oi_max.npy')
    oi_min = np.load('oi_min.npy')

    num_only_ct_pumps = 4

    beta2 = pynlin.utils.dispersion_to_beta2(
        dispersion, wavelength
    )
    ref_bandwidth = baud_rate

    if max_oi:
        actual_oi = oi_max
    else:
        actual_oi = oi_min

    fiber = pynlin.fiber.MMFiber(
        effective_area=80e-12,
        beta2=beta2,
        modes=num_modes,
        overlap_integrals=None,
        overlap_integrals_avg=actual_oi
    )
    wdm = pynlin.wdm.WDM(
        spacing=channel_spacing,
        num_channels=num_channels,
        center_frequency=190e12
    )

    # print(np.shape(fiber.overlap_integrals))
    # comute the collisions between the two furthest WDM channels
    frequency_of_interest = wdm.frequency_grid()[0]
    interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
    channel_spacing = interfering_frequency - frequency_of_interest

    fiber_length = fiber_lengths[0]
    print("Warning: selecting only the first fiber length")
    print("Warning: selecting only the first gain")

    # Waves parameters ===========================
    num_pumps = num_only_ct_pumps

    # --- uniform pump frequencies
    pump_band_b = lambda2nu(1480e-9)
    pump_band_a = lambda2nu(1400e-9)
    initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

    # --- BROMAGE pump frequencies
    initial_pump_frequencies = np.array(
        lambda2nu([1447e-9, 1467e-9, 1485e-9, 1515e-9]))

    # --- uniform power
    power_per_pump = dBm2watt(-20)
    pump_powers = np.ones((len(initial_pump_frequencies), num_modes)) * power_per_pump

    # --- JLT-NLIN-1
    initial_pump_frequencies = np.array(
        lambda2nu([1449e-9, 1465e-9, 1488e-9, 1514e-9]))

    # --- JLT-NLIN-1 power
    # warn: these are the powers at the fiber end
    # pump_powers = dBm2watt(np.array([19.6, 17.3, 19.6, 14.5]))[:, None].repeat(num_modes, axis=1)

    power_per_channel = dBm2watt(-30)

    # PRECISION REQUIREMENTS ESTIMATION =================================
    # Suggestion: 100m step is sufficient
    dz = 100
    integration_steps = int(np.ceil(fiber_length / dz))
    z_max = np.linspace(0, fiber_length, integration_steps)
    print(z_max)
    np.save("z_max.npy", z_max)

    signal_wavelengths = wdm.wavelength_grid()
    pump_wavelengths = nu2lambda(initial_pump_frequencies)
    signal_powers = np.ones((len(signal_wavelengths), num_modes)) * power_per_channel

    amplifier = MMFRamanAmplifier(fiber)
    print("________ parameters")
    print(signal_wavelengths)
    print(watt2dBm(signal_powers))
    print(pump_wavelengths)
    print(watt2dBm(pump_powers))
    print("________ end parameters")
    pump_solution, signal_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers,
        pump_wavelengths,
        z_max,
        fiber,
        counterpumping=True,
        reference_bandwidth=ref_bandwidth
    )
    # fixed mode
    plt.clf()
    cmap = viridis
    z_plot = np.linspace(0, fiber_length, len(pump_solution[:, 0, 0]))[:, None] * 1e-3
    for i in range(num_pumps):
        if i == 1:
            plt.plot(z_plot,
                     watt2dBm(pump_solution[:, i, :]), label="pump", color=cmap(i / num_pumps), ls="--")
        else:
            plt.plot(z_plot,
                     watt2dBm(pump_solution[:, i, :]), color=cmap(i / num_pumps), ls="--")
    for i in range(num_channels):
        if i == 1:
            plt.plot(z_plot, watt2dBm(signal_solution[:, i, :]), color=cmap(
                i / num_channels), label="signal")
        else:
            plt.plot(z_plot,
                     watt2dBm(signal_solution[:, i, :]), color=cmap(i / num_channels))
    plt.legend()
    plt.grid()
    plt.savefig("media/profile.pdf")
    plt.clf()
    return signal_solution[-1, :, :]


max_min = []
max_min.append(get_final_signals(max_oi=True))
max_min.append(get_final_signals(max_oi=False))
num_channels = len(max_min[0][:, 0])
num_modes = len(max_min[0][0, :])
print("num_channels = ", num_channels)
cmap = viridis
for imod in range(num_modes):
    plt.plot(range(num_channels), watt2dBm(
        max_min[0][:, imod]), color=cmap(imod / num_channels), label="maximum_oi")
    plt.plot(range(num_channels), watt2dBm(
        max_min[1][:, imod]), color=cmap(imod / num_channels), label="minimum_oi")

plt.legend()
plt.grid()
plt.savefig("media/spectra_comparison.pdf")
