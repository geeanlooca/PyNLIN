import json
import pynlin.constellations
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.raman.solvers import MMFRamanAmplifier
from scipy.constants import lambda2nu, nu2lambda
import numpy as np
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.fiber
import matplotlib.pyplot as plt
from matplotlib.cm import inferno
from matplotlib.cm import viridis
from pynlin.utils import oi_polynomial_expansion, oi_law_fit
cmap = inferno

plt.rcParams.update({
    "text.usetex": False,
})

# from multiprocessing import Pool

# import torch

# from pynlin.raman.pytorch.gain_optimizer import GainOptimizer
# from pynlin.raman.solvers import RamanAmplifier
# from random import shuffle


def get_final_signals(
    max_oi=True,
    use_average=False,
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
    num_modes = data['num_modes']
    oi_fit = np.load('oi_fit.npy')
    oi_max = np.load('oi_max.npy')
    oi_min = np.load('oi_min.npy')

    beta2 = pynlin.utils.dispersion_to_beta2(
        dispersion, wavelength
    )
    ref_bandwidth = baud_rate

    if use_average:
      if max_oi:
          oi_avg = oi_max
      else:
          oi_avg = oi_min
      oi_fit = None
    else:
      oi_avg = None
    
    fiber = pynlin.fiber.MMFiber(
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

    fiber_length = fiber_lengths[0]
    print("Warning: selecting only the first fiber length and gain")

    # Waves parameters ===========================
    # --- JLT-NLIN-1
    initial_pump_frequencies = np.array(
        lambda2nu([1429e-9, 1465e-9, 1488e-9, 1514e-9]))
    # initial_pump_frequencies = np.array(
        # lambda2nu([1449e-9]))
    
    # --- JLT-NLIN-1 power
    # warn: these are the powers at the fiber end
    pump_powers = dBm2watt(np.array([19.6-22, 17.3-22, 19.6-22, 14.5-22]))
    num_pumps = 4
    initial_pump_frequencies = initial_pump_frequencies[:num_pumps]
    pump_powers = pump_powers[:num_pumps][:, None].repeat(num_modes, axis=1)
    power_per_channel = dBm2watt(-30)

    # PRECISION REQUIREMENTS ESTIMATION =================================
    # Suggestion: 100m step is sufficient
    dz = 100
    integration_steps = int(np.ceil(fiber_length / dz))
    z_max = np.linspace(0, fiber_length, integration_steps)
    np.save("z_max.npy", z_max)

    signal_wavelengths = wdm.wavelength_grid()
    pump_wavelengths = nu2lambda(initial_pump_frequencies)
    signal_powers = np.ones((len(signal_wavelengths), num_modes)) * power_per_channel

    amplifier = MMFRamanAmplifier(fiber)
    # print("________ parameters")
    # print(signal_wavelengths)
    # print(watt2dBm(signal_powers))
    # print(pump_wavelengths)
    # print(watt2dBm(pump_powers))
    # print("________ end parameters")
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
    z_plot = np.linspace(0, fiber_length, len(pump_solution[:, 0, 0]))[:, None] * 1e-3
    for i in range(num_pumps):
        if i == 1:
            plt.plot(z_plot,
                     watt2dBm(pump_solution[:, i, :]), label="pump", color=cmap(i / num_pumps), ls="--", lw=0.5)
        else:
            plt.plot(z_plot,
                     watt2dBm(pump_solution[:, i, :]), color=cmap(i / num_pumps), ls="--", lw=0.5)
    for i in range(num_channels):
        if i == 1:
            plt.plot(z_plot, watt2dBm(signal_solution[:, i, :]), color=cmap(
                i / num_channels), label="signal", lw=0.5)
        else:
            plt.plot(z_plot,
                     watt2dBm(signal_solution[:, i, :]), color=cmap(i / num_channels), lw=0.5)
    plt.legend()
    plt.grid()
    plt.savefig("media/max.pdf" if max_oi else "media/min.pdf")
    plt.clf()
    return signal_solution[-1, :, :]


max_min = []
max_min.append(get_final_signals(use_average=False))
num_channels = len(max_min[0][:, 0])
num_modes = len(max_min[0][0, :])
plt.clf()
for imod in range(num_modes):
    plt.plot(range(num_channels), watt2dBm(
        max_min[0][:, imod]), color=cmap(imod / num_modes), label="fit", lw=0.8)
    # plt.plot(range(num_channels), watt2dBm(
    #     max_min[1][:, imod]), color=cmap(imod / num_modes), ls="--", label="min", lw=0.8)
    # plt.plot(range(num_channels), watt2dBm(
    #     max_min[2][:, imod]), color=cmap(imod / num_modes), ls=":", label="fit", lw=0.8)
plt.legend()
plt.grid()
plt.ylabel("Output power [dBm]")
plt.xlabel("Channel number")
plt.savefig("media/LP01.pdf")
    # plt.plot(range(num_channels), watt2dBm(
    #     max_min[2][:, imod]), color=cmap(imod / num_channels), ls=":", label="fit" if imod==1 else None, lw=0.8)
for i in range(0):
  plt.clf()
  plt.fill_between(range(num_channels), np.max(watt2dBm(
        max_min[i][:, :]), axis=1), np.min(watt2dBm(
        max_min[i][:, :]), axis=1), color=cmap(i / 3), ls="--", label=["max", "min", "fit"][i], lw=0.8, alpha=0.2)
  
  plt.legend()
  plt.grid()
  plt.ylabel("Output power [dBm]")
  plt.xlabel("Channel number")
  plt.savefig("media/MGD"+str(i)+".pdf")