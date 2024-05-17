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

import numpy as np
# import torch
from scipy.constants import lambda2nu, nu2lambda

# from pynlin.raman.pytorch.gain_optimizer import GainOptimizer
from pynlin.raman.solvers import MMFRamanAmplifier
# from pynlin.raman.solvers import RamanAmplifier
from pynlin.utils import dBm2watt, watt2dBm
import pynlin.constellations
# from random import shuffle
import json


f = open("./scripts/sim_config.json")
data = json.load(f)
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
oi_fit = np.load('oi_fit.npy')
oi_max = np.load('oi_max.npy')
oi_min = np.load('oi_min.npy')

# Manual configuration
power_per_channel_dBm_list = power_dBm_list
# Pumping scheme choice
pumping_schemes = ['co']
num_only_co_pumps = 4
num_only_ct_pumps = 4
optimize = True
profiles = True

num_modes = 4

beta2 = pynlin.utils.dispersion_to_beta2(
	dispersion, wavelength
)
ref_bandwidth = baud_rate

fiber = pynlin.fiber.MMFiber(
	effective_area=80e-12,
	beta2=beta2,
	modes=num_modes,
	overlap_integrals=None, 
  overlap_integrals_avg=oi_min
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
partial_collision_margin = 5
points_per_collision = 10

fiber_length = fiber_lengths[0]
print("Warning: selecting only the first fiber length")
gain_dB = gain_dB_list[0]
print("Warning: selecting only the first gain")

##  Waves parameters ===========================
num_pumps = num_only_ct_pumps

# --- uniform pump frequencies
pump_band_b = lambda2nu(1480e-9)
pump_band_a = lambda2nu(1400e-9)
initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

# --- BROMAGE pump frequencies
initial_pump_frequencies = np.array(
	lambda2nu([1447e-9, 1467e-9, 1485e-9, 1515e-9]))

# --- uniform power
power_per_pump = dBm2watt(10)
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
	counterpumping = True,
	reference_bandwidth=ref_bandwidth
)
print(np.shape(pump_solution))

### fixed mode
plt.clf()
cmap = viridis
z_plot = np.linspace(0, fiber_length, len(pump_solution[:, 0, 0]))[:, None] * 1e-3
print(pump_solution)
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
plt.show()
