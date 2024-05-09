import os
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.fiber
import matplotlib.pyplot as plt
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
oi = np.load('oi_fit.npy')
print(np.shape(oi))

# Manual configuration
power_per_channel_dBm_list = power_dBm_list
# Pumping scheme choice
pumping_schemes = ['co']
num_only_co_pumps = 4
num_only_ct_pumps = 4
optimize = True
profiles = True

###########################################
#  COMPUTATION OF AMPLITUDE FUNCTIONS
###########################################
num_modes = 4
beta2 = pynlin.utils.dispersion_to_beta2(
	dispersion * 1e-12 / (1e-9 * 1e3), wavelength
)
ref_bandwidth = baud_rate

fiber = pynlin.fiber.MMFiber(
	effective_area=80e-12,
	beta2=beta2,
	modes=num_modes,
	overlap_integrals=oi
)
wdm = pynlin.wdm.WDM(
	spacing=channel_spacing * 1e-9,
	num_channels=num_channels,
	center_frequency=190
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

num_pumps = num_only_ct_pumps

# --- uniform pump frequencies
pump_band_b = lambda2nu(1480e-9)
pump_band_a = lambda2nu(1400e-9)
initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)
# --- BROMAGE pump frequencies
initial_pump_frequencies = np.array(
	lambda2nu([1447e-9, 1467e-9, 1485e-9, 1515e-9]))

power_per_channel = dBm2watt(-30)
power_per_pump = dBm2watt(-10)
signal_wavelengths = wdm.wavelength_grid()
pump_wavelengths = nu2lambda(initial_pump_frequencies)
num_pumps = len(pump_wavelengths)
signal_powers = np.ones((len(signal_wavelengths), num_modes)) * power_per_channel
pump_powers = np.ones((len(pump_wavelengths), num_modes)) * power_per_channel * 10


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
	# pump_direction=-1,
	# use_power_at_fiber_start=True,
	reference_bandwidth=ref_bandwidth
)
print(np.shape(pump_solution))
print(np.shape(signal_solution))

### fixed mode
plt.clf()
for i in range(1):
  plt.plot(np.linspace(0, fiber_length, 500)*1e-3, watt2dBm(signal_solution[:, i*10, 1]), label="sign")
  plt.plot(np.linspace(0, fiber_length, 500)*1e-3, watt2dBm(pump_solution[:, i, :]), label="pump")
plt.legend()
plt.show()