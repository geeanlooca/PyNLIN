
import argparse
import math
import os
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import lambda2nu, nu2lambda
from scipy.interpolate import interp1d

from pynlin.fiber import Fiber
from pynlin.raman.pytorch.gain_optimizer import CopropagatingOptimizer
from pynlin.raman.pytorch.solvers import RamanAmplifier
from pynlin.raman.solvers import RamanAmplifier as NumpyRamanAmplifier
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.wdm import WDM
import pynlin.constellations

parser = argparse.ArgumentParser()
parser.add_argument(
    "-R", "--baud-rate", default=10, help="The baud rate of each WDM channel in GHz."
)
parser.add_argument(
    "-D",
    "--dispersion",
    default=18,
    type=float,
    help="The dispersion coefficient of the fiber in ps/nm km.",
)
parser.add_argument(
    "-L",
    "--fiber-length",
    default=80,
    type=float,
    help="The length of the fiber in kilometers.",
)
parser.add_argument(
    "-c",
    "--channel-spacing",
    default=100,
    type=float,
    help="The spacing between neighboring WDM channels in GHz.",
)
parser.add_argument(
    "-C",
    "--channel-count",
    default=50,
    type=int,
    help="The number of WDM channels in the grid.",
)
parser.add_argument(
    "-M",
    "--use-multiprocessing",
    action="store_true",
    default=True,
    help="If passed, this flag enables multicore processing to compute the collisions in parallel.",
)
parser.add_argument(
    "-W",
    "--wavelength",
    default=1550,
    type=float,
    help="The wavelength at which the dispersion coefficient is given (in nanometers).",
)
args = parser.parse_args()

print(args)

beta2 = pynlin.utils.dispersion_to_beta2(
    args.dispersion * 1e-12 / (1e-9 * 1e3), args.wavelength * 1e-9
)
fiber_length = args.fiber_length * 1e3
channel_spacing = args.channel_spacing * 1e9
num_channels = args.channel_count
baud_rate = args.baud_rate * 1e9

fiber = pynlin.fiber.Fiber(
    effective_area=80e-12,
    beta2=beta2
)
wdm = pynlin.wdm.WDM(
    spacing=args.channel_spacing,
    num_channels=num_channels,
    center_frequency=190
)


interfering_grid_index = 38
# compute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
channel_spacing = interfering_frequency - frequency_of_interest
partial_collision_margin = 5
points_per_collision = 10


power_per_channel_dBm = 3
print("Power per channel: ", power_per_channel_dBm, "dBm")


'''
# PRECISION REQUIREMENTS ESTIMATION =================================
max_channel_spacing = wdm.frequency_grid()[num_channels - 1] - wdm.frequency_grid()[0]

print(max_channel_spacing)

max_num_collisions = len(pynlin.nlin.get_m_values(
    fiber,
    fiber_length,
    max_channel_spacing,
    1 / baud_rate,
    partial_collisions_start=partial_collision_margin,
    partial_collisions_end=partial_collision_margin)
)
integration_steps = max_num_collisions * points_per_collision

# building high precision
z_max = np.linspace(0, fiber_length, integration_steps)

# np.save("z_max.npy", z_max)

# OPTIMIZER CO =================================
####### POWER FIXED TO -5dBm

num_pumps = 8
pump_band_b = lambda2nu(1510e-9)
pump_band_a = lambda2nu(1410e-9)
initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

power_per_channel = dBm2watt(power_per_channel_dBm)
power_per_pump = dBm2watt(-10)
signal_wavelengths = wdm.wavelength_grid()
pump_wavelengths = nu2lambda(initial_pump_frequencies) * 1e9
num_pumps = len(pump_wavelengths)

signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
pump_powers = np.ones_like(pump_wavelengths) * power_per_pump

torch_amplifier = RamanAmplifier(
    fiber_length,
    integration_steps,
    num_pumps,
    signal_wavelengths,
    power_per_channel,
    fiber,
)

optimizer = CopropagatingOptimizer(
    torch_amplifier,
    torch.from_numpy(pump_wavelengths),
    torch.from_numpy(pump_powers),
)

target_spectrum = watt2dBm(0.5 * signal_powers)
pump_wavelengths, pump_powers = optimizer.optimize(
    target_spectrum=target_spectrum,
    epochs=500
)
amplifier = NumpyRamanAmplifier(fiber)

pump_solution_co, signal_solution_co = amplifier.solve(
    signal_powers,
    signal_wavelengths,
    pump_powers,
    pump_wavelengths,
    z_max
)



# pump_solution_co = np.load("./pump_solution_cnt.npy")
# signal_solution_co = np.load("./signal_solution_cnt.npy")

# OPTIMIZER COUNTER =================================

num_pumps = 10
pump_band_b = lambda2nu(1480e-9)
pump_band_a = lambda2nu(1400e-9)
initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

power_per_channel = dBm2watt(power_per_channel_dBm)
power_per_pump = dBm2watt(-45)
signal_wavelengths = wdm.wavelength_grid()
pump_wavelengths = nu2lambda(initial_pump_frequencies) * 1e9
num_pumps = len(pump_wavelengths)


signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
pump_powers = np.ones_like(pump_wavelengths) * power_per_pump

torch_amplifier_cnt = RamanAmplifier(
    fiber_length,
    integration_steps,
    num_pumps,
    signal_wavelengths,
    power_per_channel,
    fiber,
    pump_direction=-1,
)

optimizer = CopropagatingOptimizer(
    torch_amplifier_cnt,
    torch.from_numpy(pump_wavelengths),
    torch.from_numpy(pump_powers),
)

target_spectrum = watt2dBm(0.5 * signal_powers)
pump_wavelengths_co, pump_powers_co = optimizer.optimize(
    target_spectrum=target_spectrum,
    epochs=500,
    learning_rate=1e-3,
    lock_wavelengths=150,
)
amplifier = NumpyRamanAmplifier(fiber)

pump_solution_cnt, signal_solution_cnt = amplifier.solve(
    signal_powers,
    signal_wavelengths,
    pump_powers_co,
    pump_wavelengths_co,
    z_max,
    pump_direction=-1,
    use_power_at_fiber_start=True,
)


np.save("pump_solution_co_"+power_per_channel_dBm+".npy", pump_solution_co)
np.save("signal_solution_co_"+power_per_channel_dBm+".npy", signal_solution_co)

np.save("pump_solution_cnt_"+power_per_channel_dBm+".npy", pump_solution_cnt)
np.save("signal_solution_cnt_"+power_per_channel_dBm+".npy", signal_solution_cnt)
'''

# COMPUTATION OF TIME INTEGRALS =================================
# to be computed once for all, for all channels, and saved to file
# using X0mm_time_integral_WDM_grid
m = pynlin.nlin.get_m_values(fiber, fiber_length, channel_spacing, 1 / baud_rate)

# print(m)
# z, I, m = pynlin.nlin.compute_all_collisions_X0mm_time_integrals(
#     frequency_of_interest,
#     interfering_frequency,
#     baud_rate,
#     fiber,
#     fiber_length,
#     pulse_shape="Nyquist",
#     rolloff_factor=0.1,
#     samples_per_symbol=10,
#     points_per_collision=points_per_collision,
#     use_multiprocessing=True,
#     partial_collisions_start=partial_collision_margin,
#     partial_collisions_end=partial_collision_margin,
# )

fiber = pynlin.fiber.Fiber(
    effective_area=80e-12,
    beta2=beta2
)
fiber_length = 500e3
channel_spacing =102
num_channels = 2
baud_rate = 100e9
print(channel_spacing)
fiber = pynlin.fiber.Fiber(
    effective_area=80e-12,
    beta2=beta2
)
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=190
) 

partial_collision_margin = 5
points_per_collision = 10

pynlin.nlin.X0mm_time_integral_WDM_grid(
    baud_rate,
    wdm,
    fiber,
    fiber_length,
    "mecozzi.h5",
    pulse_shape="Nyquist",
    rolloff_factor=0.1,
    samples_per_symbol=10,
    points_per_collision=points_per_collision,
    use_multiprocessing=True,
    partial_collisions_start=partial_collision_margin,
    partial_collisions_end=partial_collision_margin,
)
