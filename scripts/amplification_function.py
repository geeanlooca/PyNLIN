
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
    default=70,
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


interfering_grid_index = 10 - 1
# compute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
channel_spacing = interfering_frequency - frequency_of_interest
partial_collision_margin = 5
points_per_collision = 10

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

# OPTIMIZER =================================
scheme = "co"

if scheme == "co":
    num_pumps = 8
    pump_band_b = lambda2nu(1510e-9)
    pump_band_a = lambda2nu(1410e-9)
    initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

    power_per_channel = dBm2watt(-5)
    power_per_pump = dBm2watt(-10)

    torch_amplifier = RamanAmplifier(
        fiber_length,
        integration_steps,
        num_pumps,
        signal_wavelengths,
        power_per_channel,
        fiber,
        pump_direction=1
    )

elif scheme == "cnt":
    num_pumps = 10
    pump_band_b = lambda2nu(1480e-9)
    pump_band_a = lambda2nu(1400e-9)
    initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

    power_per_channel = dBm2watt(-5)
    power_per_pump = dBm2watt(-45)

    torch_amplifier = RamanAmplifier(
        fiber_length,
        integration_steps,
        num_pumps,
        signal_wavelengths,
        power_per_channel,
        fiber,
        pump_direction=-1
    )
else:
    raise NotImplementedError(
    "Cannot yet use bidirectional pumping, not implemented!"
    )

signal_wavelengths = wdm.wavelength_grid()
pump_wavelengths = nu2lambda(initial_pump_frequencies) * 1e9
num_pumps = len(pump_wavelengths)

signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
pump_powers = np.ones_like(pump_wavelengths) * power_per_pump

optimizer = CopropagatingOptimizer(
    torch_amplifier,
    torch.from_numpy(pump_wavelengths),
    torch.from_numpy(pump_powers),
)

target_spectrum = watt2dBm(0.5 * signal_powers)
pump_wavelengths, pump_powers = optimizer.optimize(
    target_spectrum=target_spectrum,
    epochs=500,
    learning_rate=1e-3,
    lock_wavelengths=150,
)
amplifier = NumpyRamanAmplifier(fiber)

if scheme == "co":
    pump_solution, signal_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers_co,
        pump_wavelengths_co,
        z_max,
        pump_direction=-1,
    )

    np.save("pump_solution_co.npy", pump_solution_co)
    np.save("signal_solution_co.npy", signal_solution_co)
elif scheme == "cnt":
    pump_solution, signal_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers_co,
        pump_wavelengths_co,
        z_max,
        pump_direction=-1,
        use_power_at_fiber_start=True,
    )

    pump_solution_co = np.load("./pump_solution_cnt.npy")
    signal_solution_co = np.load("./signal_solution_cnt.npy")
else:
    raise NotImplementedError(
    "Cannot yet use bidirectional pumping, not implemented!"
    )