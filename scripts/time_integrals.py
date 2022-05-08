
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
from scipy.constants import lambda2nu, nu2lambda
from scipy.interpolate import interp1d

from pynlin.fiber import Fiber
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

## MECOZZI
# fiber = pynlin.fiber.Fiber(
#     effective_area=80e-12,
#     beta2=beta2
# )
# fiber_length = 500e3
# channel_spacing =102
# num_channels = 2
# baud_rate = 100e9
# print(channel_spacing)
# fiber = pynlin.fiber.Fiber(
#     effective_area=80e-12,
#     beta2=beta2
# )
# wdm = pynlin.wdm.WDM(
#     spacing=channel_spacing,
#     num_channels=num_channels,
#     center_frequency=190
# ) 

# partial_collision_margin = 5
# points_per_collision = 10

interfering_index = [0, 9]

pynlin.nlin.X0mm_time_integral_WDM_selection(
    baud_rate,
    wdm,
    interfering_index,
    fiber,
    fiber_length,
    "0_9_results.h5",
    pulse_shape="Nyquist",
    rolloff_factor=0.1,
    samples_per_symbol=10,
    points_per_collision=points_per_collision,
    use_multiprocessing=True,
    partial_collisions_start=partial_collision_margin,
    partial_collisions_end=partial_collision_margin,
)