import argparse
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from itertools import product

from scipy.constants import lambda2nu

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
    default=100,
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

fiber = pynlin.fiber.Fiber(effective_area=80e-12, beta2=beta2)
wdm = pynlin.wdm.WDM(
    spacing=args.channel_spacing, num_channels=num_channels, center_frequency=190
)

fig, ax = plt.subplots()
wdm.plot(ax, xaxis="frequency")


# compute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[2]
channel_spacing = interfering_frequency - frequency_of_interest

# pynlin.nlin.X0mm_time_integral_WDM_grid(
#     baud_rate,
#     wdm,
#     fiber,
#     fiber_length,
#     "results.h5",
#     rolloff_factor=0.1,
#     samples_per_symbol=10,
#     points_per_collision=10,
#     use_multiprocessing=True,
#     partial_collisions_start=5,
#     partial_collisions_end=5,
# )


z, I, m = pynlin.nlin.compute_all_collisions_X0mm_time_integrals(
    frequency_of_interest,
    interfering_frequency,
    baud_rate,
    fiber,
    fiber_length,
    pulse_shape = "RaisedCosine",
    rolloff_factor=0.1,
    samples_per_symbol=10,
    points_per_collision=10,
    use_multiprocessing=True,
    partial_collisions_start=5,
    partial_collisions_end=5,
)

# use the approximation given in the paper ~= 1/beta2 * Omega and compare it to the numerical results
approximation = np.ones_like(m, dtype=float)
approximation *= 1 / (beta2 * 2 * np.pi * channel_spacing)

# compute the X0mm coefficients given the precompute time integrals:
# bonus of this approach -> no need to recompute the time integrals if
# we want to compare different amplification schemes or constellations
X0mm = pynlin.nlin.Xhkm_precomputed(z, I, amplification_function=None)
locs = pynlin.nlin.get_collision_location(m, fiber, channel_spacing, 1 / baud_rate)


plt.figure(figsize=(10, 5))
for i, m_ in enumerate(m):
    plt.plot(z * 1e-3, np.abs(I[i]), color="black")
    plt.axvline(locs[i] * 1e-3, color="red", linestyle="dashed")
plt.xlabel("Position [km]")

plt.figure()
plt.semilogy(m, np.abs(X0mm), marker="o", label="Numerical")
plt.semilogy(m, np.abs(approximation), marker="x", label=r"$1/(\beta_2 \Omega)$")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel(r"Collision index $m$")
plt.ylabel(r"$X_{0,m,m}$")
plt.title(
    rf"$f_B(z)=1$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
)
plt.legend()

plt.show()