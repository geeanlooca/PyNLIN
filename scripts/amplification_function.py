
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
    default=50,
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
    default=90,
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


interfering_grid_index = 10
# compute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
channel_spacing = interfering_frequency - frequency_of_interest
partial_collision_margin = 5
points_per_collision = 10
# PRECISION REQUIREMENTS ESTIMATION =================================
max_num_collisions = len(pynlin.nlin.get_m_values(
    fiber,
    fiber_length,
    channel_spacing,
    1 / baud_rate,
    partial_collisions_start=partial_collision_margin,
    partial_collisions_end=partial_collision_margin)
)
integration_steps = max_num_collisions * points_per_collision

z_max = np.linspace(0, fiber_length, integration_steps)
# COMPUTATION OF TIME INTEGRALS =================================

z, I, m = pynlin.nlin.compute_all_collisions_X0mm_time_integrals(
    frequency_of_interest,
    interfering_frequency,
    baud_rate,
    fiber,
    fiber_length,
    rolloff_factor=0.1,
    samples_per_symbol=10,
    points_per_collision=points_per_collision,
    use_multiprocessing=True,
    partial_collisions_start=partial_collision_margin,
    partial_collisions_end=partial_collision_margin,
)

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


print("z axis")
print(z)
print(np.shape(z))

# OPTIMIZER =================================
'''
num_pumps = 8
pump_band_b = lambda2nu(1510e-9)
pump_band_a = lambda2nu(1410e-9)
initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

power_per_channel = dBm2watt(-5)
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
    torch_amplifier,
    torch.from_numpy(pump_wavelengths),
    torch.from_numpy(pump_powers),
)

target_spectrum = watt2dBm(0.5 * signal_powers)

pump_wavelengths_co, pump_powers_co = optimizer.optimize(
    target_spectrum=target_spectrum, epochs=500
)

amplifier = NumpyRamanAmplifier(fiber)

pump_solution_co, signal_solution_co = amplifier.solve(
    signal_powers,
    signal_wavelengths,
    pump_powers_co,
    pump_wavelengths_co,
    z_max
)

plt.figure()
plt.plot(z_max, watt2dBm(pump_solution_co), color="red", label="pump solution")
plt.plot(z_max, watt2dBm(signal_solution_co), color="black", label="signal solution")
plt.xlabel("Position [m]")
plt.ylabel("Power [dBm]")

plt.figure()
plt.stem(lambda2nu(pump_wavelengths_co) * 1e-12,
         watt2dBm(pump_powers_co), markerfmt=".")
plt.stem(
    lambda2nu(signal_wavelengths) * 1e-12,
    watt2dBm(signal_solution_co[-1]),
    markerfmt=".",
)

plt.figure()
plt.plot(signal_wavelengths * 1e9, watt2dBm(signal_solution_co[-1]))
plt.plot(signal_wavelengths * 1e9, target_spectrum)
plt.show()

np.save("pump_solution_co.npy", pump_solution_co)
np.save("signal_solution_co.npy", signal_solution_co)
'''
pump_solution_co = np.load("./pump_solution_co.npy")
signal_solution_co = np.load("./signal_solution_co.npy")

# XPM COEFFICIENT EVALUATION =================================

fig, ax = plt.subplots()
wdm.plot(ax, xaxis="frequency")

# use the approximation given in the paper ~= 1/beta2 * Omega and compare it to the numerical results
approximation = np.ones_like(m, dtype=float)
approximation *= 1 / (beta2 * 2 * np.pi * channel_spacing)

# interpolate the amplification function using optimization results
fA = interp1d(z_max, signal_solution_co[:, interfering_grid_index], kind='linear')

# compute the X0mm coefficients given the precompute time integrals:
# bonus of this approach -> no need to recompute the time integrals if
# we want to compare different amplification schemes or constellations
X0mm = pynlin.nlin.Xhkm_precomputed(
    z, I, amplification_function=fA(z))
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
