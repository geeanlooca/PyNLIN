import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math
import os
from scipy.interpolate import interp1d

import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
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
    default=2,
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
interfering_grid_index = 1
# compute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
channel_spacing = interfering_frequency - frequency_of_interest
partial_collision_margin = 5
points_per_collision = 10


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

# SIMULATION DATA LOAD =================================
pump_solution_cnt = np.load('scripts/light_pump_solution_cnt.npy')
signal_solution_cnt = np.load('scripts/light_signal_solution_cnt.npy')
signal_solution_co = np.load('scripts/light_signal_solution_co.npy')
pump_solution_co = np.load('scripts/light_pump_solution_co.npy')


f = h5py.File('scripts/light_results_2.h5', 'r')
print(f)
z_max = np.linspace(0, fiber_length, np.shape(pump_solution_cnt)[0])

# XPM COEFFICIENT EVALUATION, single m =================================

interfering_grid_index = 1
single_interference_channel_spacing = channel_spacing * interfering_grid_index
fig, ax = plt.subplots()
wdm.plot(ax, xaxis="frequency")


# interpolate the amplification function using optimization results
fB = interp1d(z_max, signal_solution_co[:, interfering_grid_index], kind='linear')

# compute the X0mm coefficients given the precompute time integrals:
# bonus of this approach -> no need to recompute the time integrals if
# we want to compare different amplification schemes or constellations
m = np.array(f['/time_integrals/channel_0/interfering_channel_0/m'])
z = np.array(f['/time_integrals/channel_0/interfering_channel_0/z'])
I = np.array(f['/time_integrals/channel_0/interfering_channel_0/integrals'])
print(m)
print(z)
print(I)
X0mm = pynlin.nlin.Xhkm_precomputed(
    z, I, amplification_function=fB(z))
locs = pynlin.nlin.get_collision_location(
    m, fiber, single_interference_channel_spacing, 1 / baud_rate)
print(locs)

plt.figure(figsize=(10, 5))
for i, m_ in enumerate(m):
    plt.plot(z*1e-3, np.abs(I[i]), color="black")
    plt.axvline(locs[i] * 1e-3, color="red", linestyle="dashed")
plt.xlabel("Position [km]")

plt.figure()
plt.semilogy(m, np.abs(X0mm), marker="o", label="Numerical")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel(r"Collision index $m$")
plt.ylabel(r"$X_{0,m,m}$")
plt.title(
    rf"$f_B(z)=1$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
)
plt.legend()

plt.show()

# FULL X0mm EVALUATION FOR EVERY m =======================
X_co = []
X_cnt = []
for interfering_grid_index in range(np.shape(signal_solution_co)[1]):
    print("summing all m for interfering channel ", interfering_grid_index)
    fB_co = interp1d(
        z_max, signal_solution_co[:, interfering_grid_index], kind='linear')
    X0mm_co = pynlin.nlin.Xhkm_precomputed(
        z, I, amplification_function=fB_co(z))
    X_co.append(np.sum(X0mm_co**2))

    fB_cnt = interp1d(
        z_max, signal_solution_cnt[:, interfering_grid_index], kind='linear')
    X0mm_cnt = pynlin.nlin.Xhkm_precomputed(
        z, I, amplification_function=fB_cnt(z))
    X_cnt.append(np.sum(X0mm_cnt**2))

# PHASE NOISE COMPUTATION =======================
# copropagating
qam = pynlin.constellations.QAM(32)

qam_symbols = qam.symbols()
cardinality = len(qam_symbols)

# assign specific average optical power
power_dBm = -5
average_power =  dBm2watt(power_dBm)
qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power)
print(qam_symbols)
Delta_theta_2 = 0
print("variance: ", (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) **2) 
)
print(np.abs(X_co[1]))

for i in range(0, num_channels):
   Delta_theta_ch_2 = 4 * fiber.gamma**2 * (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) **2) * np.abs(X_co[i])
   print("Interference with channel ", i+1)
   print("\tPhase std deviation: ", (np.sqrt(Delta_theta_ch_2)))
   Delta_theta_2 += Delta_theta_ch_2

print("Total phase std deviation: ", np.sqrt(Delta_theta_2))
print("Symbol average optical power: ", (np.abs(qam_symbols)**2).mean())

# SIGNAL POWER MAY BE TOO HIGH = express b0 and g in phyisical terms

# equations are expressed in W