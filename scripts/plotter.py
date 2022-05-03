import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math
import os
from scipy.interpolate import interp1d
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.fiber import Fiber
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.wdm import WDM
import pynlin.constellations

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '16'


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



interfering_grid_index = 49
partial_collision_margin = 5
points_per_collision = 10

power_dBm = -10
average_power =  dBm2watt(power_dBm)

results_path = '../results/'
print(results_path + 'light_pump_solution_cnt_' + str(power_dBm) + '.npy')
# SIMULATION DATA LOAD =================================
pump_solution_cnt =    np.load(results_path + 'pump_solution_cnt_' + str(power_dBm) + '.npy')
signal_solution_cnt =  np.load(results_path + 'signal_solution_cnt_' + str(power_dBm) + '.npy')
signal_solution_co =   np.load(results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
pump_solution_co =     np.load(results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')

z_max = np.load(results_path + 'z_max.npy')
f = h5py.File(results_path + 'results_multi.h5', 'r')
z_max = np.linspace(0, fiber_length, np.shape(pump_solution_cnt)[0])

# XPM COEFFICIENT EVALUATION, single m =================================

interfering_grid_index = 49
# compute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
single_interference_channel_spacing = interfering_frequency - frequency_of_interest



# interpolate the amplification function using optimization results
fB = interp1d(z_max, signal_solution_co[:, interfering_grid_index], kind='linear')

# compute the X0mm coefficients given the precompute time integrals:
# bonus of this approach -> no need to recompute the time integrals if
# we want to compare different amplification schemes or constellations
m = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/m'])
z = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/z'])
I = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/integrals'])

X0mm = pynlin.nlin.Xhkm_precomputed(
    z, I, amplification_function=fB(z))
locs = pynlin.nlin.get_collision_location(
    m, fiber, single_interference_channel_spacing, 1 / baud_rate)

fig0, ax = plt.subplots()
wdm.plot(ax, xaxis="frequency")

fig1 = plt.figure(figsize=(10, 5))
for i, m_ in enumerate(m):
    plt.plot(z*1e-3, np.abs(I[i]), color="black")
    plt.axvline(locs[i] * 1e-3, color="red", linestyle="dashed")
plt.xlabel("Position [km]")
fig1.savefig('fig1.pdf')


fig2 = plt.figure()
plt.semilogy(m, np.abs(X0mm), marker="o", label="Numerical")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel(r"Collision index $m$")
plt.ylabel(r"$X_{0,m,m}$ [m/s]")
# plt.title(
#     rf"$f_B(z)$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
# )
plt.legend()
fig2.savefig('fig2.png')

plt.show()

# FULL X0mm EVALUATION FOR EVERY m =======================
X_co = []
X_co.append(0.0)
X_cnt = []
X_cnt.append(0.0)

pbar_description = "Computing space integrals"
collisions_pbar = tqdm.tqdm(range(np.shape(signal_solution_co)[1]), leave=False)
collisions_pbar.set_description(pbar_description)

for interf_index in collisions_pbar:
    fB_co = interp1d(
        z_max, signal_solution_co[:, interf_index], kind='linear')
    X0mm_co = pynlin.nlin.Xhkm_precomputed(
        z, I, amplification_function=fB_co(z))
    X_co.append(np.sum(np.abs(X0mm_co)**2))

    fB_cnt = interp1d(
        z_max, signal_solution_cnt[:, interf_index], kind='linear')
    X0mm_cnt = pynlin.nlin.Xhkm_precomputed(
        z, I, amplification_function=fB_cnt(z))
    X_cnt.append(np.sum(np.abs(X0mm_cnt)**2))

# PHASE NOISE COMPUTATION =======================
# copropagating
qam = pynlin.constellations.QAM(64)

qam_symbols = qam.symbols()
cardinality = len(qam_symbols)

# assign specific average optical power

qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power/baud_rate)
fig3 = plt.plot(reuse = False, show = True)
plt.scatter(np.real(qam_symbols), np.imag(qam_symbols))
plt.show()
Delta_theta_2 = 0
constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) **2)
print("constellation energy variance: ",  constellation_variance)

print(np.abs(X_co[1]))

for i in range(1, num_channels):
   Delta_theta_ch_2 = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[i])
   Delta_theta_2 += Delta_theta_ch_2
   Delta_tot = np.sqrt(Delta_theta_ch_2)

print("Total phase VARIANCE: ", Delta_theta_2)
print("Total phase sum: ", Delta_tot)

print("\nSymbol average optical power: ", (np.abs(qam_symbols)**2 * baud_rate).mean(), "W")
print("Symbol average optical energy: ", (np.abs(qam_symbols)**2).mean(), "J")

print("Symbol average magnitude: ", (np.abs(qam_symbols)).mean(), "sqrt(W*s)")
# SIGNAL POWER MAY BE TOO HIGH = express b0 and g in phyisical terms

# equations are expressed in W