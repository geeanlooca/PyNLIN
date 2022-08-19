import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "-L",
    "--fiber-length",
    default=80,
    type=float,
    help="The length of the fiber in kilometers.",
)
args = parser.parse_args()

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '26'

###############################
#### fiber length setup #######
###############################
length_setup = int(args.fiber_length)
fiber_length = length_setup * 1e3

num_co  = 8
num_cnt = 8
results_path = '../results_'+str(length_setup)+'/'
results_path_bi = '../results_'+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'
plot_save_path = '/home/lorenzi/Scrivania/tesi/tex/images/classical/'+str(length_setup)+'km/'
time_integrals_results_path = '../results/'


# PLOTTING PARAMETERS
interfering_grid_index = 1
#power_dBm_list = [-20, -10, -5, 0]
power_dBm_list = np.linspace(-20, 0, 11)
arity_list = [16]
coi_list = [0, 9, 19, 29, 39, 49]

wavelength = 1550
baud_rate = 10
dispersion = 18
channel_spacing = 100
num_channels = 50
baud_rate = baud_rate * 1e9

beta2 = -pynlin.utils.dispersion_to_beta2(
    dispersion * 1e-12 / (1e-9 * 1e3), wavelength * 1e-9
)
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

print("beta2: ", fiber.beta2)
print("gamma: ", fiber.gamma)
print(wdm.frequency_grid())
Delta_theta_2_co = np.zeros_like(
    np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(arity_list)))
)
Delta_theta_2_cnt = np.zeros_like(Delta_theta_2_co)
Delta_theta_2_bi = np.zeros_like(Delta_theta_2_co)
Delta_theta_2_none =  np.zeros_like(Delta_theta_2_co)

show_flag = False
compute_X0mm_space_integrals = True

if input("\nX0mm and noise variance plotter: \n\t>Length= "+str(length_setup)+"km \n\t>power list= "+str(power_dBm_list)+" \n\t>coi_list= "+str(coi_list)+"\n\t>compute_X0mm_space_integrals= "+str(compute_X0mm_space_integrals)+"\nAre you sure? (y/[n])") != "y":
    exit()

f_0_9 = h5py.File(time_integrals_results_path + '0_9_results.h5', 'r')
f_19_29 = h5py.File(time_integrals_results_path + '19_29_results.h5', 'r')
f_39_49 = h5py.File(time_integrals_results_path + '39_49_results.h5', 'r')
# print(np.array(f_39_49['/time_integrals/channel_1/interfering_channel_2/m']))
# print(np.array(f_19_29['/time_integrals/channel_1/interfering_channel_1/m']))
# print(np.array(f_39_49['/time_integrals/channel_1/interfering_channel_1/integrals']))

# f = h5py.File('merged_time_integrals.h5', 'w')
# f.create_group('time_integrals')
# h5py.copy(f_0_9['time_integrals'], f)
# h5py.copy(f_19_29['time_integrals'], f)
# h5py.copy(f_39_49['time_integrals'], f)
if compute_X0mm_space_integrals:
    # sum of all the varianced (variances are the sum of all the X0mm over m)
    X_co = np.zeros_like(
        np.ndarray(shape=(len(coi_list), len(power_dBm_list)))
    )
    X_cnt = np.zeros_like(X_co)
    X_bi = np.zeros_like(X_co)
    X_none = np.zeros_like(X_co)
    ase_co = np.zeros_like(X_co)
    ase_cnt = np.zeros_like(X_co)
    ase_bi = np.zeros_like(X_co)

    for pow_idx, power_dBm in enumerate(power_dBm_list):
        print("Computing power ", power_dBm)
        average_power = dBm2watt(power_dBm)
        # SIMULATION DATA LOAD =================================

        pump_solution_co = np.load(
            results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')
        signal_solution_co = np.load(
            results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
        pump_solution_cnt = np.load(
            results_path + 'pump_solution_cnt_' + str(power_dBm) + '.npy')
        signal_solution_cnt = np.load(
            results_path + 'signal_solution_cnt_' + str(power_dBm) + '.npy')
        pump_solution_bi = np.load(
            results_path_bi + 'pump_solution_bi_' + str(power_dBm) + '.npy')
        signal_solution_bi = np.load(
            results_path_bi + 'signal_solution_bi_' + str(power_dBm) + '.npy')
    
        # ASE power evolution
        ase_solution_co = np.load(
            results_path + 'ase_solution_co_' + str(power_dBm) + '.npy')
        ase_solution_cnt = np.load(
            results_path + 'ase_solution_cnt_' + str(power_dBm) + '.npy')
        ase_solution_bi = np.load(
            results_path_bi + 'ase_solution_bi_' + str(power_dBm) + '.npy')

        # compute fB squaring
        pump_solution_co =    np.divide(pump_solution_co, pump_solution_co[0, :])
        signal_solution_co =  np.divide(signal_solution_co, signal_solution_co[0, :])
        pump_solution_cnt =   np.divide(pump_solution_cnt, pump_solution_cnt[0, :])
        signal_solution_cnt = np.divide(signal_solution_cnt, signal_solution_cnt[0, :])
        pump_solution_bi =    np.divide(pump_solution_bi, pump_solution_bi[0, :])
        signal_solution_bi =  np.divide(signal_solution_bi, signal_solution_bi[0, :])

        #z_max = np.load(results_path + 'z_max.npy')
        #f = h5py.File(results_path + 'results_multi.h5', 'r')
        z_max = np.linspace(0, fiber_length, np.shape(pump_solution_cnt)[0])

        # compute the X0mm coefficients given the precompute time integrals
        # FULL X0mm EVALUATION FOR EVERY m =======================
        for coi_idx, coi in enumerate(coi_list):

            print("Computing Channel Of Interest ", coi + 1)

            # compute the first num_channels interferents (assume the WDM grid is identical)
            interfering_frequencies = pynlin.nlin.get_interfering_frequencies(
                coi, wdm.frequency_grid())
            pbar_description = "Computing space integrals"
            collisions_pbar = tqdm.tqdm(range(np.shape(signal_solution_co)[1])[
                                        0:num_channels - 1], leave=False)
            collisions_pbar.set_description(pbar_description)
            for incremental, interf_index in enumerate(collisions_pbar):
                #print("interfering channel : ", incremental)
                if coi == 0:
                    m = np.array(
                        f_0_9['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_0_9['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_0_9['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 9:
                    m = np.array(
                        f_0_9['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_0_9['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_0_9['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 19:
                    m = np.array(
                        f_19_29['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_19_29['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_19_29['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 29:
                    m = np.array(
                        f_19_29['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_19_29['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_19_29['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 39:
                    m = np.array(
                        f_39_49['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_39_49['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_39_49['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 49:
                    m = np.array(
                        f_39_49['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_39_49['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_39_49['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/integrals'])

                # upper cut z
                z = np.array(list(filter(lambda x: x<=fiber_length, z)))
                I = I[:int(len(m)*(fiber_length/80e3)), :len(z)]
                m = m[:int(len(m)*(fiber_length/80e3))]

                fB_co = interp1d(
                    z_max, signal_solution_co[:, incremental], kind='linear')
                X0mm_co = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=fB_co(z))
                X_co[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_co)**2))

                fB_cnt = interp1d(
                    z_max, signal_solution_cnt[:, incremental], kind='linear')
                X0mm_cnt = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=fB_cnt(z))
                X_cnt[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_cnt)**2))

                fB_bi = interp1d(
                    z_max, signal_solution_bi[:, incremental], kind='linear')
                X0mm_bi = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=fB_bi(z))
                X_bi[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_bi)**2))

                X0mm_none = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=None)
                X_none[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_none)**2))
                #print(X_co)
                #print(X_cnt)
                #print(X_none)
        print("\ncomputing channel: ", coi_idx, "\n\n")
        print(ase_solution_co[-1, coi_idx])
        ase_co[coi_idx,pow_idx] = ase_solution_co[-1, coi_idx]
        ase_cnt[coi_idx,pow_idx] = ase_solution_cnt[-1, coi_idx]
        ase_bi[coi_idx,pow_idx] = ase_solution_bi[-1, coi_idx]

    print(ase_co)
    np.save("X_co.npy", X_co)
    np.save("X_cnt.npy", X_cnt)
    np.save("X_bi.npy", X_bi)
    np.save("X_none.npy", X_none)
    np.save("ase_co.npy", ase_co)
    np.save("ase_cnt.npy", ase_cnt)
    np.save("ase_bi.npy", ase_bi)

else:
    X_co = np.load("X_co.npy")
    X_cnt = np.load("X_cnt.npy")
    X_bi = np.load("X_bi.npy")
    X_none = np.load("X_none.npy")
    ase_co = np.load("ase_co.npy")
    ase_cnt = np.load("ase_cnt.npy")
    ase_bi = np.load("ase_bi.npy")
    print(X_co)
    print(X_cnt)
    print(X_bi)
    print(X_none)
    print(ase_co)
ar_idx = 0  # 16-QAM
M = 16
for pow_idx, power_dBm in enumerate(power_dBm_list):
    average_power = dBm2watt(power_dBm)
    qam = pynlin.constellations.QAM(M)
    qam_symbols = qam.symbols()

    # assign specific average optical energy
    qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power / baud_rate)
    constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)

    print("X co: ", np.sum(np.abs(X_co)))
    print("X cnt: ", np.sum(np.abs(X_cnt)))
    print("X bi: ", np.sum(np.abs(X_bi)))
    print("X none: ", np.sum(np.abs(X_none)))

    for coi_idx, coi in enumerate(coi_list):
        Delta_theta_2_co[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[coi_idx, pow_idx])
        Delta_theta_2_cnt[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_cnt[coi_idx, pow_idx])
        Delta_theta_2_bi[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_bi[coi_idx, pow_idx])
        Delta_theta_2_none[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_none[coi_idx, pow_idx])

    # print("delta co: ", Delta_theta_2_co)
    # print("delta cnt: ", Delta_theta_2_cnt)
    # print("delta none: ", Delta_theta_2_none)

markers = ["x", "+", "o", "o", "x", "+"]

fig_power, (ax1, ax2, ax3, ax4) = plt.subplots(nrows= 4, sharex = True, figsize=(10, 10))
plt.plot(show=True)
coi_selection = [0, 19, 49]
for coi_idx, coi in enumerate(coi_selection):
    ax1.semilogy(power_dBm_list, Delta_theta_2_co[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='green', label="ch." + str(coi) + " co.")
    ax2.semilogy(power_dBm_list, Delta_theta_2_cnt[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='blue', label="ch." + str(coi) + " count.")
    ax3.semilogy(power_dBm_list, Delta_theta_2_bi[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='orange', label="ch." + str(coi+1))
    ax4.semilogy(power_dBm_list, Delta_theta_2_none[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='grey', label="ch." + str(coi+1))
ax1.grid(which="both")
#plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")

plt.xlabel(r"Power [dBm]")
ax1.set_ylabel(r"$\Delta \theta^2$")
ax2.set_ylabel(r"$\Delta \theta^2$")
ax3.set_ylabel(r"$\Delta \theta^2$")
ax4.set_ylabel(r"$\Delta \theta^2$")
ax4.legend()
plt.minorticks_on()
plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_power.savefig(plot_save_path+"power_noise.pdf")


print(ase_co)
fig_ase, (ax1, ax2, ax3) = plt.subplots(nrows= 3, sharex = True, figsize=(10, 10))
plt.plot(show=True)
coi_selection = [0, 19, 49]
for coi_idx, coi in enumerate(coi_selection):
    ax1.semilogy(power_dBm_list, ase_co[coi_idx, :], marker=markers[coi_idx],
                markersize=10, color='green', label="ch." + str(coi) + " co.")
    ax2.semilogy(power_dBm_list, ase_cnt[coi_idx, :], marker=markers[coi_idx],
                markersize=10, color='blue', label="ch." + str(coi) + " count.")
    ax3.semilogy(power_dBm_list, ase_bi[coi_idx, :], marker=markers[coi_idx],
                markersize=10, color='orange', label="ch." + str(coi+1))
ax1.grid(which="both")
#plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")

plt.xlabel(r"Power [dBm]")
ax1.set_ylabel(r"ASE power")
ax2.set_ylabel(r"ASE power")
ax3.set_ylabel(r"ASE power")
ax3.legend()
plt.minorticks_on()
plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_ase.savefig(plot_save_path+"ase_power_noise.pdf")

fig_comparison, (ax1, ax2, ax3, ax4) = plt.subplots(nrows= 4, sharex = True, figsize=(10, 10))
plt.plot(show=True)
coi_selection = [0, 19, 49]
for coi_idx, coi in enumerate(coi_selection):
    ax1.semilogy(power_dBm_list, Delta_theta_2_co[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='green', label="ch." + str(coi) + " co.")
    ax2.semilogy(power_dBm_list, Delta_theta_2_cnt[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='blue', label="ch." + str(coi) + " count.")
    ax3.semilogy(power_dBm_list, Delta_theta_2_bi[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='orange', label="ch." + str(coi+1))
    ax1.semilogy(power_dBm_list, ase_co[coi_idx, :], marker=markers[coi_idx],
                markersize=10, color='black', label="ch." + str(coi) + " co.")
    ax2.semilogy(power_dBm_list, ase_cnt[coi_idx, :], marker=markers[coi_idx],
                markersize=10, color='black', label="ch." + str(coi) + " count.")
    ax3.semilogy(power_dBm_list, ase_bi[coi_idx, :], marker=markers[coi_idx],
                markersize=10, color='black', label="ch." + str(coi+1))
ax1.grid(which="both")
#plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")

plt.xlabel(r"Power [dBm]")
ax1.set_ylabel(r"$\Delta \theta^2$")
ax2.set_ylabel(r"$\Delta \theta^2$")
ax3.set_ylabel(r"$\Delta \theta^2$")
ax4.set_ylabel(r"$\Delta \theta^2$")
ax4.legend()
plt.minorticks_on()
plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_ase.savefig(plot_save_path+"comparison.pdf")


pow_idx = np.where(power_dBm_list==-10)[0]

fig_channel, (ax1, ax2, ax3, ax4) = plt.subplots(nrows= 4, sharex = True, figsize=(12, 10))
plt.plot(show=True)
ax1.semilogy(coi_list, Delta_theta_2_co[:, pow_idx, ar_idx], marker='s', markersize=10, color='green', label="ch." + str(coi) + "co.")
plt.grid(which="both")
ax2.semilogy(coi_list, Delta_theta_2_cnt[:, pow_idx, ar_idx], marker='s', markersize=10, color='blue', label="ch." + str(coi) + "count.")
plt.grid(which="both")
ax3.semilogy(coi_list, Delta_theta_2_bi[:, pow_idx, ar_idx], marker='s', markersize=10, color='orange', label="ch." + str(coi) + "perf.")
plt.grid(which="both")
ax4.semilogy(coi_list, Delta_theta_2_none[:, pow_idx, ar_idx], marker='s', markersize=10, color='grey', label="ch." + str(coi) + "perf.")
plt.grid(which="both")
plt.xlabel(r"Channel index")
plt.xticks(ticks=coi_list, labels=[k+1 for k in coi_list])
# ax1.grid(which="both")
# ax2.grid(which="both")
# ax3.grid(which="both")
# ax4.grid(which="both")

ax1.set_ylabel(r"$\Delta \theta^2$")
ax2.set_ylabel(r"$\Delta \theta^2$")
ax4.set_ylabel(r"$\Delta \theta^2$")
ax3.set_ylabel(r"$\Delta \theta^2$")
plt.subplots_adjust(left = 0.2, wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_channel.savefig(plot_save_path+"channel_noise.pdf")


fig_ase_channel, (ax1, ax2, ax3) = plt.subplots(nrows= 3, sharex = True, figsize=(12, 10))
plt.plot(show=True)
ax1.semilogy(coi_list, ase_co[:, pow_idx], marker='s', markersize=10, color='green', label="ch." + str(coi) + "co.")
plt.grid(which="both")
ax2.semilogy(coi_list, ase_cnt[:, pow_idx], marker='s', markersize=10, color='blue', label="ch." + str(coi) + "count.")
plt.grid(which="both")
ax3.semilogy(coi_list, ase_bi[:, pow_idx], marker='s', markersize=10, color='orange', label="ch." + str(coi) + "bi.")
plt.grid(which="both")
plt.xlabel(r"Channel index")
plt.xticks(ticks=coi_list, labels=[k+1 for k in coi_list])
ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")

ax1.set_ylabel(r"$\Delta \theta^2$")
ax2.set_ylabel(r"$\Delta \theta^2$")
ax3.set_ylabel(r"$\Delta \theta^2$")
plt.subplots_adjust(left = 0.2, wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_ase_channel.savefig(plot_save_path+"ase_channel_noise.pdf")