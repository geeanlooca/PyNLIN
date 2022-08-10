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

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '26'

# PLOTTING PARAMETERS
interfering_grid_index = 1
#power_dBm_list = [-20, -10, -5, 0]
power_dBm_list = np.linspace(-20, 0, 11)
arity_list = [16]
coi_list = [0, 9, 19, 29, 39, 49]

wavelength = 1550
baud_rate = 10
dispersion = 18
fiber_length = 70 * 1e3
channel_spacing = 100
num_channels = 50
baud_rate = baud_rate * 1e9

beta2 = pynlin.utils.dispersion_to_beta2(
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

Delta_theta_2_co = np.zeros_like(
    np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(arity_list)))
)
Delta_theta_2_cnt = np.zeros_like(Delta_theta_2_co)
Delta_theta_2_none =  np.zeros_like(Delta_theta_2_co)
show_flag = False

results_path = '../results/'

f_0_9 = h5py.File(results_path + '0_9_results.h5', 'r')
f_19_29 = h5py.File(results_path + '19_29_results.h5', 'r')
f_39_49 = h5py.File(results_path + '39_49_results.h5', 'r')
# print(np.array(f_39_49['/time_integrals/channel_1/interfering_channel_2/m']))
# print(np.array(f_19_29['/time_integrals/channel_1/interfering_channel_1/m']))
# print(np.array(f_39_49['/time_integrals/channel_1/interfering_channel_1/integrals']))

# f = h5py.File('merged_time_integrals.h5', 'w')
# f.create_group('time_integrals')
# h5py.copy(f_0_9['time_integrals'], f)
# h5py.copy(f_19_29['time_integrals'], f)
# h5py.copy(f_39_49['time_integrals'], f)
if False:
    # sum of all the varianced (variances are the sum of all the X0mm over m)
    X_co = np.zeros_like(
        np.ndarray(shape=(len(coi_list), len(power_dBm_list)))
    )
    X_cnt = np.zeros_like(
        np.ndarray(shape=(len(coi_list), len(power_dBm_list)))
    )
    X_none = np.zeros_like(
        np.ndarray(shape=(len(coi_list), len(power_dBm_list)))
    )

    for pow_idx, power_dBm in enumerate(power_dBm_list):
        print("Computing power ", power_dBm)
        average_power = dBm2watt(power_dBm)
        # SIMULATION DATA LOAD =================================

        pump_solution_cnt = np.load(
            results_path + 'pump_solution_cnt_' + str(power_dBm) + '.npy')
        signal_solution_cnt = np.load(
            results_path + 'signal_solution_cnt_' + str(power_dBm) + '.npy')
        signal_solution_co = np.load(
            results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
        pump_solution_co = np.load(
            results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')

        # compute fB squaring
        pump_solution_cnt = np.power(np.divide(pump_solution_cnt, pump_solution_cnt[0, :]), 2)
        signal_solution_cnt = np.power(np.divide(signal_solution_cnt, signal_solution_cnt[0, :]), 2)
        signal_solution_co = np.power(np.divide(signal_solution_co, signal_solution_co[0, :]), 2)
        pump_solution_co = np.power(np.divide(pump_solution_co, pump_solution_co[0, :]), 2)

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
                #print(np.shape(I))
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

                X0mm_none = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=None)
                X_none[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_none)**2))
                #print(X_co)
                #print(X_cnt)
                #print(X_none)

    np.save("X_co.npy", X_co)
    np.save("X_cnt.npy", X_cnt)
    np.save("X_none.npy", X_none)
else:
    X_co = np.load("X_co.npy")
    X_cnt = np.load("X_cnt.npy")
    X_none = np.load("X_none.npy")
    print(X_co)
    print(X_cnt)
    print(X_none)

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
    print("X none: ", np.sum(np.abs(X_none)))
    for coi_idx, coi in enumerate(coi_list):
        Delta_theta_2_co[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[coi_idx, pow_idx])
        Delta_theta_2_cnt[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_cnt[coi_idx, pow_idx])
        Delta_theta_2_none[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_none[coi_idx, pow_idx])

    # print("delta co: ", Delta_theta_2_co)
    # print("delta cnt: ", Delta_theta_2_cnt)
    # print("delta none: ", Delta_theta_2_none)

markers = ["x", "+", "o", "o", "x", "+"]

fig_power, (ax1, ax2, ax3) = plt.subplots(nrows= 3, sharex = True, figsize=(10, 10))
plt.plot(show=True)
coi_selection = [0, 19, 49]
for coi_idx, coi in enumerate(coi_selection):
    ax1.semilogy(power_dBm_list, Delta_theta_2_co[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='green', label="ch." + str(coi) + " co.")
    ax2.semilogy(power_dBm_list, Delta_theta_2_cnt[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='blue', label="ch." + str(coi) + " count.")
    ax3.semilogy(power_dBm_list, Delta_theta_2_none[coi_idx, :, ar_idx], marker=markers[coi_idx],
                markersize=10, color='grey', label="ch." + str(coi+1))
ax1.grid(which="both")
plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")
plt.xlabel(r"Power [dBm]")
ax1.set_ylabel(r"$\Delta \theta^2$")
ax2.set_ylabel(r"$\Delta \theta^2$")
ax3.set_ylabel(r"$\Delta \theta^2$")
ax3.legend()
plt.minorticks_on()
plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_power.savefig("power_noise_single.pdf")


pow_idx = np.where(power_dBm_list==-10)[0]

fig_channel, (ax1, ax2, ax3) = plt.subplots(nrows= 3, sharex = True, figsize=(10, 7))
plt.plot(show=True)
ax1.semilogy(coi_list, Delta_theta_2_co[:, pow_idx, ar_idx], marker='s', markersize=10, color='green', label="ch." + str(coi) + "co.")
plt.grid(which="both")
ax2.semilogy(coi_list, Delta_theta_2_cnt[:, pow_idx, ar_idx], marker='s', markersize=10, color='blue', label="ch." + str(coi) + "count.")
plt.grid(which="both")
ax3.semilogy(coi_list, Delta_theta_2_none[:, pow_idx, ar_idx], marker='s', markersize=10, color='grey', label="ch." + str(coi) + "perf.")
plt.xlabel(r"Channel index")
plt.xticks(ticks=coi_list, labels=[k+1 for k in coi_list])
ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")
ax1.set_ylabel(r"$\Delta \theta^2$")
ax2.set_ylabel(r"$\Delta \theta^2$")
ax3.set_ylabel(r"$\Delta \theta^2$")
plt.subplots_adjust(left = 0.2, wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_channel.savefig("channel_noise_single.pdf")