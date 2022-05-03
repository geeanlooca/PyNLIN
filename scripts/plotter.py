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
plt.rcParams['font.size'] = '26'

# PLOTTING PARAMETERS
interfering_grid_index = 1
#power_dBm_list = [-20, -10, -5, 0]
power_dBm_list = np.linspace(-20, 0, 11)

wavelength = 1550
baud_rate = 10
dispersion = 18
fiber_length = 80 * 1e3
channel_spacing = 100
num_channels = 2
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

Delta_theta_2_co = np.zeros(len(power_dBm_list))
Delta_theta_2_cnt = np.zeros(len(power_dBm_list))

show_flag = False

for idx, power_dBm in enumerate(power_dBm_list):
    average_power =  dBm2watt(power_dBm)

    # SIMULATION DATA LOAD =================================
    results_path = '../results/'

    pump_solution_cnt =    np.load(results_path + 'pump_solution_cnt_' + str(power_dBm) + '.npy')
    signal_solution_cnt =  np.load(results_path + 'signal_solution_cnt_' + str(power_dBm) + '.npy')
    signal_solution_co =   np.load(results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
    pump_solution_co =     np.load(results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')

    z_max = np.load(results_path + 'z_max.npy')
    f = h5py.File(results_path + 'mecozzi.h5', 'r')
    z_max = np.linspace(0, fiber_length, np.shape(pump_solution_cnt)[0])


    # XPM COEFFICIENT EVALUATION, single m =================================
    # compute the collisions between the two furthest WDM channels
    frequency_of_interest = wdm.frequency_grid()[0]
    interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
    single_interference_channel_spacing = interfering_frequency - frequency_of_interest

    # interpolate the amplification function using optimization results
    fB_co = interp1d(z_max, signal_solution_co[:, interfering_grid_index], kind='linear')
    fB_cnt = interp1d(z_max, signal_solution_cnt[:, interfering_grid_index], kind='linear')

    # compute the X0mm coefficients given the precompute time integrals
    m = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/m'])
    z = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/z'])
    I = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/integrals'])

    approx = np.ones_like(m) /(beta2 * 2 * np.pi * single_interference_channel_spacing)
    # X0mm_co = pynlin.nlin.Xhkm_precomputed(
    #     z, I, amplification_function=fB_co(z))
    # X0mm_cnt = pynlin.nlin.Xhkm_precomputed(
    #     z, I, amplification_function=fB_cnt(z))
    X0mm_none = pynlin.nlin.Xhkm_precomputed(
        z, I, amplification_function=None)

    locs = pynlin.nlin.get_collision_location(
        m, fiber, single_interference_channel_spacing, 1 / baud_rate)
    fig1 = plt.figure(figsize=(10, 7))
    plt.plot(show = show_flag)
    for i, m_ in enumerate(m):
        plt.plot(z*1e-3, np.abs(I[i]), color="black")
        plt.axvline(locs[i] * 1e-3, color="blue", linestyle="dashed")
    plt.xlabel("Position [km]")
    fig1.tight_layout()
    fig1.savefig('collision_shape_'+str(power_dBm)+'.pdf')

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(show = show_flag)

    # plt.semilogy(m, np.abs(X0mm_co), marker='x', color='green', label="coprop.")
    # plt.semilogy(m, np.abs(X0mm_cnt), marker='x', color='blue', label="counterprop.")
    plt.semilogy(m, np.abs(X0mm_none), marker="s", color='grey', label="perfect ampl.")
    plt.semilogy(m, np.abs(approx), marker="s", label="approximation")
    plt.minorticks_on()
    plt.grid(which="both")
    plt.xlabel(r"Collision index $m$")
    plt.ylabel(r"$X_{0,m,m}$ [m/s]")
    # plt.title(
    #     rf"$f_B(z)$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
    # )
    plt.legend()
    fig2.tight_layout()
    fig2.savefig('X0mm_'+str(power_dBm)+'.pdf')

    # FULL X0mm EVALUATION FOR EVERY m =======================
    X_co = []
    X_co.append(0.0)
    X_cnt = []
    X_cnt.append(0.0)

    # compute the first num_channels interferents (assume the WDM grid is identical)
    pbar_description = "Computing space integrals"
    collisions_pbar = tqdm.tqdm(range(np.shape(signal_solution_co)[1])[0:num_channels], leave=False)
    collisions_pbar.set_description(pbar_description)

    for interf_index in collisions_pbar:
        fB_co = interp1d(
            z_max, signal_solution_co[:, interf_index], kind='linear')
        X0mm_co = pynlin.nlin.Xhkm_precomputed(
            z, I, amplification_function=None)
        X_co.append(np.sum(np.abs(X0mm_co)**2))

        fB_cnt = interp1d(
            z_max, signal_solution_cnt[:, interf_index], kind='linear')
        X0mm_cnt = pynlin.nlin.Xhkm_precomputed(
            z, I, amplification_function=None)
        X_cnt.append(np.sum(np.abs(X0mm_cnt)**2))

    # PHASE NOISE COMPUTATION =======================
    # copropagating

    M = 16
    qam = pynlin.constellations.QAM(M)

    qam_symbols = qam.symbols()
    cardinality = len(qam_symbols)

    # assign specific average optical energy
    qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power/baud_rate)

    fig4 = plt.figure(figsize=(10, 10))
    plt.plot(show = show_flag)
    plt.scatter(np.real(qam_symbols), np.imag(qam_symbols))

    print("\nConstellation average: ")
    print("\toptical power  =  ", (np.abs(qam_symbols)**2 * baud_rate).mean(), "W")
    print("\toptical energy = ", (np.abs(qam_symbols)**2).mean(), "J")
    print("\tmagnitude      = ", (np.abs(qam_symbols)).mean(), "sqrt(W*s)")

    constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) **2)

    for i in range(1, num_channels):
        Delta_theta_ch_2_co = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[i])
        Delta_theta_2_co[idx] += Delta_theta_ch_2_co
        Delta_theta_ch_2_cnt = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_cnt[i])
        Delta_theta_2_cnt[idx] += Delta_theta_ch_2_cnt

    print("Total phase variance (CO): ", Delta_theta_2_co[idx])
    print("Total phase variance (CNT): ", Delta_theta_2_cnt[idx])


fig_final = plt.figure(figsize=(10, 5))
plt.plot(show = True)
plt.semilogy(power_dBm_list, Delta_theta_2_co, marker='x', color='green', label="coprop.")
plt.semilogy(power_dBm_list, Delta_theta_2_cnt, marker='x', color='blue',label="counterprop.")
plt.minorticks_on()
plt.grid(which="both")
plt.xlabel(r"Power [dBm]")
plt.ylabel(r"$\Delta \theta^2$")
plt.legend()
fig_final.tight_layout()
fig_final.savefig("power_noise.pdf")