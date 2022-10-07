
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
from random import shuffle
import json

f = open("/home/lorenzi/Scrivania/progetti/NLIN/PyNLIN/scripts/sim_config.json")
data = json.load(f)
print(data)
dispersion=data["dispersion"] 
effective_area=data["effective_area"] 
baud_rate=data["baud_rate"] 
fiber_lengths=data["fiber_length"] 
num_channels=data["num_channels"]
interfering_grid_index=data["interfering_grid_index"]
channel_spacing=data["channel_spacing"] 
center_frequency=data["center_frequency"] 
store_true=data["store_true"] 
pulse_shape=data["pulse_shape"] 
partial_collision_margin=data["partial_collision_margin"] 
num_co= data["num_co"] 
num_cnt=data["num_cnt"] 
wavelength=data["wavelength"]
for fiber_length in fiber_lengths:
    length_setup = int(fiber_length*1e-3) 
    optimization_result_path = '../results_'+str(length_setup)+'/optimization/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'
    optimization_result_path_cocnt = '../results_'+str(length_setup)+'/optimization/'

    results_path = '../results_'+str(length_setup)+'/'
    results_path_bi = '../results_'+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'
    #
    plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/results_"+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    #
    if not os.path.exists(results_path_bi):
        os.makedirs(results_path_bi)
    #
    if not os.path.exists(optimization_result_path):
        os.makedirs(optimization_result_path)
    #
    if not os.path.exists(optimization_result_path_cocnt):
        os.makedirs(optimization_result_path_cocnt)
    time_integrals_results_path = '../results/'

    # Warning: manual selection of loading of previous data: be sure about previously used params
    # No sanity check is done
    optimize = True
    profiles = True

    # if input("\nAmplification function profiles: \n\t>"+str(length_setup)+"km \n\t>optimize="+str(optimize)+" \n\t>profiles="+str(profiles)+" \n\t>bi setup= ("+str(num_co)+"_co, "+str(num_cnt)+"_cnt) \nAre you sure? (y/[n])") != "y":
    #     exit()

    beta2 = pynlin.utils.dispersion_to_beta2(
        dispersion * 1e-12 / (1e-9 * 1e3), wavelength
    )

    ref_bandwidth = baud_rate
    fiber = pynlin.fiber.Fiber(
        effective_area=80e-12,
        beta2=beta2
    )
    wdm = pynlin.wdm.WDM(
        spacing=channel_spacing *1e-9,
        num_channels=num_channels,
        center_frequency=190
    )


    # comute the collisions between the two furthest WDM channels
    frequency_of_interest = wdm.frequency_grid()[0]
    interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
    channel_spacing = interfering_frequency - frequency_of_interest
    partial_collision_margin = 5
    points_per_collision = 10


    power_per_channel_dBm_list = [-8.0, -6.0, -4.0, -2.0, 0.0]
    power_per_channel_dBm_list = [-20.0, -18.0, -16.0, -14.0, -12.0, -10.0]
    power_per_channel_dBm_list = [0.0, -2.0, -4.0]
    power_per_channel_dBm_list = np.linspace(-20, 0, 11)
    print(power_per_channel_dBm_list)
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

    # Suggestion: 100m step is sufficient
    dz = 100
    integration_steps = int(np.ceil(fiber_length/dz))
    z_max = np.linspace(0, fiber_length, integration_steps)

    np.save("z_max.npy", z_max)
    pbar_description = "Optimizing vs signal power"
    pbar = tqdm.tqdm(power_per_channel_dBm_list, leave=False)
    pbar.set_description(pbar_description)
    '''
    for power_per_channel_dBm in pbar:
        #print("Power per channel: ", power_per_channel_dBm, "dBm")
    # OPTIMIZER BIDIRECTIONAL =================================
        num_pumps = num_co+num_cnt
        pump_band_b = lambda2nu(1510e-9)
        pump_band_a = lambda2nu(1410e-9)
        initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)
        #shuffle(initial_pump_frequencies)
        
        print("INITIAL PUMP FREQUENCIES:\n\t")
        print(initial_pump_frequencies)

        power_per_channel = dBm2watt(power_per_channel_dBm)
        power_per_pump = dBm2watt(-10)
        signal_wavelengths = wdm.wavelength_grid()
        pump_wavelengths = nu2lambda(initial_pump_frequencies)*1e9 
        num_pumps = len(pump_wavelengths)

        signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
        
        initial_power_co = dBm2watt(-10)
        initial_power_cnt = dBm2watt(-30)
        pump_directions = np.hstack((np.ones(num_co), -np.ones(num_cnt)))
        #pump_directions = [-1, 1, -1, 1, -1, 1, -1, 1]

        print(pump_directions)

        pump_powers = []
        for direc in pump_directions:
            if direc == 1:
                pump_powers.append(initial_power_co)
            else:
                pump_powers.append(initial_power_cnt)

        pump_powers = np.array(pump_powers)

        if optimize:
            torch_amplifier = RamanAmplifier(
                fiber_length,
                integration_steps,
                num_pumps,
                signal_wavelengths,
                power_per_channel,
                fiber,
                pump_direction=pump_directions
            )

            optimizer = CopropagatingOptimizer(
                torch_amplifier,
                torch.from_numpy(pump_wavelengths),
                torch.from_numpy(pump_powers),
            )

            target_spectrum = watt2dBm(0.5 * signal_powers)
            if power_per_channel>-6.0:
                learning_rate=3e-4
            else:
                learning_rate=5e-3

            pump_wavelengths_bi, pump_powers_bi = optimizer.optimize(
                target_spectrum=target_spectrum,
                epochs=600,
                learning_rate=learning_rate
            )
            print("\n\nOPTIMIZED POWERS= ", pump_powers_bi, "\n\n")
            np.save(optimization_result_path+"opt_wavelengths_bi"+str(power_per_channel_dBm)+".npy", pump_wavelengths_bi)
            np.save(optimization_result_path+"opt_powers_bi"+str(power_per_channel_dBm)+".npy", pump_powers_bi)
        else:
            pump_wavelengths_bi = np.load(optimization_result_path+"opt_wavelengths_bi"+str(power_per_channel_dBm)+".npy")
            pump_powers_bi = np.load(optimization_result_path+"opt_powers_bi"+str(power_per_channel_dBm)+".npy")
        if profiles: 
            amplifier = NumpyRamanAmplifier(fiber)

            pump_solution_bi, signal_solution_bi, ase_solution_bi = amplifier.solve(
                signal_powers,
                signal_wavelengths,
                pump_powers_bi,
                pump_wavelengths_bi,
                z_max,
                pump_direction=pump_directions,
                use_power_at_fiber_start=True,
                reference_bandwidth=ref_bandwidth
            )

            np.save(results_path_bi+"pump_solution_bi_"+str(power_per_channel_dBm)+".npy", pump_solution_bi)
            np.save(results_path_bi+"signal_solution_bi_"+str(power_per_channel_dBm)+".npy", signal_solution_bi)
            np.save(results_path_bi+"ase_solution_bi_"+str(power_per_channel_dBm)+".npy", ase_solution_bi)
        plt.figure()
        plt.plot(signal_wavelengths, watt2dBm(signal_solution_bi[-1]), color="k")
        plt.savefig(results_path+"signal_profile_"+str(power_per_channel_dBm)+".pdf")
    '''
    for power_per_channel_dBm in pbar:
        #print("Power per channel: ", power_per_channel_dBm, "dBm")
        
    # OPTIMIZER CO =================================
        num_pumps = 8
        pump_band_b = lambda2nu(1510e-9)
        pump_band_a = lambda2nu(1410e-9)
        initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

        power_per_channel = dBm2watt(power_per_channel_dBm)
        power_per_pump = dBm2watt(-10)
        signal_wavelengths = wdm.wavelength_grid()
        pump_wavelengths = nu2lambda(initial_pump_frequencies)*1e9
        num_pumps = len(pump_wavelengths)

        signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
        pump_powers = np.ones_like(pump_wavelengths) * power_per_pump
        if optimize:
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

            pump_wavelengths_co, pump_powers_co = optimizer.optimize(
                target_spectrum=target_spectrum,
                epochs=500
            )

            np.save(optimization_result_path_cocnt+"opt_wavelengths_co"+str(power_per_channel_dBm)+".npy", pump_wavelengths_co)
            np.save(optimization_result_path_cocnt+"opt_powers_co"+str(power_per_channel_dBm)+".npy", pump_powers_co)
        else: 
            pump_wavelengths_co = np.load(optimization_result_path_cocnt+"opt_wavelengths_co"+str(power_per_channel_dBm)+".npy")
            pump_powers_co = np.load(optimization_result_path_cocnt+"opt_powers_co"+str(power_per_channel_dBm)+".npy")

        if profiles:
            amplifier = NumpyRamanAmplifier(fiber)

            pump_solution_co, signal_solution_co, ase_solution_co = amplifier.solve(
                signal_powers,
                signal_wavelengths,
                pump_powers_co,
                pump_wavelengths_co,
                z_max,
                reference_bandwidth=ref_bandwidth
            )

            np.save(results_path+"pump_solution_co_"+str(power_per_channel_dBm)+".npy", pump_solution_co)
            np.save(results_path+"signal_solution_co_"+str(power_per_channel_dBm)+".npy", signal_solution_co)
            np.save(results_path+"ase_solution_co_"+str(power_per_channel_dBm)+".npy", ase_solution_co)

    for power_per_channel_dBm in pbar:
        #print("Power per channel: ", power_per_channel_dBm, "dBm")
    # OPTIMIZER COUNTER =================================

        num_pumps = 10
        pump_band_b = lambda2nu(1480e-9)
        pump_band_a = lambda2nu(1400e-9)
        initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

        power_per_channel = dBm2watt(power_per_channel_dBm)
        power_per_pump = dBm2watt(-45)
        signal_wavelengths = wdm.wavelength_grid()
        pump_wavelengths = nu2lambda(initial_pump_frequencies) *1e9
        num_pumps = len(pump_wavelengths)


        signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
        pump_powers = np.ones_like(pump_wavelengths) * power_per_pump

        if optimize:    
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
            pump_wavelengths_cnt, pump_powers_cnt = optimizer.optimize(
                target_spectrum=target_spectrum,
                epochs=500,
                learning_rate=1e-3,
                lock_wavelengths=150,
            )
            np.save(optimization_result_path_cocnt+"opt_wavelengths_cnt"+str(power_per_channel_dBm)+".npy", pump_wavelengths_cnt)
            np.save(optimization_result_path_cocnt+"opt_powers_cnt"+str(power_per_channel_dBm)+".npy", pump_powers_cnt)
        else: 
            pump_wavelengths_cnt = np.load(optimization_result_path_cocnt+"opt_wavelengths_cnt"+str(power_per_channel_dBm)+".npy")
            pump_powers_cnt = np.load(optimization_result_path_cocnt+"opt_powers_cnt"+str(power_per_channel_dBm)+".npy")


        if profiles:
            amplifier = NumpyRamanAmplifier(fiber)

            pump_solution_cnt, signal_solution_cnt, ase_solution_cnt = amplifier.solve(
                signal_powers,
                signal_wavelengths,
                pump_powers_cnt,
                pump_wavelengths_cnt,
                z_max,
                pump_direction=-1,
                use_power_at_fiber_start=True,
                reference_bandwidth=ref_bandwidth
            )

            np.save(results_path+"pump_solution_cnt_"+str(power_per_channel_dBm)+".npy", pump_solution_cnt)
            np.save(results_path+"signal_solution_cnt_"+str(power_per_channel_dBm)+".npy", signal_solution_cnt)
            np.save(results_path+"ase_solution_cnt_"+str(power_per_channel_dBm)+".npy", ase_solution_cnt)
