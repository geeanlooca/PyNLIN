import argparse
import math
import os
import wave
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from multiprocessing import Pool

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
dispersion = data["dispersion"]
effective_area = data["effective_area"]
baud_rate = data["baud_rate"]
fiber_lengths = data["fiber_length"]
num_channels = data["num_channels"]
interfering_grid_index = data["interfering_grid_index"]
channel_spacing = data["channel_spacing"]
center_frequency = data["center_frequency"]
store_true = data["store_true"]
pulse_shape = data["pulse_shape"]
partial_collision_margin = data["partial_collision_margin"]
num_co = data["num_co"]
num_ct = data["num_ct"]
wavelength = data["wavelength"]
special = data["special"]
pump_direction = data["pump_direction"]
num_only_co_pumps=data['num_only_co_pumps']
num_only_ct_pumps=data['num_only_ct_pumps']

# Manual configuration
power_per_channel_dBm_list = [0.0, -2.0, -4.0, -6.0]
#power_per_channel_dBm_list = np.linspace(-20, 0, 11)
# Pumping scheme choice
pumping_schemes = ['co']
num_only_co_pumps = 4
num_only_ct_pumps = 4
optimize = True
profiles = True

###########################################
#  COMPUTATION OF AMPLITUDE FUNCTIONS
###########################################
beta2 = pynlin.utils.dispersion_to_beta2(
	dispersion * 1e-12 / (1e-9 * 1e3), wavelength
)
ref_bandwidth = baud_rate
fiber = pynlin.fiber.Fiber(
	effective_area=80e-12,
	beta2=beta2
)
wdm = pynlin.wdm.WDM(
	spacing=channel_spacing * 1e-9,
	num_channels=num_channels,
	center_frequency=190
)

# comute the collisions between the two furthest WDM channels
frequency_of_interest = wdm.frequency_grid()[0]
interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
channel_spacing = interfering_frequency - frequency_of_interest
partial_collision_margin = 5
points_per_collision = 10

for fiber_length in fiber_lengths:
	length_setup = int(fiber_length * 1e-3)
	optimization_result_path_co = '../results_' + str(length_setup) + '/optimization/' + str(num_only_co_pumps) + '_co/'
	optimization_result_path_ct = '../results_' + str(length_setup) + '/optimization/' + str(num_only_ct_pumps) + '_ct/'
	optimization_result_path_bi = '../results_' + str(length_setup) + '/optimization/' + str(num_co) + '_co_' + str(num_ct) + '_ct_' + special + '/'
	
	results_path_co = '../results_' + str(length_setup) + '/' + str(num_only_co_pumps) + '_co/'
	results_path_ct = '../results_' + str(length_setup) + '/' + str(num_only_ct_pumps) + '_ct/'
	results_path_bi = '../results_' + str(length_setup) + '/' + str(num_co) + '_co_' + str(num_ct) + '_ct_' + special + '/'
	#
	if not os.path.exists(results_path_co):
		os.makedirs(results_path_co)
	if not os.path.exists(results_path_ct):
		os.makedirs(results_path_ct)
	if not os.path.exists(results_path_bi):
		os.makedirs(results_path_bi)				
	#
	if not os.path.exists(optimization_result_path_co):
		os.makedirs(optimization_result_path_co)
	if not os.path.exists(optimization_result_path_ct):
		os.makedirs(optimization_result_path_ct)
	if not os.path.exists(optimization_result_path_bi):
		os.makedirs(optimization_result_path_bi)

	time_integrals_results_path = '../results/'

	# PRECISION REQUIREMENTS ESTIMATION =================================
	max_channel_spacing = wdm.frequency_grid(
	)[num_channels - 1] - wdm.frequency_grid()[0]
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
	integration_steps = int(np.ceil(fiber_length / dz))
	z_max = np.linspace(0, fiber_length, integration_steps)

	np.save("z_max.npy", z_max)
	pbar_description = "Optimizing vs signal power"
	pbar = tqdm.tqdm(power_per_channel_dBm_list, leave=False)
	pbar.set_description(pbar_description)
		
	def bi_solver(power_per_channel_dBm):
		#print("Power per channel: ", power_per_channel_dBm, "dBm")
		num_pumps = num_co + num_ct
		pump_band_b = lambda2nu(1510e-9)
		pump_band_a = lambda2nu(1410e-9)
		#initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)
		# BROMAGE
		initial_pump_frequencies = np.array(lambda2nu([1485e-9, 1515e-9, 1447e-9, 1467e-9, 1485e-9, 1515e-9]))

		print("INITIAL PUMP FREQUENCIES:\n\t")
		print(initial_pump_frequencies)

		power_per_channel = dBm2watt(power_per_channel_dBm)
		power_per_pump = dBm2watt(-10)
		# signal carriers: 1.558 -> 1.599
		signal_wavelengths = wdm.wavelength_grid()
		pump_wavelengths = nu2lambda(initial_pump_frequencies) * 1e9
		num_pumps = len(pump_wavelengths)

		signal_powers = np.ones_like(signal_wavelengths) * power_per_channel

		initial_power_co = dBm2watt(10)
		initial_power_ct = dBm2watt(-10)
		if pump_direction == "direct":
			pump_directions = np.hstack((np.ones(num_co), -np.ones(num_ct)))
		elif pump_direction == "reversed":
			pump_directions = np.hstack((-np.ones(num_ct), np.ones(num_co)))
		elif pump_direction == "interleaved":
			pump_directions = [1, 1, 1, 1, -1, -1, -1, -1]
		else:
			print("\n Invalid pump direction!\nExiting...")
			exit()

		pump_powers = []
		for direc in pump_directions:
			if direc == 1:
				pump_powers.append(initial_power_co)
			else:
				pump_powers.append(initial_power_ct)

		#pump_powers = dBm2watt(np.array([10, 10, 10, 10, -60, -60, -50, -40]))
		pump_powers = np.array(pump_powers)
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

		target_spectrum = watt2dBm(0.5*signal_powers)
		if power_per_channel > -6.0:
			learning_rate = 1e-4
		else:
			learning_rate = 5e-3

		pump_wavelengths_bi, pump_powers_bi = optimizer.optimize(
			target_spectrum=target_spectrum,
			epochs=600,
			learning_rate=learning_rate
		)
		print("\n\nOPTIMIZED POWERS= ", pump_powers_bi, "\n\n")
		np.save(optimization_result_path_bi + "opt_wavelengths_bi" + str(power_per_channel_dBm) + ".npy", pump_wavelengths_bi)
		np.save(optimization_result_path_bi + "opt_powers_bi" + str(power_per_channel_dBm) + ".npy", pump_powers_bi)
		
		amplifier = NumpyRamanAmplifier(fiber)

		pump_solution_bi, signal_solution_bi, ase_solution_bi = amplifier.solve(
			signal_powers,
			signal_wavelengths,
			pump_powers_bi,
			pump_wavelengths_bi,
			z_max,
			reference_bandwidth=ref_bandwidth
		)

		np.save(results_path_bi+"pump_solution_co_"+str(power_per_channel_dBm)+".npy", pump_solution_bi)
		np.save(results_path_bi+"signal_solution_co_"+str(power_per_channel_dBm)+".npy", signal_solution_bi)
		np.save(results_path_bi+"ase_solution_co_"+str(power_per_channel_dBm)+".npy", ase_solution_bi)
		return 

	def co_solver(power_per_channel_dBm):
		#print("Power per channel: ", power_per_channel_dBm, "dBm")
		num_pumps = num_only_co_pumps
		pump_band_b = lambda2nu(1510e-9)
		pump_band_a = lambda2nu(1410e-9)
		#initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)
		initial_pump_frequencies = np.array(lambda2nu([1447e-9, 1467e-9, 1485e-9, 1515e-9]))

		power_per_channel = dBm2watt(power_per_channel_dBm)
		power_per_pump = dBm2watt(-5)
		signal_wavelengths = wdm.wavelength_grid()
		pump_wavelengths = nu2lambda(initial_pump_frequencies)*1e9
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
		optimizer = CopropagatingOptimizer(
			torch_amplifier,
			torch.from_numpy(pump_wavelengths),
			torch.from_numpy(pump_powers),
		)

		target_spectrum = watt2dBm(0.5*signal_powers)

		target_spectrum = watt2dBm(0.5 * signal_powers)
		if power_per_channel >= -6.0:
			learning_rate = 1e-4
		else:
			learning_rate = 1e-3

		pump_wavelengths_co, pump_powers_co = optimizer.optimize(
			target_spectrum=target_spectrum,
			epochs=500,
			learning_rate=learning_rate,
		)

		np.save(optimization_result_path_co+"opt_wavelengths_co"+str(power_per_channel_dBm)+".npy", pump_wavelengths_co)
		np.save(optimization_result_path_co+"opt_powers_co"+str(power_per_channel_dBm)+".npy", pump_powers_co)

		amplifier = NumpyRamanAmplifier(fiber)

		pump_solution_co, signal_solution_co, ase_solution_co = amplifier.solve(
			signal_powers,
			signal_wavelengths,
			pump_powers_co,
			pump_wavelengths_co,
			z_max,
			reference_bandwidth=ref_bandwidth
		)

		np.save(results_path_co+"pump_solution_co_"+str(power_per_channel_dBm)+".npy", pump_solution_co)
		np.save(results_path_co+"signal_solution_co_"+str(power_per_channel_dBm)+".npy", signal_solution_co)
		np.save(results_path_co+"ase_solution_co_"+str(power_per_channel_dBm)+".npy", ase_solution_co)
		return 

	def ct_solver(power_per_channel_dBm):
		#print("Power per channel: ", power_per_channel_dBm, "dBm")
		num_pumps = num_only_ct_pumps
		pump_band_b = lambda2nu(1480e-9)
		pump_band_a = lambda2nu(1400e-9)
		#initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)
		# BROMAGE
		initial_pump_frequencies = np.array(
			lambda2nu([1447e-9, 1467e-9, 1485e-9, 1515e-9]))
		power_per_channel = dBm2watt(power_per_channel_dBm)
		power_per_pump = dBm2watt(-10)
		signal_wavelengths = wdm.wavelength_grid()
		pump_wavelengths = nu2lambda(initial_pump_frequencies) * 1e9
		num_pumps = len(pump_wavelengths)
		signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
		pump_powers = np.ones_like(pump_wavelengths) * power_per_pump
		torch_amplifier_ct = RamanAmplifier(
			fiber_length,
			integration_steps,
			num_pumps,
			signal_wavelengths,
			power_per_channel,
			fiber,
			pump_direction=-1,
		)

		optimizer = CopropagatingOptimizer(
			torch_amplifier_ct,
			torch.from_numpy(pump_wavelengths),
			torch.from_numpy(pump_powers),
		)

		target_spectrum = watt2dBm(0.5 * signal_powers)
		if power_per_channel > -6.0:
			learning_rate = 1e-4
		else:
			learning_rate = 1e-3

		pump_wavelengths_ct, pump_powers_ct = optimizer.optimize(
			target_spectrum=target_spectrum,
			epochs=500,
			learning_rate=learning_rate,
			lock_wavelengths=200,
		)
		np.save(optimization_result_path_ct + "opt_wavelengths_ct" + str(power_per_channel_dBm) + ".npy", pump_wavelengths_ct)
		np.save(optimization_result_path_ct + "opt_powers_ct" + str(power_per_channel_dBm) + ".npy", pump_powers_ct)

		amplifier = NumpyRamanAmplifier(fiber)

		pump_solution_ct, signal_solution_ct, ase_solution_ct = amplifier.solve(
			signal_powers,
			signal_wavelengths,
			pump_powers_ct,
			pump_wavelengths_ct,
			z_max,
			pump_direction=-1,
			use_power_at_fiber_start=True,
			reference_bandwidth=ref_bandwidth
		)

		np.save(results_path_ct + "pump_solution_ct_" + str(power_per_channel_dBm) + ".npy", pump_solution_ct)
		np.save(results_path_ct + "signal_solution_ct_" + str(power_per_channel_dBm) + ".npy", signal_solution_ct)
		np.save(results_path_ct + "ase_solution_ct_" + str(power_per_channel_dBm) + ".npy", ase_solution_ct)
		return

# OPTIMIZER BIDIRECTIONAL =================================
	if 'bi' in pumping_schemes:
		with Pool(os.cpu_count()) as p:
			p.map(bi_solver, power_per_channel_dBm_list)	
			
# OPTIMIZER CO =================================
	if 'co' in pumping_schemes:
		with Pool(os.cpu_count()) as p:
			p.map(co_solver, power_per_channel_dBm_list)	

# OPTIMIZER COUNTER =================================
	if 'ct' in pumping_schemes:
		with Pool(os.cpu_count()) as p:
			p.map(ct_solver, power_per_channel_dBm_list)	