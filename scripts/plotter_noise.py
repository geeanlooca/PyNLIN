# THIS SCRIPT PLOTS
# NLIN vs Power
# OSNR vs Power
# ASE vs Power
# NLIN ASE comparison
# NLIN vs channel
# OSNR vs channel
# ASE vs channel

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
from pynlin.utils import dBm2watt, watt2dBm, nu2lambda
from pynlin.wdm import WDM
import pynlin.constellations
from scipy import optimize
from scipy.special import erfc
import json

f = open("/home/lorenzi/Scrivania/progetti/NLIN/PyNLIN/scripts/sim_config.json")
data = json.load(f)
dispersion = data["dispersion"]
effective_area = data["effective_area"]
baud_rate = data["baud_rate"]
fiber_lengths = data["fiber_length"]
num_channels = data["num_channels"]
channel_spacing = data["channel_spacing"]
center_frequency = data["center_frequency"]
store_true = data["store_true"]
pulse_shape = data["pulse_shape"]
partial_collision_margin = data["partial_collision_margin"]
num_co = data["num_co"]
num_ct = data["num_ct"]
wavelength = data["wavelength"]
time_integral_length = data['time_integral_length']
special = data['special']
num_only_co_pumps=data['num_only_co_pumps']
num_only_ct_pumps=data['num_only_ct_pumps']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'

def H(n):
		s = 0
		n = int(n)
		for i in range(n):
				s += 1 / (i + 1)
		return s


def NLIN(n, a, b):
		return [a * (2 * H(np.min([xxx, 50 - xxx + 1]) - 1) + H(50) - H(2 * np.min([xxx, 50 - xxx + 1]))) + b for xxx in n]


def OSNR_to_EVM(osnr):
		osnr = 10**(osnr / 10)
		M = 16
		i_range = [1 + item for item in range(int(np.floor(np.sqrt(M))))]
		beta = [2 * ii - 1 for ii in i_range]
		alpha = [3 * ii * osnr / (2 * (M - 1)) for ii in beta]
		gamma = [1 - ii / np.sqrt(M) for ii in i_range]
		sum1 = np.sum(np.multiply(gamma, list(map(np.exp, [-1 * aa for aa in alpha]))))
		sum2 = np.sum(np.multiply(np.multiply(gamma, beta),
									[erfc(np.sqrt(aa)) for aa in alpha]))

		return np.sqrt(np.divide(1, osnr) - np.sqrt(np.divide(96 / np.pi / (M - 1), osnr)) * sum1 + sum2)


def EVM_to_BER(evm):
		M = 16
		L = 4
		return (1 - 1 / L) / np.log2(L) * erfc(np.sqrt((3 * np.log2(L) * np.sqrt(2)) / ((L**2 - 1) * np.power(evm, 2) * np.log2(M))))


# PLOTTING PARAMETERS
interfering_grid_index = 1
power_dBm_list = np.linspace(-20, 0, 11)
power_list = dBm2watt(power_dBm_list)
coi_list = [0, 9, 19, 29, 39, 49]

wavelength = 1550

beta2 = -pynlin.utils.dispersion_to_beta2(
		dispersion * 1e-12 / (1e-9 * 1e3), wavelength * 1e-9
)
fiber = pynlin.fiber.Fiber(
		effective_area=80e-12,
		beta2=beta2
)
wdm = pynlin.wdm.WDM(
		spacing=channel_spacing * 1e-9,
		num_channels=num_channels,
		center_frequency=190
)
points_per_collision = 10

print("beta2: ", fiber.beta2)
print("gamma: ", fiber.gamma)
Delta_theta_2_co = np.zeros_like(
		np.ndarray(shape=(len(coi_list), len(power_dBm_list)))
)
Delta_theta_2_ct = np.zeros_like(Delta_theta_2_co)
Delta_theta_2_bi = np.zeros_like(Delta_theta_2_co)
Delta_theta_2_none = np.zeros_like(Delta_theta_2_co)
R_co = np.zeros_like(Delta_theta_2_co)
R_ct = np.zeros_like(Delta_theta_2_co)
R_bi = np.zeros_like(Delta_theta_2_co)
R_none = np.zeros_like(Delta_theta_2_co)

show_flag = False
compute_X0mm_space_integrals = True

# if input("\nX0mm and noise variance plotter: \n\t>Length= "+str(fiber_lengths)+"km \n\t>power list= "+str(power_dBm_list)+" \n\t>coi_list= "+str(coi_list)+"\n\t>compute_X0mm_space_integrals= "+str(compute_X0mm_space_integrals)+"\nAre you sure? (y/[n])") != "y":
#   exit()
time_integrals_results_path = '../results/'

f_0_9 = h5py.File(time_integrals_results_path + '0_9_results.h5', 'r')
f_19_29 = h5py.File(time_integrals_results_path + '19_29_results.h5', 'r')
f_39_49 = h5py.File(time_integrals_results_path + '39_49_results.h5', 'r')
file_length = time_integral_length

for fiber_length in fiber_lengths:
		length_setup = int(fiber_length * 1e-3)
		plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_" + \
				str(length_setup) + '/' + str(num_co) + '_co_' + \
				str(num_ct) + '_ct_' + special + '/'
		#
		if not os.path.exists(plot_save_path):
				os.makedirs(plot_save_path)
		#
		results_path_co = '../results_' + str(length_setup) + '/' + str(num_only_co_pumps) + '_co/'
		results_path_ct = '../results_' + str(length_setup) + '/' + str(num_only_ct_pumps) + '_ct/'
		results_path_bi = '../results_' + str(length_setup) + '/' + str(num_co) + '_co_' + str(num_ct) + '_ct_' + special + '/'
		noise_path = '../noises/'

		# retrieve ASE and SIGNAL power at receiver
		X_co = np.zeros_like(
				np.ndarray(shape=(len(coi_list), len(power_dBm_list)))
		)
		X_ct = np.zeros_like(X_co)
		X_bi = np.zeros_like(X_co)
		X_none = np.zeros_like(X_co)
		
		T_co = np.zeros_like(X_co)
		T_ct = np.zeros_like(X_co)
		T_bi = np.zeros_like(X_co)
		T_none = np.zeros_like(X_co)

		ase_co = np.zeros_like(X_co)
		ase_ct = np.zeros_like(X_co)
		ase_bi = np.zeros_like(X_co)

		power_at_receiver_co = np.zeros_like(X_co)
		power_at_receiver_ct = np.zeros_like(X_co)
		power_at_receiver_bi = np.zeros_like(X_co)

		for pow_idx, power_dBm in enumerate(power_dBm_list):
				# PUMP power evolution
				pump_solution_co = np.load(
						results_path_co + 'pump_solution_co_' + str(power_dBm) + '.npy')
				pump_solution_ct = np.load(
						results_path_ct + 'pump_solution_ct_' + str(power_dBm) + '.npy')
				pump_solution_bi = np.load(
						results_path_bi + 'pump_solution_bi_' + str(power_dBm) + '.npy')

				# SIGNAL power evolution
				signal_solution_co = np.load(
						results_path_co + 'signal_solution_co_' + str(power_dBm) + '.npy')
				signal_solution_ct = np.load(
						results_path_ct + 'signal_solution_ct_' + str(power_dBm) + '.npy')
				signal_solution_bi = np.load(
						results_path_bi + 'signal_solution_bi_' + str(power_dBm) + '.npy')

				# ASE power evolution
				ase_solution_co = np.load(
						results_path_co + 'ase_solution_co_' + str(power_dBm) + '.npy')
				ase_solution_ct = np.load(
						results_path_ct + 'ase_solution_ct_' + str(power_dBm) + '.npy')
				ase_solution_bi = np.load(
						results_path_bi + 'ase_solution_bi_' + str(power_dBm) + '.npy')

				z_max = np.linspace(0, fiber_length, np.shape(pump_solution_ct)[0])

				# compute the X0mm coefficients given the precompute time integrals
				# FULL X0mm EVALUATION FOR EVERY m =======================
				for coi_idx, coi in enumerate(coi_list):
						power_at_receiver_co[coi_idx,pow_idx] = signal_solution_co[-1, coi_idx]
						power_at_receiver_ct[coi_idx,pow_idx] = signal_solution_ct[-1, coi_idx]
						power_at_receiver_bi[coi_idx,pow_idx] = signal_solution_bi[-1, coi_idx]
						ase_co[coi_idx, pow_idx] = ase_solution_co[-1, coi_idx]
						ase_ct[coi_idx, pow_idx] = ase_solution_ct[-1, coi_idx]
						ase_bi[coi_idx, pow_idx] = ase_solution_bi[-1, coi_idx]

		# Retrieve sum of X0mm^2: noises
		X_co =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_X_co.npy')
		X_ct =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_X_ct.npy')
		X_bi =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_X_bi.npy')
		X_none = np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_X_none.npy')
		
		# Retrieve sum of X0mm^2: noises
		T_co =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_T_co.npy')
		T_ct =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_T_ct.npy')
		T_bi =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_T_bi.npy')
		T_none = np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_T_none.npy')
		# choose modulation format and compute phase noise
		M = 16
		for pow_idx, power_dBm in enumerate(power_dBm_list):
				average_power = dBm2watt(power_dBm)
				qam = pynlin.constellations.QAM(M)
				qam_symbols = qam.symbols()

				# assign specific average optical energy
				qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power / baud_rate)
				constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)

				for coi_idx, coi in enumerate(coi_list):
						Delta_theta_2_co[coi_idx, pow_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[coi_idx, pow_idx])
						Delta_theta_2_ct[coi_idx, pow_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_ct[coi_idx, pow_idx])
						Delta_theta_2_bi[coi_idx, pow_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_bi[coi_idx, pow_idx])
						Delta_theta_2_none[coi_idx, pow_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_none[coi_idx, pow_idx])

						R_co[coi_idx, pow_idx] = constellation_variance * np.abs(T_co[coi_idx, pow_idx])
						R_ct[coi_idx, pow_idx] = constellation_variance * np.abs(T_ct[coi_idx, pow_idx])
						R_bi[coi_idx, pow_idx] = constellation_variance * np.abs(T_bi[coi_idx, pow_idx])
						R_none[coi_idx, pow_idx] = constellation_variance * np.abs(T_none[coi_idx, pow_idx])
## PLOTTING
		plot_height = 7
		plot_width = 10
		aspect_ratio = 10/7
		markers = ["x", "+", "o", "o", "x", "+"]
		wavelength_list = [nu2lambda(wdm.frequency_grid()[_]) * 1e9 for _ in coi_list]
		coi_selection = [0, 19, 49]
		coi_selection_idx = [0, 2, 5]
		coi_selection_average = [0, 9, 19, 29, 39, 49]
		coi_selection_idx_average = [0, 1, 2, 3, 4, 5]
		selected_power = -10.0
		pow_idx = np.where(power_dBm_list == selected_power)[0]
		P_B = 10**((selected_power-30) / 10)  # average power of the constellation in mW
		T = (1 / baud_rate)

		full_coi = [i + 1 for i in range(50)]
		# selection between 'NLIN_vs_power', 'ASE_vs_power', 'NLIN_and_ASE_vs_power', 'OSNR_vs_power', 'OSNR_ASE_vs_wavelength', 'EVM_BER_vs_power'
		plot_selection = ['NLIN_vs_power', 
											'ASE_vs_power', 
											'NLIN_and_ASE_vs_power',
											'OSNR_vs_power', 
											'OSNR_ASE_vs_wavelength', 
											'NLIN_vs_wavelength',
											'NLIN_and_ASE_and_SRSN_vs_power', 
											'(NLIN_plus_SRSN)_vs_wavelength']
		# evaluation of metrics
		# Average OSNR vs power
		osnr_co = np.ndarray(shape=(len(power_dBm_list)))
		osnr_ct = np.zeros_like(osnr_co)
		osnr_bi = np.zeros_like(osnr_co)
		osnr_none = np.zeros_like(osnr_co)
		for scan in range(len(coi_selection_average)):
				osnr_co += 10 * np.log10(power_at_receiver_co[coi_selection_idx_average[scan], :]/   (power_at_receiver_co[coi_selection_idx_average[scan], :] * Delta_theta_2_co[coi_selection_idx_average[scan], :] + ase_co[coi_selection_idx_average[scan], :]))
				osnr_ct += 10 * np.log10(power_at_receiver_ct[coi_selection_idx_average[scan], :]/ (power_at_receiver_ct[coi_selection_idx_average[scan], :] * Delta_theta_2_ct[coi_selection_idx_average[scan], :] + ase_ct[coi_selection_idx_average[scan], :]))
				osnr_bi += 10 * np.log10(power_at_receiver_bi[coi_selection_idx_average[scan], :]/   (power_at_receiver_bi[coi_selection_idx_average[scan], :] * Delta_theta_2_bi[coi_selection_idx_average[scan], :] + ase_bi[coi_selection_idx_average[scan], :]))
				osnr_none += power_dBm_list - 3 - 10 * np.log10(power_list * 0.5 * Delta_theta_2_none[coi_selection_idx_average[scan], :]) - 30
		osnr_co /= len(coi_selection_idx_average)
		osnr_ct /= len(coi_selection_idx_average)
		osnr_bi /= len(coi_selection_idx_average)
		osnr_none /= len(coi_selection_idx_average)

		##############################
		# power plots
		##############################
		if 'NLIN_vs_power' in plot_selection:
				fig_power, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
						nrows=2, ncols=2, sharex=True, figsize=(plot_width, plot_height))
				plt.plot(show=True)
				for scan in range(len(coi_selection)):
						ax1.plot(power_dBm_list, 10 * np.log10(Delta_theta_2_co[coi_selection_idx[scan], :] * power_at_receiver_co[coi_idx, :]) , marker=markers[scan],
										markersize=10, color='green', label="ch." + str(coi_selection[scan]) + " co.")
						ax2.plot(power_dBm_list, 10 * np.log10(Delta_theta_2_ct[coi_selection_idx[scan], :] * power_at_receiver_ct[coi_idx, :]) , marker=markers[scan],
										markersize=10, color='blue', label="ch." + str(coi_selection[scan]) + " count.")
						ax3.plot(power_dBm_list, 10 * np.log10(Delta_theta_2_bi[coi_selection_idx[scan], :] * power_at_receiver_bi[coi_idx, :]), marker=markers[scan],
										markersize=10, color='orange', label="ch." + str(coi_selection[scan] + 1))
						ax4.plot(power_dBm_list, 10 * np.log10(Delta_theta_2_none[coi_selection_idx[scan], :]) + power_dBm_list-3, marker=markers[scan],
										markersize=10, color='grey', label="ch." + str(coi_selection[scan] + 1))
				ax1.grid(which="both")
				#plt.annotate("ciao", (0, 0))
				ax2.grid(which="both")
				ax3.grid(which="both")
				ax4.grid(which="both")

				ax1.set_ylabel(r"NLIN [dBm]")
				ax2.set_ylabel(r"NLIN [dBm]")
				ax3.set_ylabel(r"NLIN [dBm]")
				ax4.set_ylabel(r"NLIN [dBm]")
				ax3.set_xlabel(r"Power [dBm]")
				ax4.set_xlabel(r"Power [dBm]")

				ax1.text(-15, -20, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax2.text(-15, -40, 'ct', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax3.text(-15, -40, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax4.text(-5, -80, 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax4.legend()
				ax2.yaxis.set_label_position("right")
				ax2.yaxis.tick_right()
				ax4.yaxis.set_label_position("right")
				ax4.yaxis.tick_right()
				plt.subplots_adjust(wspace=0.0, hspace=0, right=8.5 / 10, top=9.9 / 10)
				fig_power.savefig(plot_save_path + "NLIN_vs_power.pdf")


		if 'ASE_vs_power' in plot_selection:
				# separate plotting
				fig_ase, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(plot_width, plot_height))
				plt.plot(show=True)
				for scan in range(len(coi_selection)):
						ax1.plot(power_dBm_list, 10 * np.log10(ase_co[coi_selection_idx[scan], :]) + 30, marker=markers[scan],
										markersize=10, color='green', label="ch." + str(coi_selection[scan]) + " co.")
						ax2.plot(power_dBm_list, 10 * np.log10(ase_ct[coi_selection_idx[scan], :]) + 30, marker=markers[scan],
										markersize=10, color='blue', label="ch." + str(coi_selection[scan]) + " count.")
						ax3.plot(power_dBm_list, 10 * np.log10(ase_bi[coi_selection_idx[scan], :]) + 30, marker=markers[scan],
										markersize=10, color='orange', label="ch." + str(coi_selection[scan] + 1))
				ax1.grid(which="both")
				ax3.grid(which="both")
				ax3.grid(which="both")

				ax3.set_xlabel(r"Power [dBm]")

				ax1.set_ylabel(r"ASE noise [dBm]")
				ax2.set_ylabel(r"ASE noise [dBm]")
				ax3.set_ylabel(r"ASE noise [dBm]")
				ax3.legend()
				plt.minorticks_on()
				plt.subplots_adjust(wspace=0.0, hspace=0, right=9.8 / 10, top=9.9 / 10)
				fig_ase.savefig(plot_save_path + "ASE_vs_power.pdf")


		if 'NLIN_and_ASE_vs_power' in plot_selection:
				# averaged plotting
				fig_comparison, ((ax1)) = plt.subplots(nrows=1, sharex=True, figsize=(10, 7))
				plt.plot(show=True)
				axis_num = 0
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_co[coi_idx, :] * Delta_theta_2_co[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[0],
										markersize=10, color='green', label="NLIN")
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_ct[coi_idx, :] * Delta_theta_2_ct[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[0],
										markersize=10, color='blue')
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_bi[coi_idx, :] * Delta_theta_2_bi[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[0],
										markersize=10, color='orange')
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([ase_co[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[2],
										markersize=10, color='green', label="ASE")
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([ase_ct[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[2],
										markersize=10, color='blue')
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([ase_bi[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[2],
										markersize=10, color='orange')
				ax1.grid(which="both")
				#plt.annotate("ciao", (0, 0))
				plt.grid(which="both")

				plt.xlabel(r"Input power [dBm]")
				plt.minorticks_on()
				plt.ylabel(r"Noise power [dBm]")
				plt.grid()
				plt.minorticks_on()
				plt.ylim([-90, 0])

				plt.legend()
				leg = ax1.get_legend()
				leg.legendHandles[0].set_color('grey')
				leg.legendHandles[1].set_color('grey')
				#plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
				#plt.axis([-13, -5, -60, -45])
				plt.tight_layout()
				fig_comparison.savefig(plot_save_path + "NLIN_and_ASE_vs_power.pdf")

		if 'NLIN_and_ASE_and_SRSN_vs_power' in plot_selection:
				# averaged plotting
				fig_comparison, ((ax1)) = plt.subplots(nrows=1, sharex=True, figsize=(10, 7))
				plt.plot(show=True)
				axis_num = 0
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_co[coi_idx, :] * Delta_theta_2_co[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[0], markersize=10, color='green', label="NLIN")
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_ct[coi_idx, :] * Delta_theta_2_ct[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[0], markersize=10, color='blue')
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_bi[coi_idx, :] * Delta_theta_2_bi[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[0], markersize=10, color='orange')

				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_co[coi_idx, :] * (R_co[coi_idx, :] - Delta_theta_2_co[coi_idx, :]) for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[1], markersize=10, color='green', label="SRSN")
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_ct[coi_idx, :] * (R_ct[coi_idx, :] - Delta_theta_2_ct[coi_idx, :]) for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[1], markersize=10, color='blue')
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([power_at_receiver_bi[coi_idx, :] * (R_bi[coi_idx, :] - Delta_theta_2_bi[coi_idx, :]) for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[1], markersize=10, color='orange')

				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([ase_co[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[2], markersize=10, color='green', label="ASE")
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([ase_ct[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[2], markersize=10, color='blue')
				plt.plot(power_dBm_list, 30 + 10 * np.log10(np.average([ase_bi[coi_idx, :] for coi_idx in coi_selection_idx_average], axis=axis_num)), marker=markers[2], markersize=10, color='orange')
				ax1.grid(which="both")
				#plt.annotate("ciao", (0, 0))
				plt.grid(which="both")

				plt.xlabel(r"Input power [dBm]")
				plt.minorticks_on()
				plt.ylabel(r"Noise power [dBm]")
				plt.grid()
				plt.minorticks_on()
				#plt.ylim([-90, 0])

				plt.legend()
				leg = ax1.get_legend()
				leg.legendHandles[0].set_color('grey')
				leg.legendHandles[1].set_color('grey')
				#plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
				#plt.axis([-13, -5, -60, -45])
				plt.tight_layout()
				fig_comparison.savefig(plot_save_path + "NLIN_and_ASE_and_SRSN_vs_power.pdf")

		if 'OSNR_vs_power' in plot_selection:
				fig_powsnr, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(plot_width, plot_height))
				plt.plot(power_dBm_list, osnr_co, marker=markers[scan],
										markersize=10, color='green')
				plt.plot(power_dBm_list, osnr_ct, marker=markers[scan],
										markersize=10, color='blue')
				plt.plot(power_dBm_list, osnr_bi, marker=markers[scan],
										markersize=10, color='orange')
				plt.grid(which="both")

				#plt.ylim([20, 50])
				plt.ylabel(r"$OSNR$ [dB]")
				plt.xlabel(r"Power [dBm]")

				plt.subplots_adjust(wspace=0.0, hspace=0, right=8.5 / 10, top=9.9 / 10)
				fig_powsnr.savefig(plot_save_path + "OSNR_vs_power.pdf")

		#####################################
		# wavelength plots
		#####################################
		if 'OSNR_ASE_vs_wavelength' in plot_selection:
				fig_ASE_channel, ((ax1)) = plt.subplots(
						nrows=1, ncols=1, sharex=True, figsize=(plot_width, plot_height))
				plt.plot(show=True)
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_co[:, pow_idx] / ase_ct[:, pow_idx]), marker='x', markersize=15, color='blue', label="ch." + str(coi) + "CO")
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_bi[:, pow_idx] / ase_bi[:, pow_idx]), marker='x', markersize=15, color='orange', label="ch." + str(coi) + "ct.")
				# ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
				plt.xlabel(r"Channel wavelength [nm]")
				plt.xticks(ticks=[wavelength_list[0], wavelength_list[-1]],
										labels=["%4.0f" % (_) for _ in [wavelength_list[0], wavelength_list[-1]]])
				plt.ylabel(r"$OSNR_{ASE}$ [dB]")
				#plt.ylim([-50, -45])
				plt.grid(which="both")

				plt.tight_layout()
				fig_ASE_channel.savefig(plot_save_path + "OSNR_ASE_vs_wavelength.pdf")
		print(power_at_receiver_co[0, 0])
		if 'NLIN_vs_wavelength' in plot_selection:
				fig_NLIN_channel, ((ax1)) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(plot_width, 10))
				plt.plot(show=True)
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_co[:, pow_idx] * Delta_theta_2_co[:, pow_idx]) + 30, marker='x', markersize=15, color='green', label="ch." + str(coi) + "co")
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_bi[:, pow_idx] * Delta_theta_2_ct[:, pow_idx]) + 30, marker='x', markersize=15, color='blue', label="ch." + str(coi) + "ct")
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_bi[:, pow_idx] * Delta_theta_2_bi[:, pow_idx]) + 30, marker='x', markersize=15, color='orange', label="ch." + str(coi) + "bi")
				plt.plot(wavelength_list, 10 * np.log10(P_B/2 * Delta_theta_2_none[:, pow_idx])+30, marker='x', markersize=15, color='grey', label="ch." + str(coi) + "perfect")
				# ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
				plt.xlabel(r"Channel wavelength [nm]")
				plt.xticks(ticks=[wavelength_list[0], wavelength_list[-1]],
										labels=["%4.0f" % (_) for _ in [wavelength_list[0], wavelength_list[-1]]])
				plt.ylabel(r"$NLIN$ [dBm]")
				#plt.ylim([-50, -45])
				plt.grid(which="both")

				plt.tight_layout()
				fig_NLIN_channel.savefig(plot_save_path + "NLIN_vs_wavelength.pdf")

		if '(NLIN_plus_SRSN)_vs_wavelength' in plot_selection:
				fig_NLIN_channel, ((ax1)) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(plot_width, 10))
				plt.plot(show=True)
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_co[:, pow_idx] * (R_co[:, pow_idx]- Delta_theta_2_co[:, pow_idx]))+30, marker='x', markersize=15, color='green', label="ch." + str(coi) + "co")
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_bi[:, pow_idx] * (R_ct[:, pow_idx]- Delta_theta_2_ct[:, pow_idx]))+30, marker='x', markersize=15, color='blue', label="ch." + str(coi) + "ct")
				plt.plot(wavelength_list, 10 * np.log10(power_at_receiver_bi[:, pow_idx] * (R_bi[:, pow_idx]- Delta_theta_2_bi[:, pow_idx]))+30, marker='x', markersize=15, color='orange', label="ch." + str(coi) + "bi")
				plt.plot(wavelength_list, 10 * np.log10(P_B/2 * Delta_theta_2_none[:, pow_idx])+30, marker='x', markersize=15, color='grey', label="ch." + str(coi) + "perfect")
				# ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
				plt.xlabel(r"Channel wavelength [nm]")
				plt.xticks(ticks=[wavelength_list[0], wavelength_list[-1]],
										labels=["%4.0f" % (_) for _ in [wavelength_list[0], wavelength_list[-1]]])
				plt.ylabel(r"$(NLIN + SRSN)$ [dBm]")
				#plt.ylim([-50, -45])
				plt.grid(which="both")

				plt.tight_layout()
				fig_NLIN_channel.savefig(plot_save_path + "(NLIN_plus_SRSN)_vs_wavelength.pdf")
				
		#####################################
		# error metrics vs power
		#####################################
		if 'EVM_BER_vs_power' in plot_selection:
				fig_ber, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
						nrows=2, ncols=2, sharex=True, figsize=(16, 8))
				power_list = list(map(dBm2watt, power_dBm_list))
				for scan in range(len(coi_selection)):
						osnr_co = power_dBm_list - 10 * \
								np.log10(power_list * Delta_theta_2_co[coi_selection_idx[scan],
												 :] + ase_co[coi_selection_idx[scan], :]) - 30
						osnr_ct = power_dBm_list - 10 * \
								np.log10(power_list * Delta_theta_2_ct[coi_selection_idx[scan],
												 :] + ase_ct[coi_selection_idx[scan], :]) - 30
						osnr_bi = power_dBm_list - 10 * \
								np.log10(power_list * Delta_theta_2_bi[coi_selection_idx[scan],
												 :] + ase_bi[coi_selection_idx[scan], :]) - 30
						osnr_none = power_dBm_list - 10 * \
								np.log10(power_list *
												 Delta_theta_2_none[coi_selection_idx[scan], :]) - 30
						osnr_list = [osnr_co, osnr_ct, osnr_bi, osnr_none]
						evms = []
						bers = []
						for osnr in osnr_list:
								evms.append(list(map(OSNR_to_EVM, osnr)))
								bers.append(list(map(EVM_to_BER, map(OSNR_to_EVM, osnr))))

						ax1.semilogy(power_dBm_list, bers[0], marker=markers[scan],
												 markersize=10, color='green', label="ch." + str(coi_selection[scan]) + " co.")
						ax2.semilogy(power_dBm_list, bers[1], marker=markers[scan],
												 markersize=10, color='blue', label="ch." + str(coi_selection[scan]) + " count.")
						ax3.semilogy(power_dBm_list, bers[2], marker=markers[scan],
												 markersize=10, color='orange', label="ch." + str(coi_selection[scan] + 1))
						ax4.semilogy(power_dBm_list, bers[3], marker=markers[scan],
												 markersize=10, color='grey', label="ch." + str(coi_selection[scan] + 1))
				ax1.grid(which="both")
				#plt.annotate("ciao", (0, 0))
				ax2.grid(which="both")
				ax3.grid(which="both")
				ax4.grid(which="both")

				ax1.set_ylabel(r"BER")
				ax2.set_ylabel(r"BER")
				ax3.set_ylabel(r"BER")
				ax4.set_ylabel(r"BER")
				ax3.set_xlabel(r"Power [dBm]")
				ax4.set_xlabel(r"Power [dBm]")
				ax1.set_ylim([10**-50, 1])
				ax2.set_ylim([10**-50, 1])
				ax3.set_ylim([10**-50, 1])
				ax4.set_ylim([10**-50, 1])

				ax1.text(-15, 10**(-6), 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax2.text(-15, 10**(-6), 'ct', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax3.text(-15, 10**(-6), 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax4.text(-10, 10**(-6), 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
				ax4.legend()
				ax2.yaxis.set_label_position("right")
				ax2.yaxis.tick_right()
				ax4.yaxis.set_label_position("right")
				ax4.yaxis.tick_right()
				plt.subplots_adjust(wspace=0.0, hspace=0, right=8.5 / 10, top=9.9 / 10)
				fig_ber.savefig(plot_save_path + "BER_vs_power.pdf")

				fig_ber, ax1 = plt.subplots(nrows=1, sharex=True, figsize=(8, 6))

				coi_selection = [0, 19, 49]
				coi_selection_idx = [0, 2, 5]
				power_list = list(map(dBm2watt, power_dBm_list))
				for scan in range(len(coi_selection)):
						osnr_co = power_dBm_list - 10 * \
								np.log10(power_list * Delta_theta_2_co[coi_selection_idx[scan],
												 :] + ase_co[coi_selection_idx[scan], :]) - 30
						osnr_list = [osnr_co]
						evms = []
						bers = []
						for osnr in osnr_list:
								evms.append(list(map(OSNR_to_EVM, osnr)))
								bers.append(list(map(EVM_to_BER, map(OSNR_to_EVM, osnr))))

						ax1.semilogy(power_dBm_list[-3:], bers[0][-3:], marker=markers[scan],
												 markersize=10, color='green', label="ch." + str(coi_selection[scan] + 1) + " co.")
				ax1.grid(which="both")
				ax1.set_ylabel(r"BER")
				ax1.set_xlabel(r"Power [dBm]")
				plt.legend()

				ax1.text(-15, 10**(-6), 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
				plt.subplots_adjust(left=0.2, bottom=0.15, wspace=0.0,
														hspace=0, right=9.5 / 10, top=9.8 / 10)
				plt.tight_layout()
				fig_ber.savefig(plot_save_path + "BER_vs_power_zoom.pdf")
