# This script plot
# hybrid OSRN tradeoff vs signal input power, Raman gain

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.utils import dBm2watt, watt2dBm, nu2lambda
import pynlin.constellations
from scipy.special import erfc
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly as ptly
f = open("./scripts/sim_config.json")
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
gain_dB_setup=data['gain_dB_list']
gain_dB_list = np.linspace(gain_dB_setup[0], gain_dB_setup[1], gain_dB_setup[2])
total_gain_dB = 0.0
power_dBm_setup=data['power_dBm_list']
power_dBm_list = np.linspace(power_dBm_setup[0], power_dBm_setup[1], power_dBm_setup[2])

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
Delta_theta_2_ct = np.zeros_like(
		np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(gain_dB_list)))
)
R_ct =   np.zeros_like(Delta_theta_2_ct)
Delta_theta_2_co = np.zeros_like(Delta_theta_2_ct)
R_co = np.zeros_like(Delta_theta_2_ct)

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
		results_path_ct = '../results_' + str(length_setup) + '/' + str(num_only_ct_pumps) + '_ct/'
		results_path_co = '../results_' + str(length_setup) + '/' + str(num_only_co_pumps) + '_co/'

		noise_path = '../noises/'

		# retrieve ASE and SIGNAL power at receiver
		X_ct = np.zeros_like(
				np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(gain_dB_list)))
		)
		X_co = np.zeros_like(
				np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(gain_dB_list)))
		)
		
		T_ct = np.zeros_like(X_ct)
		ase_ct = np.zeros_like(X_ct)
		power_at_receiver_ct = np.zeros_like(X_ct)
		all_ct_pumps = np.zeros((len(power_dBm_list), 4))
		avg_pump_dBm_ct = np.zeros((len(power_dBm_list)))

		T_co = np.zeros_like(X_co)
		ase_co = np.zeros_like(X_co)
		power_at_receiver_co = np.zeros_like(X_co)
		all_co_pumps = np.zeros((len(power_dBm_list), 4))
		avg_pump_dBm_co = np.zeros((len(power_dBm_list)))

		for gain_idx, gain_dB in enumerate(gain_dB_list):
			for pow_idx, power_dBm in enumerate(power_dBm_list):
					# PUMP power evolution
					pump_solution_ct = np.load(results_path_ct + 'pump_solution_ct_power' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')
					# SIGNAL power evolution
					signal_solution_ct = np.load(results_path_ct + 'signal_solution_ct_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')
					# ASE power evolution
					ase_solution_ct = np.load(results_path_ct + 'ase_solution_ct_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')

					# PUMP power evolution
					pump_solution_co = np.load(results_path_co + 'pump_solution_co_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')
					# SIGNAL power evolution
					signal_solution_co = np.load(results_path_co + 'signal_solution_co_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')
					# ASE power evolution
					ase_solution_co = np.load(results_path_co + 'ase_solution_co_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')

					z_max = np.linspace(0, fiber_length, np.shape(pump_solution_ct)[0])

					# compute the X0mm coefficients given the precompute time integrals
					# FULL X0mm EVALUATION FOR EVERY m =======================
					all_ct_pumps[pow_idx, :] = watt2dBm(pump_solution_ct[-1, :])
					avg_pump_dBm_ct[pow_idx] = watt2dBm(np.mean(pump_solution_ct[-1, :]))

					all_co_pumps[pow_idx, :] = watt2dBm(pump_solution_co[-1, :])
					avg_pump_dBm_co[pow_idx] = watt2dBm(np.mean(pump_solution_co[-1, :]))

					for coi_idx, coi in enumerate(coi_list):
							power_at_receiver_ct[coi_idx, pow_idx, gain_idx] = signal_solution_ct[-1, coi_idx]
							ase_ct[coi_idx, pow_idx, gain_idx] = ase_solution_ct[-1, coi_idx]

							power_at_receiver_co[coi_idx, pow_idx, gain_idx] = signal_solution_co[-1, coi_idx]
							ase_co[coi_idx, pow_idx, gain_idx] = ase_solution_co[-1, coi_idx]

		# Retrieve sum of X0mm^2: noises
		M = 16
		X_ct =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_X_ct.npy')
		X_co =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_X_co.npy')

		for gain_idx, gain_dB in enumerate(gain_dB_list):
			for pow_idx, power_dBm in enumerate(power_dBm_list):
					average_power = dBm2watt(power_dBm)
					qam = pynlin.constellations.QAM(M)
					qam_symbols = qam.symbols()

					# assign specific average optical energy
					qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power / baud_rate)
					constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)
					for coi_idx, coi in enumerate(coi_list):
							#print("\n\nreassinging delta theta")
							Delta_theta_2_ct[coi_idx, pow_idx, gain_idx] =   16/9 * fiber.gamma**2 * constellation_variance * np.abs(X_ct[coi_idx, pow_idx, gain_idx])
							print(np.shape(X_ct))
							print(np.shape(X_co))

							Delta_theta_2_co[coi_idx, pow_idx, gain_idx] =   16/9 * fiber.gamma**2 * constellation_variance * np.abs(X_co[coi_idx, pow_idx, gain_idx])

# ==============================
# PLOTTING
# ==============================

plot_height = 7
plot_width = 10
aspect_ratio = 10/7
markers = ["x", "+", "o", "o", "x", "+"]
wavelength_list = [nu2lambda(wdm.frequency_grid()[_]) * 1e6 for _ in coi_list]
freq_list = [wdm.frequency_grid()[_] for _ in coi_list]
coi_selection = [0, 19, 49]
coi_selection_idx = [0, 2, 5]
coi_selection_average = [49]
coi_selection_idx_average = [5]


selected_power = -14.0
pow_idx = np.where(power_dBm_list == selected_power)[0]
P_B = 10**((selected_power-30) / 10)  # average power of the constellation in mW
T = (1 / baud_rate)
P_A = power_list
full_coi = [i + 1 for i in range(50)]

# EDFA noise from model
h_planck = 6.626e-34
edfa_noise = np.ndarray(shape=(len(coi_list), len(gain_dB_list)))
NG = 2
B = baud_rate
for chan_idx, freq in enumerate(freq_list):
	for gain_idx, gain_dB in enumerate(gain_dB_list):
		G = 10**((total_gain_dB-gain_dB)/10) # the remaining part of the gain for full compensation
		edfa_noise[chan_idx, gain_idx] = h_planck * freq * NG * (G - 1) * B

osnr_ct = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
osnr_co = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))

osnr_ct_no_edfa_noise = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
nlin = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
nlin_co = np.zeros_like(nlin)
ase = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
ase_raman = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
nlin_raman = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
edfa_ase = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
rec_pow = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))
out_pow = np.zeros(shape=(len(power_dBm_list), len(gain_dB_list)))

ase_vec_ct = np.zeros(shape=(len(coi_selection_average), len(power_dBm_list), len(gain_dB_list)))
nli_vec_ct = np.zeros(shape=(len(coi_selection_average), len(power_dBm_list), len(gain_dB_list)))
pow_vec_ct = np.zeros(shape=(len(coi_selection_average), len(power_dBm_list), len(gain_dB_list)))

ase_vec_co = np.zeros_like(ase_vec_ct) 
nli_vec_co = np.zeros_like(nli_vec_ct) 
pow_vec_co = np.zeros_like(pow_vec_ct) 

# channel selection 
for scan in range(len(coi_selection_average)):
	ase_vec_ct[scan, :, :] = ase_ct[coi_selection_idx_average[scan], :, :]
	pow_vec_ct[scan, :, :] = power_at_receiver_ct[coi_selection_idx_average[scan], :, :]
	nli_vec_ct[scan, :, :] = np.transpose([P_A * Delta_theta_2_ct[coi_selection_idx_average[scan], :, _] for _ in range(len(gain_dB_list))] )

	ase_vec_co[scan, :, :] = ase_co[coi_selection_idx_average[scan], :, :]
	pow_vec_co[scan, :, :] = power_at_receiver_co[coi_selection_idx_average[scan], :, :]
	nli_vec_co[scan, :, :] = np.transpose([P_A * Delta_theta_2_co[coi_selection_idx_average[scan], :, _] for _ in range(len(gain_dB_list))] )

for pow_idx, power in enumerate(power_dBm_list):
	for gain_idx, gain in enumerate(gain_dB_list):
		edfa_gain_dB = -gain
		edfa_gain_lin = 10**(edfa_gain_dB/10)
		print("gain         dB : ", gain)
		print("egain        dB : ", 10*np.log10(edfa_gain_lin))
		print("overall_gain dB : ", gain+10*np.log10(edfa_gain_lin))
		nlin[pow_idx, gain_idx]     = edfa_gain_lin* np.sum(nli_vec_ct[:, pow_idx, gain_idx], axis=0)
		nlin_co[pow_idx, gain_idx]     = edfa_gain_lin* np.sum(nli_vec_co[:, pow_idx, gain_idx], axis=0)

		ase[pow_idx, gain_idx]      = edfa_gain_lin* np.sum(ase_vec_ct[:, pow_idx, gain_idx])
		ase_raman[pow_idx, gain_idx] = np.sum(ase_vec_ct[:, pow_idx, gain_idx])
		nlin_raman[pow_idx, gain_idx] = np.sum(nli_vec_ct[:, pow_idx, gain_idx], axis=0)

		edfa_ase[pow_idx, gain_idx] = np.sum(edfa_noise[:, gain_idx])
		rec_pow[pow_idx, gain_idx]  = np.sum(pow_vec_ct[:, pow_idx, gain_idx])
		out_pow[pow_idx, gain_idx]  = edfa_gain_lin* np.sum(pow_vec_ct[:, pow_idx, gain_idx])
		print("sum", np.sum(pow_vec_ct[:, pow_idx, gain_idx]))
		print("non sum", pow_vec_ct[:, pow_idx, gain_idx] )

		# suppose to have full power restoration
		#print("edfa residual gain: ", edfa_gain_lin)
		osnr_ct[pow_idx, gain_idx]  = 10 * np.log10( 
					 np.sum(
					 (edfa_gain_lin*pow_vec_ct[:, pow_idx, gain_idx]) / (edfa_noise[:, gain_idx]
																													+ edfa_gain_lin*(ase_vec_ct[:, pow_idx, gain_idx] + nli_vec_ct[:, pow_idx, gain_idx]) )
																													/len(coi_selection_idx_average) 
						)
		)
		osnr_co[pow_idx, gain_idx]  = 10 * np.log10( 
					 np.sum(
					 (edfa_gain_lin*pow_vec_co[:, pow_idx, gain_idx]) / (edfa_noise[:, gain_idx]
																													+ edfa_gain_lin*(ase_vec_co[:, pow_idx, gain_idx] + nli_vec_co[:, pow_idx, gain_idx]) )
																													/len(coi_selection_idx_average) 
						)
		)
		osnr_ct_no_edfa_noise[pow_idx, gain_idx]  = 10 * np.log10( 
					 np.sum(
					 (edfa_gain_lin*pow_vec_ct[:, pow_idx, gain_idx]) / (edfa_gain_lin*(ase_vec_ct[:, pow_idx, gain_idx] + nli_vec_ct[:, pow_idx, gain_idx]) )
																													/len(coi_selection_idx_average) 
						)
		)
		print("In power                 : ", power)
		print("Raman out power          : ", watt2dBm(sum(pow_vec_ct[:, pow_idx, gain_idx])/len(coi_selection_idx_average)))
		print("Predicted Raman out power: ", power+gain)
		print("Out power                : ", watt2dBm(sum(edfa_gain_lin * pow_vec_ct[:, pow_idx, gain_idx]) /len
		(coi_selection_idx_average)))
		print("\n")
nlin /=       len(coi_selection_idx_average)
nlin_co /=       len(coi_selection_idx_average)

ase /=        len(coi_selection_idx_average)
ase_raman /=  len(coi_selection_idx_average)
nlin_raman /=len(coi_selection_idx_average)
edfa_ase /= len(coi_selection_idx_average)
rec_pow /=  len(coi_selection_idx_average)
out_pow /=  len(coi_selection_idx_average)

print("Plotting...")

# plt.contour(osnr_ct, levels=14, extent=(gain_dB_list[0], gain_dB_list[-1], power_dBm_list[0],power_dBm_list[-1]), cm='inferno', label=r"total OSNR [dB]")

# plt.ylabel(r"Signal input power [dBm]")
# plt.xlabel(r"DRA gain [dB]")
# plt.colorbar()
# plt.show()

def contour_plot(z, title):
	fig = go.Figure(data=
		go.Contour(z=z,
							 x=gain_dB_list, 
							 y=power_dBm_list,
							line_smoothing=0,
							contours=dict(
											showlabels = True, # show labels on contours
											labelfont = dict( # label font properties
													size = 20,
													color = 'black',
											)
							),
							colorbar=dict(
									title=title, # title here
									titleside='right'),
			),

		)


	# apply changes
	fig.update_layout(
			autosize=False,
			width= 700,
			height=700,
			font_family = "serif",
			yaxis=dict(title=r"Signal input power [dBm]"),
			xaxis=dict(title=r"DRA gain [dB]"),
			# yaxis_ticksuffix = r"$  ",
			# yaxis_tickprefix = r"$",
			# xaxis_tickprefix = r"$",
			# xaxis_ticksuffix = r"$  ",
			font_size = 30
	)
	fig.show()
	return

def combo_contour_plot(z1, z2, title):
	fig = make_subplots(rows=1, cols=2, shared_yaxes=True, vertical_spacing=0.05, horizontal_spacing=0.025)
	print(np.max(z1))
	minimum = np.min([np.min(z1), np.min(z2)])
	maximum = np.max([np.max(z1), np.max(z2)])
	fig.add_trace(go.Contour(z=z1,
							 x=gain_dB_list, 
							 y=power_dBm_list,
							line_smoothing=0,
							contours=dict(
											showlabels = True, # show labels on contours
											labelfont = dict( # label font properties
													size = 20,
													color = 'black',
											),
											start = minimum,
											end = maximum
							),
							colorbar=dict(
									tickformat = ".0f",nticks=12)
			), row=1, col=1
		)
	fig.add_trace(go.Contour(z=z2,
							 x=gain_dB_list, 
							 y=power_dBm_list,
							line_smoothing=0,
							contours=dict(
											showlabels = True, # show labels on contours
											labelfont = dict( # label font properties
													size = 20,
													color = 'black',
											),
											start = minimum,
											end = maximum
							),							
							
							colorbar=dict(
									tickformat = ".0f",
									title=title, # title here
									titleside='right',  nticks=12)
			), row=1, col=2,

		)

	# apply changes
	fig.update_layout(
			autosize=False,
			width= 1000,
			height= 700,
			font_family = "serif",
			yaxis=dict(title=r"Signal input power [dBm]"),
			# yaxis_ticksuffix = r"$  ",
			# yaxis_tickprefix = r"$",
			# xaxis_tickprefix = r"$",
			# xaxis_ticksuffix = r"$  ",
			font_size = 24
	)
	fig.update_xaxes(title=r"DRA gain (CT) [dB]",row=1, col=1) 
	fig.update_xaxes(title=r"DRA gain (CO) [dB]",row=1, col=2) 
	fig.show()
	ptly.io.write_image(fig, 'osnr_gain.png', format='png')
	return

# contour_plot(osnr_ct,        r'OSNR [dB]')
# contour_plot(osnr_co,        r'OSNR [dB]')
# contour_plot(watt2dBm(nlin_co),        r'total NLIN CO [dB]')
combo_contour_plot(osnr_ct, osnr_co, r'OSNR [dB]')

# contour_plot(watt2dBm(ase+edfa_ase), r'output ASE (DRA+EDFA) [dBm]')
# contour_plot(watt2dBm(nlin), r'output NLIN [dBm]')
# contour_plot(watt2dBm(ase_raman), r'end of Raman span ASE [dBm]')
# contour_plot(watt2dBm(nlin_raman), r'end of Raman span NLIN [dBm]')

print("Done!")