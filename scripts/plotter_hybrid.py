# This script plot
# hybrid OSRN tradeoff vs signal input power, Raman gain

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
import pickle
from matplotlib.lines import Line2D

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
Delta_theta_2_ct = np.zeros_like(
		np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(gain_dB_list)))
)
R_ct =   np.zeros_like(Delta_theta_2_ct)

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
    noise_path = '../noises/'

    # retrieve ASE and SIGNAL power at receiver
    X_ct = np.zeros_like(
        np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(gain_dB_list)))
    )
    
    T_ct = np.zeros_like(X_ct)

    ase_ct = np.zeros_like(X_ct)

    power_at_receiver_ct = np.zeros_like(X_ct)
    
    all_ct_pumps = np.zeros((len(power_dBm_list), 4))

    avg_pump_dBm_ct = np.zeros((len(power_dBm_list)))


    for gain_idx, gain_dB in enumerate(gain_dB_list):
      for pow_idx, power_dBm in enumerate(power_dBm_list):
          # PUMP power evolution
          pump_solution_ct = np.load(results_path_ct + 'pump_solution_ct_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')
          # SIGNAL power evolution
          signal_solution_ct = np.load(results_path_ct + 'signal_solution_ct_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')
          # ASE power evolution
          ase_solution_ct = np.load(results_path_ct + 'ase_solution_ct_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB)  + '.npy')
          z_max = np.linspace(0, fiber_length, np.shape(pump_solution_ct)[0])

          # compute the X0mm coefficients given the precompute time integrals
          # FULL X0mm EVALUATION FOR EVERY m =======================
          all_ct_pumps[pow_idx, :] = watt2dBm(pump_solution_ct[-1, :])
          avg_pump_dBm_ct[pow_idx] = watt2dBm(np.mean(pump_solution_ct[-1, :]))

          for coi_idx, coi in enumerate(coi_list):
              power_at_receiver_ct[coi_idx, pow_idx, gain_idx] = signal_solution_ct[-1, coi_idx]
              ase_ct[coi_idx, pow_idx, gain_idx] = ase_solution_ct[-1, coi_idx]

    # Retrieve sum of X0mm^2: noises
    X_ct =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + "_opt_gain_" + str(gain_dB) + '_ct_X_ct.npy')
    
    # Retrieve sum of X0mm^2: noises
    T_ct =   np.load('../noises/'+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + "_opt_gain_" + str(gain_dB) + '_ct_T_ct.npy')
    # choose modulation format and compute phase noise
    M = 16

    for gain_idx, gain_dB in enumerate(gain_dB_list):
      for pow_idx, power_dBm in enumerate(power_dBm_list):
          average_power = dBm2watt(power_dBm)
          qam = pynlin.constellations.QAM(M)
          qam_symbols = qam.symbols()

          # assign specific average optical energy
          qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power / baud_rate)
          constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)
          for coi_idx, coi in enumerate(coi_list):
              print("\n\nreassinging delta theta")
              Delta_theta_2_ct[coi_idx,   pow_idx, gain_idx] =   16/9 * fiber.gamma**2 * constellation_variance * np.abs(X_ct[coi_idx,   pow_idx, gain_idx])
              R_ct[coi_idx, pow_idx] = constellation_variance * np.abs(T_ct[coi_idx, pow_idx])

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
coi_selection_average = [0, 9, 19, 29, 39, 49]
coi_selection_idx_average = [0, 1, 2, 3, 4, 5]

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
for chan_idx, freq in enumerate(freq_list):
  for gain_idx, gain_dB in enumerate(gain_dB_list):
    G = 10^((total_gain_dB-gain_dB)/10) # the remaining part of the gain for full compensation
    edfa_noise[chan_idx, gain_idx] = h_planck * freq * NG * (G - 1) 

osnr_ct = np.ndarray(shape=(len(power_dBm_list), len(gain_dB_list)))

# vectorized over power and gain
for scan in range(len(coi_selection_average)):
  osnr_ct += 10 * np.log10(power_at_receiver_ct[coi_selection_idx_average[scan], :, :]/   (P_A * Delta_theta_2_ct[coi_selection_idx_average[scan], :, :] + ase_ct[coi_selection_idx_average[scan], :, :]))
osnr_ct /= len(coi_selection_idx_average)


plt.imshow(osnr_ct, cmap='hot', interpolation='nearest')
plt.show()