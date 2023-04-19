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
from multiprocessing import Pool

from pynlin.raman.response import gain_spectrum, impulse_response
from pynlin.raman.solvers import RamanAmplifier as NumpyRamanAmplifier

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
power_dBm_setup=data['power_dBm_list']
power_dBm_list = np.linspace(power_dBm_setup[0], power_dBm_setup[1], power_dBm_setup[2])

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'

# PLOTTING PARAMETERS
interfering_grid_index = 1

arity_list = [16]
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
time_integrals_results_path = '../results/'

f_0_9 = h5py.File(time_integrals_results_path + '0_9_results.h5', 'r')
f_19_29 = h5py.File(time_integrals_results_path + '19_29_results.h5', 'r')
f_39_49 = h5py.File(time_integrals_results_path + '39_49_results.h5', 'r')
file_length = time_integral_length
noise_path = '../noises/'
if not os.path.exists(noise_path):
		os.makedirs(noise_path)
                

for fiber_length in fiber_lengths:
  X_ct = np.zeros_like(
      np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(gain_dB_list)))
  )
  T_ct = np.zeros_like(X_ct)
  for gain_idx, gain_dB in enumerate(gain_dB_list):
    print("Computing gain", gain_dB)
    length_setup = int(fiber_length * 1e-3)
    #
    results_path_ct = '../results_' + str(length_setup) + '/' + str(num_only_ct_pumps) + '_ct/'

    # overall NLIN sum of variances for all m

    
    '''
    load the fB(z) function and compute the space integral
    '''
    def space_integral_power(power_dBm):
      j = np.array(+1j)
      X_ct_pow = np.zeros_like(
        np.ndarray(shape=(len(coi_list)))
      )

      T_ct_pow =   np.zeros_like(X_ct_pow)
      print("Computing power ", power_dBm)
      average_power = dBm2watt(power_dBm)
      # SIMULATION DATA LOAD =================================
      pump_solution_ct =   np.load(results_path_ct + 'pump_solution_ct_power'   + str(power_dBm)+ "_opt_gain_" + str(gain_dB) + '.npy')
      signal_solution_ct = np.load(results_path_ct + 'signal_solution_ct_' + str(power_dBm)+ "_opt_gain_" + str(gain_dB) + '.npy')

      # compute fB squaring
      pump_solution_ct = np.divide(pump_solution_ct, pump_solution_ct[0, :])
      sampled_fB_ct = np.divide(signal_solution_ct, signal_solution_ct[0, :])
      z_max = np.linspace(0, fiber_length, np.shape(pump_solution_ct)[0])

      # compute the X0mm coefficients given the precompute time integrals
      # FULL X0mm EVALUATION FOR EVERY m =======================
      for coi_idx, coi in enumerate(coi_list):
        print("Computing Channel Of Interest ", coi + 1)

        # compute the first num_channels interferents (assume the WDM grid is identical)
        interfering_frequencies = pynlin.nlin.get_interfering_frequencies(
            wdm.frequency_grid()[coi], wdm.frequency_grid())
        delta_frequencies = interfering_frequencies-wdm.frequency_grid()[coi]
        # print("\nWDM frequencies : ",interfering_frequencies)
        # print("\n COI frequency: ", wdm.frequency_grid())
        # print("\nDelta frequencies : ",delta_frequencies)
        amplifier = NumpyRamanAmplifier(fiber)
        c_r_matrix = amplifier.compute_gain_matrix(wdm.frequency_grid())

        pbar_description = "Computing space integrals"
        collisions_pbar = tqdm.tqdm(interfering_frequencies, leave=False)
        collisions_pbar.set_description(pbar_description)
        for incremental, interf_index in enumerate(collisions_pbar):
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

            #upper cut z
            z = np.array(list(filter(lambda x: x <= fiber_length, z)))
            net_m = m[partial_collision_margin:len(
                m) - partial_collision_margin]
            I = I[:, :len(z)]
            m = m[:int((len(net_m)) * (fiber_length / file_length)) +
                    2 * partial_collision_margin]
            fB_ct = interp1d(z_max, sampled_fB_ct[:, incremental], kind='linear')

            X0mm_ct = pynlin.nlin.Xhkm_precomputed(z, I, amplification_function=fB_ct(z))

            X_ct_pow[coi_idx] += (np.sum(np.abs(X0mm_ct)**2))
            
            c_r = c_r_matrix[incremental]
            c_r = c_r[np.arange(len(c_r)) != incremental]
            
            gamma_srsn = fiber.gamma 
            T_ct_pow[coi_idx] +=   (np.sum(np.abs(2*j*gamma_srsn+c_r[incremental]/2)**2 * np.abs(X0mm_ct)**2))
      return [[X_ct_pow], [T_ct_pow]]


    compute = ["X", "T"]
    with Pool(os.cpu_count()) as p:
      result = p.map(space_integral_power, power_dBm_list)
    for pp_idx, pp in enumerate(power_dBm_list):
      X_ct[:, pp_idx, gain_idx] =   result[pp_idx][0][0]
      T_ct[:, pp_idx, gain_idx] =   result[pp_idx][1][0]
    print("\n\t result for gain: ", gain_dB, " = ", X_ct)


  if "X" in compute:
    np.save(noise_path+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_X_ct.npy', X_ct)

  if "T" in compute:
    np.save(noise_path+str(length_setup) + '_' + str(num_co) + '_co_' + str(num_ct) + '_ct_T_ct.npy', T_ct)