import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from scipy.interpolate import interp1d
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.utils import dBm2watt
import pynlin.constellations
import json
from multiprocessing import Pool
from scipy.special import erf, erfc
from pynlin.raman.solvers import RamanAmplifier as NumpyRamanAmplifier

'''
  Compute the time integrals
'''
def get_space_integrals(intf):
  f = open("./scripts/sim_config.json")
  data = json.load(f)
  dispersion = data["dispersion"]
  channel_spacing = data["channel_spacing"]
  wavelength = data["wavelength"]
  # 
  plt.rcParams['mathtext.fontset'] = 'stix'
  plt.rcParams['font.family'] = 'STIXGeneral'
  plt.rcParams['font.weight'] = '500'
  plt.rcParams['font.size'] = '24'
  # 
  beta2 = -pynlin.utils.dispersion_to_beta2(
      dispersion * 1e-12 / (1e-9 * 1e3), wavelength
  )
  print("beta2:", beta2)
  print(channel_spacing*1e-9)
  fiber = pynlin.fiber.Fiber(
      effective_area=80e-12,
      beta2=beta2
  )
  print("beta2: ", fiber.beta2)
  print("gamma: ", fiber.gamma)
  (m, z, I) = get_space(intf)
  print(m)
  X0mm = np.zeros_like(m)
  X0mm = pynlin.nlin.Xhkm_precomputed(z, I, amplification_function=None)
  return X0mm


'''
  get the approximation 1/(beta2 Omega)
'''
def get_space_integral_approximation(intf):
  f = open("./scripts/sim_config.json")
  data = json.load(f)
  dispersion = data["dispersion"]
  num_channels = data["num_channels"]
  channel_spacing = data["channel_spacing"]
  wavelength = data["wavelength"]
  baud_rate = data["baud_rate"]
  # INCORRECT, we have custom length
  L = 1e3*200

  beta2 = -pynlin.utils.dispersion_to_beta2(
      dispersion * 1e-12 / (1e-9 * 1e3), wavelength
  )
  # print("beta2:", beta2)
  # # print(channel_spacing*1e-9)
  wdm = pynlin.wdm.WDM(
      spacing=channel_spacing * 1e-9,
      num_channels=num_channels,
      center_frequency=190
  )
  freqs = wdm.frequency_grid()
  (m, z, I) = get_space(intf)
  X0mm_ana = np.ones_like(m, dtype=np.float64)
  L_d = 1/(baud_rate**2 *np.abs(beta2))
  Omega = 2*np.pi*(freqs[intf+1]-freqs[0])
  z_w = L_d * baud_rate / Omega
  print("z_w walkoff:", z_w)
  # maybe incorrect
  z_all = get_zm(m, Omega, beta2, baud_rate)
  for zx, z_m in enumerate(z_all):
    # implement a simple 
    # print((L-z_m))
    # TODO check correctness of the formula
    z_w_site = z_w * np.sqrt(1 + (z_m/L_d)**2)
    # print(z_w_site)
    X0mm_ana[zx] = 1/(beta2*Omega) * (1-(1-erf((L-z_m)/z_w_site))/2-(1-erf((z_m)/z_w_site))/2)
  # print("X ana = ", X0mm_ana)
  return X0mm_ana

'''
  compare the analytical approximation with the numerical.
  plot the results
'''
def compare_interferent(interfering_channels = []):
  print("========== Interfering channels = ", interfering_channels)
  f = open("./scripts/sim_config.json")
  data = json.load(f)
  dispersion = data["dispersion"]
  wavelength = data["wavelength"]
  for intf in interfering_channels:
    (m, z, I) = get_space(intf)
    X0mm = get_space_integrals(intf)
    X0mm_ana = get_space_integral_approximation(intf)
    print(X0mm**2)
    print(X0mm_ana**2)
    print("NOISE numerical  :             {:4.3e}".format(np.real(np.sum(X0mm**2))))
    print("NOISE analytical :             {:4.3e}".format(-2*np.real(np.sum(X0mm_ana**2))))
    print("RELATIVE ERROR on noise term : {:4.3e}".format(np.real((np.sum(X0mm_ana**2))/np.sum(X0mm**2))-1.0))
    beta2 = -pynlin.utils.dispersion_to_beta2(
        dispersion * 1e-12 / (1e-9 * 1e3), wavelength
    )
    plt.clf() 
    for mx in m:
      plt.axvline(x=mx, lw=0.3, color="gray", ls="dotted")
      plt.plot(m, np.real(X0mm_ana), color="gray", ls="dashed")
      plt.plot(m, np.real(X0mm), color="red", ls="solid")
      plt.xlabel(r"$m$")
      plt.ylabel(r"$X_{\mathrm{0mm}}$")
      plt.tight_layout()
      plt.savefig('media/Interferent_'+str(intf)+'.pdf')
  
  # for intf in interfering_channels:
  #   (m, z, I) = get_space(intf)
  #   X0mm = np.zeros_like(m)
  #   X0mm = get_space_integrals(intf)
  #   X0mm_ana = get_space_integral_approximation(intf)
  #   error = (X0mm-X0mm_ana)/X0mm
  #   plt.plot(m, error)
  #   # plt.scatter(m, error, marker="x", color=colors[intf])
  #   plt.plot(m, X0mm_ana*0.0, color="gray", ls="dashed")
  #   plt.ylim([-0.1, 0.0])
  #   plt.grid()
  #   for mx in m:
  #     plt.axvline(x=mx, lw=0.3, color="gray", ls="dotted")
  #     plt.xlabel(r"$m$")
  #     plt.ylabel(r"$\varepsilon_R$")
  #     plt.tight_layout()
  #     plt.savefig('media/error_'+str(intf)+'.pdf')
  return


'''
  read from file the integral data
'''
def get_space(intf):
  f = open("./scripts/sim_config.json")
  data = json.load(f)
  num_channels = data["num_channels"]
  # 
  time_integrals_results_path = 'results/'
  f_general = h5py.File(time_integrals_results_path + 'general_results_alt.h5', 'r')
  for key in f_general.keys():
    print(key)
  # print("_________________ WDM _________________")
  m = np.array(f_general['/time_integrals/channel_0/interfering_channel_' + str(intf) + '/m'])
  z = np.array(f_general['/time_integrals/channel_0/interfering_channel_' + str(intf) + '/z'])
  I = np.array(f_general['/time_integrals/channel_0/interfering_channel_' + str(intf) + '/integrals'])
  return (m, z, I)

def get_zm(m, Omega, beta2, baud_rate):
  return m/(baud_rate*beta2*Omega)
  