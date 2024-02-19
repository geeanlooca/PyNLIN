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
  Read the time integrals from the result file
'''
def get_space(intf):
  time_integrals_results_path = 'results/'
  f_general = h5py.File(time_integrals_results_path + 'general_results_alt.h5', 'r')
  m = np.array(f_general['/time_integrals/channel_0/interfering_channel_' + str(intf) + '/m'])
  z = np.array(f_general['/time_integrals/channel_0/interfering_channel_' + str(intf) + '/z'])
  I = np.array(f_general['/time_integrals/channel_0/interfering_channel_' + str(intf) + '/integrals'])
  return (m, z, I)

def get_zm(m, Omega, beta2, baud_rate):
  return m/(baud_rate*beta2*Omega)


'''
  Read the time integral file and compute the space integrals 
'''
def get_space_integrals(intf):
  (m, z, I) = get_space(intf)
  X0mm = np.zeros_like(m)
  X0mm = pynlin.nlin.Xhkm_precomputed(z, I, amplification_function=None)
  return X0mm


'''
  Get the approximation 1/(beta2 Omega)*(1-1/2*erf(start)-1/2*erf(end))
'''
def get_space_integral_approximation(intf):
  with open("./scripts/sim_config.json") as f:
    data = json.load(f)
    dispersion = data["dispersion"]
    num_channels = data["num_channels"]
    channel_spacing = data["channel_spacing"]
    wavelength = data["wavelength"]
    baud_rate = data["baud_rate"]
    L = data["fiber_length"][0]

    beta2 = -pynlin.utils.dispersion_to_beta2(
        dispersion * 1e-12 / (1e-9 * 1e3), wavelength
    )
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
    assert(Omega == channel_spacing*2*np.pi)
    z_w = L_d * baud_rate / Omega
    print("z_walkoff:", z_w)
    
    # get all the collision peak locations. They are not given in the result file 
    z_all = get_zm(m, Omega, beta2, baud_rate)
    for zx, z_m in enumerate(z_all):
      # find Gaussian width at collision peak
      z_w_site = z_w * np.sqrt((1 + (z_m/L_d)**2))
      # compute the Gaussian integral
      X0mm_ana[zx] = 1/(beta2*Omega) * (1 - erfc((L-z_m)/z_w_site)/2 - erfc((z_m)/z_w_site)/2)
  return X0mm_ana


'''
  compare the analytical approximation with the numerical.
  plot the results
'''
def compare_interferent(interfering_channels = []):
  for intf in interfering_channels:
    (m, z, I) = get_space(intf)
    X0mm = get_space_integrals(intf)
    X0mm_ana = get_space_integral_approximation(intf)
    print("NOISE numerical         = {:4.3e}".format(np.real(np.sum(X0mm**2))))
    print("NOISE analytical        = {:4.3e}".format(np.real(np.sum(X0mm_ana**2))))
    print("RELATIVE ERROR on noise = {:4.3e}".format(np.real((np.sum(X0mm_ana**2))/np.sum(X0mm**2))-1.0))

    plt.clf() 
    for mx in m:
      plt.axvline(x=mx, lw=0.3, color="gray", ls="dotted")
    plt.plot(m, np.real(X0mm_ana), color="gray", ls="dashed", label="analytical")
    plt.plot(m, np.real(X0mm), color="red", ls="solid", label="numerical")
    plt.xlabel(r"$m$")
    plt.ylabel(r"$X_{\mathrm{0mm}}$")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('media/Interferent_'+str(intf)+'.pdf')
    plt.show()
    # X0mm is supposed to be real
    error = (X0mm-X0mm_ana)/X0mm
    plt.plot(m, np.real(error))
    # plt.scatter(m, error, marker="x", color=colors[intf])
    # plt.ylim([-0.1, 0.05])
    plt.grid()
    for mx in m:
      plt.axvline(x=mx, lw=0.3, color="gray", ls="dotted")
    plt.xlabel(r"$m$")
    plt.ylabel(r"$\varepsilon_R$")
    plt.ylim([-0.1, 0.005])
    plt.tight_layout()
    plt.savefig('media/error_'+str(intf)+'.pdf')
  return



  