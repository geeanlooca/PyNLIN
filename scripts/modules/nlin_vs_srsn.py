import matplotlib.pyplot as plt
import numpy as np
import os
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.constellations
import json
from multiprocessing import Pool

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


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'

# PLOTTING PARAMETERS
interfering_grid_index = 1
#power_dBm_list = [-20, -10, -5, 0]
power_dBm_list = np.linspace(-20, 0, 11)
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
wdm_bandwidths = np.linspace(2, 15, 20) # in THz
xi_N = np.ones(len(wdm_bandwidths))*  16/9*fiber.gamma**2

for fiber_length in fiber_lengths:
  length_setup = int(fiber_length * 1e-3)
  plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_"+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_ct)+'_ct_'+special+'/'
  #
  if not os.path.exists(plot_save_path):
      os.makedirs(plot_save_path)

  xi_R = []
  for band in wdm_bandwidths:
    num_channels =int(band*1e12/channel_spacing)
    wdm = pynlin.wdm.WDM(
        spacing = channel_spacing * 1e-9,
        num_channels=num_channels,
        center_frequency=190
    )
    amplifier = NumpyRamanAmplifier(fiber)
    c_r_matrix = amplifier.compute_gain_matrix(wdm.frequency_grid())
    print((c_r_matrix[1, -1]**2)/4)
    xi_R.append((c_r_matrix[1, -1]**2)/4)


  fig_arity = plt.figure(figsize=(10, 6))
  # normalized to 16-QAM variance
  plt.plot(wdm_bandwidths, np.divide(xi_R, (xi_R+xi_N)), marker='x', markersize=10, color='black')
  plt.grid()
  plt.minorticks_on()
  plt.xlabel("WDM bandwidth [THz]")
  plt.ylabel(r"$\xi_R/(\xi_R+\xi_N)$")
  plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, bottom=-1)

  fig_arity.tight_layout()
  fig_arity.savefig(plot_save_path+"comparison.pdf")