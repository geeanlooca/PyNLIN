"""
script for finding the overall noise on a single channel 
given a fiber link configuration consisting 
"""

# A workaround for setting the correct workind dir.
# beware to run only once per session!
# is there a better way? :)

from scripts.modules.space_integrals_general import *
from scripts.modules.time_integrals import do_time_integrals
from scripts.modules.load_fiber_values import *
import matplotlib.pyplot as plt
import scipy
from pynlin import *

with open("./scripts/sim_config.json") as f:
  f = open("./scripts/sim_config.json")
  
data = json.load(f)
# print(data)
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
num_only_co_pumps = data['num_only_co_pumps']
num_only_ct_pumps = data['num_only_ct_pumps']
gain_dB_setup = data['gain_dB_list']
gain_dB_list = np.linspace(gain_dB_setup[0], gain_dB_setup[1], gain_dB_setup[2])
power_dBm_setup = data['power_dBm_list']
power_dBm_list = np.linspace(power_dBm_setup[0], power_dBm_setup[1], power_dBm_setup[2])
num_modes = data['num_modes']
oi_fit = np.load('oi_fit.npy')
oi_avg = np.load('oi_avg.npy')

length = data["fiber_length"][0]

beta2 = -pynlin.utils.dispersion_to_beta2(
    dispersion, wavelength
)

wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=center_frequency
)

fiber = pynlin.fiber.SMFiber(
      effective_area=80e-12,
      beta2=beta2, 
      length=length
)
print(f"beta_2 = {beta2:.9e}")

# make the time integral take as an input (pulse, fiber, wdm)
pulse = pynlin.pulses.GaussianPulse(
  baud_rate = baud_rate,
  num_symbols = 1e3, # ???
  samples_per_symbol = 2**5,
  rolloff = 0.1,
)
 
freqs = wdm.frequency_grid()

s_limit = 1460e-9
l_limit = 1625e-9
s_freq = 3e8/s_limit
l_freq = 3e8/l_limit

print(s_freq*1e-12)
print(l_freq*1e-12)
delta = (s_freq - l_freq) *1e-12
avg = print((s_freq+l_freq) *1e-12 /2)
beta_file = './results/fitBeta.mat'
mat = scipy.io.loadmat(beta_file)['fitParams'] * 1.0

beta_file = './results/fitBeta.mat'
omega = 2 * np.pi * freqs
omega_norm = scipy.io.loadmat(beta_file)['omega_std']
omega_n = (omega - scipy.io.loadmat(beta_file)['omega_mean']) / omega_norm
beta1 = np.zeros((4, len(freqs)))

for i in range(4):
    beta1[i, :] = (3 * mat[i, 0] * (omega_n ** 2) + 2 *
                   mat[i, 1] * omega_n + mat[i, 2]) / omega_norm

# write the results file in ../results/general_results.h5 with the correct time integrals
# the file contains, for each interferent channel, the values (z, m, I) of the z
# channel of interest is set to channel 0, and interferent channel index start from 0 for simplicity
a_chan = (0,0)
print("@@@@@@@@ Time integrals  @@@@@@@@")
do_time_integrals(a_chan, fiber, wdm, pulse, overwrite=True)
print("@@@@@@@@ Space integrals @@@@@@@@")
compare_interferent(a_chan, [(0, 2)], fiber, wdm, pulse)