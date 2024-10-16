import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.graph_objects as go
import seaborn as sns
import pynlin.wdm
from pynlin.utils import nu2lambda
from scripts.modules.load_fiber_values import load_group_delay, load_dummy_group_delay
from numpy import polyval
from pynlin.fiber import *
from pynlin.nlin import compute_all_collisions_time_integrals, get_dgd, X0mm_space_integral
from matplotlib.gridspec import GridSpec
import json
rc('text', usetex=True)
logging.basicConfig(filename='MMF_optimizer.log', encoding='utf-8', level=logging.INFO)
log = logging.getLogger(__name__)
log.debug("starting to load sim_config.json")
f = open("./scripts/sim_config.json")

data = json.load(f)
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
use_avg_oi = False


def get_space_integrals(m, z, I):
    '''
      Read the time integral file and compute the space integrals 
    '''
    X0mm = np.zeros_like(m)
    X0mm = X0mm_space_integral(z, I, amplification_function=None)
    return X0mm


beta1_params = load_dummy_group_delay()

dpi = 300
grid = False
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=center_frequency
)
freqs = wdm.frequency_grid()
modes = [0, 1, 2, 3]
mode_names = ['LP01', 'LP11', 'LP21', 'LP02']

dummy_fiber = MMFiber(
    effective_area=80e-12,
    overlap_integrals=oi_fit,
    group_delay=beta1_params,
    length=100e3,
    n_modes=4
)

beta1 = np.zeros((len(modes), len(freqs)))
for i in modes:
    beta1[i, :] = dummy_fiber.group_delay.evaluate_beta1(i, freqs)
beta2 = np.zeros((len(modes), len(freqs)))
for i in modes:
    beta2[i, :] = dummy_fiber.group_delay.evaluate_beta2(i, freqs)
beta1 = np.array(beta1)
beta2 = np.array(beta2)

pulse = pynlin.pulses.GaussianPulse(
    baud_rate=baud_rate,
    num_symbols=1e2,
    samples_per_symbol=2**5,
    rolloff=0.0,
)

n_samples = 50
partial_nlin = np.zeros(n_samples)
dgd1 = 1e-12
dgd2 = 6e-12
# 6e-9 for our fiber
dgds =  np.linspace(dgd1, dgd2, n_samples)
dgds2 = np.linspace(dgd1, dgd2, n_samples)
if True:
  for id, dgd in enumerate(dgds):
      # print(f"DGD: {dgd:10.3e}")
      z, I, m = compute_all_collisions_time_integrals(
          (-1, -1), (-1, -1), dummy_fiber, wdm, pulse, dgd)
      # space integrals
      X0mm = get_space_integrals(m, z, I)
      partial_nlin[id] = np.sum(X0mm**2)
  np.save("results/partial_nlin.npy", partial_nlin)
  
partial_nlin = np.load("results/partial_nlin.npy")
# for each channel, we compute the total number of collisions that
# needs to be computed for evaluating the total noise on that channel.
T = 100e-12
L = dummy_fiber.length
print(partial_nlin)
fig = plt.figure(figsize=(4, 3))  # Overall figure size
# plt.plot(dgds2 * 1e9, L / (T * dgds2), color='red', label='approximation')
plt.plot(dgds * 1e9, (partial_nlin), color='green', label='numerics')
plt.legend()
plt.ylabel('partial NLIN')
plt.xlabel('DGD (ns/m)')
plt.tight_layout()
plt.savefig(f"media/dispersion/partial_NLIN.png", dpi=dpi)

fig = plt.figure(figsize=(4, 3))  # Overall figure size
plt.plot(dgds2 * 1e9, ((L / (T * dgds2))-partial_nlin)/partial_nlin, color='red')
plt.legend()
plt.ylabel('partial NLIN')
plt.xlabel('DGD (ns/m)')
plt.tight_layout()
plt.savefig(f"media/dispersion/rel_error_of_mecozzi.png", dpi=dpi)