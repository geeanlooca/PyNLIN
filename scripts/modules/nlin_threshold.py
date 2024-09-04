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
from pynlin.nlin import compute_all_collisions_time_integrals, get_dgd
from matplotlib.gridspec import GridSpec
import json
rc('text', usetex=True)
logging.basicConfig(filename='MMF_optimizer.log', encoding='utf-8', level=logging.INFO)
log = logging.getLogger(__name__)
log.debug("starting to load sim_config.json")
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
use_avg_oi = False

def get_space_integrals(m, z, I):
    '''
      Read the time integral file and compute the space integrals 
    '''
    X0mm = np.zeros_like(m)
    X0mm = pynlin.nlin.X0mm_space_integral(z, I, amplification_function=None)
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
    overlap_integrals = oi_fit,
    group_delay = beta1_params,
    length=100e3,
    n_modes = 4
)

beta1 = np.zeros((len(modes), len(freqs)))
for i in modes:
  beta1[i, :] = dummy_fiber.group_delay.evaluate_beta1(i, freqs)
beta2 = np.zeros((len(modes), len(freqs)))
for i in modes:
  beta2[i, :] = dummy_fiber.group_delay.evaluate_beta2(i, freqs)
beta1 = np.array(beta1)
beta2 = np.array(beta2)
# print(beta2[1, :])

pulse = pynlin.pulses.GaussianPulse(
    baud_rate=baud_rate,
    num_symbols=1e2,
    samples_per_symbol=2**5,
    rolloff=0.0,
)

a_chan = (0, 0)
freqs = wdm.frequency_grid()
partial_nlin = np.zeros(len(freqs))
dgds = np.zeros_like(partial_nlin)
for ib, b_chan in enumerate([(0, i) for i in range(len(freqs))]):
  # compute time integrals
  dgd = get_dgd(a_chan, b_chan, dummy_fiber, wdm)
  print(f"DGD: {dgd:10.3e}")
  z, I, m = compute_all_collisions_time_integrals(a_chan, b_chan, dummy_fiber, wdm, pulse)
  print(m.shape)
  print(z.shape)
  print(I[1].shape)
  # space integrals
  X0mm = get_space_integrals(m, z, I)
  partial_nlin[ib] = np.sum(X0mm**2)
  dgds[ib] = np.abs(dgd) * 1e12

# for each channel, we compute the total number of collisions that
# needs to be computed for evaluating the total noise on that channel.
T = 100e-12
L = 100e3

collisions = np.zeros((len(modes), len(freqs)))
for i in range(len(modes)):
    for j in range(len(freqs)):
        collisions[i, j] = np.floor(np.abs(np.sum(beta1 - beta1[i, j])) * L / T)


collisions_single = np.zeros((1, len(freqs)))
for j in range(len(freqs)):
    collisions_single[0, j] = np.floor(np.abs(np.sum(beta1[0 :] - beta1[0, j])) * L / T)
        
nlin = np.zeros((len(modes), len(freqs)))
for i in range(len(modes)):
    for j in range(len(freqs)):
        nlin[i, j] = np.sum(L / (np.abs(beta1 - beta1[i, j])[(beta1 - beta1[i, j]) != 0] * T))

nlin_no_cross = np.zeros((len(modes), len(freqs)))
for i in range(len(modes)):
  for j in range(len(freqs)):
      nlin_no_cross[i, j] = np.sum(L / (np.abs(beta1[i, :] - beta1[i, j])[(beta1[i, :] - beta1[i, j]) != 0] * T))

beta1_differences = np.abs(beta1[:, :, np.newaxis, np.newaxis] - beta1[np.newaxis, np.newaxis, :, :])
beta1_differences = beta1_differences[beta1_differences!= 0]

mask = (beta1_differences < 200 * 1e-12)
hist, edges = np.histogram(beta1_differences[mask]*1e12, bins=200)

fig =plt.figure(figsize=(6, 3))  # Overall figure size
plt.semilogy(edges[:-1], L / T / edges[:-1] * 1e12, color='red')
plt.semilogy(dgds, partial_nlin, color='green')
plt.ylabel('partial NLIN')
plt.xlabel('DGD (ps/m)')
plt.tight_layout()
plt.savefig(f"media/dispersion/partial_NLIN.png", dpi=dpi)