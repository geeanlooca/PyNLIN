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
from pynlin.fiber import MMFiber
from matplotlib.gridspec import GridSpec
import json
rc('text', usetex=True)
logging.basicConfig(filename='MMF_optimizer.log', encoding='utf-8', level=logging.INFO)
log = logging.getLogger(__name__)
log.debug("starting to load itu_config.json")

f = open("./scripts/itu_config.json")
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



beta1_params = load_dummy_group_delay()
# beta1_params = load_dummy_group_delay()
# print(beta1_params.shape)
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

fiber = MMFiber(
    effective_area=80e-12,
    overlap_integrals = oi_fit,
    group_delay = beta1_params,
    length=100e3
)

beta1 = np.zeros((len(modes), len(freqs)))
for i in modes:
  beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
beta2 = np.zeros((len(modes), len(freqs)))
for i in modes:
  beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
beta1 = np.array(beta1)
beta2 = np.array(beta2)
# print(beta2[1, :])

plt.clf()
sns.heatmap(beta1, cmap="coolwarm", square=False, xticklabels=freqs, yticklabels=modes)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Modes')
plt.title('Beta1 Heatmap')
plt.savefig("media/dispersion/disp.png", dpi=dpi)

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
print("Unlucky channel has noise: ", np.min(nlin))
print("Lucky channel has noise: ", np.max(nlin))

nlin_no_cross = np.zeros((len(modes), len(freqs)))
for i in range(len(modes)):
  for j in range(len(freqs)):
      nlin_no_cross[i, j] = np.sum(L / (np.abs(beta1[i, :] - beta1[i, j])[(beta1[i, :] - beta1[i, j]) != 0] * T))

# plt.clf(
# sns.heatmap(collisions, cmap="magma", square=False, xticklabels=freqs, yticklabels=modes)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Modes')
# plt.title('Total number of collision due to system')
# plt.savefig("media/dispersion/disp.png")
# plt.show()


plt.clf()
for i in range(4):
    plt.plot(freqs * 1e-12, collisions[i, :], label=mode_names[i])
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$m_{\mathrm{max}}$')
plt.legend()
# plt.grid(grid)
plt.savefig(f"media/dispersion/collisions.png", dpi=dpi)
# plt.show()

plt.clf()
plt.plot(freqs * 1e-12, collisions_single[0, :], label=mode_names[i])
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$m_{\mathrm{max}}$')
plt.legend()
plt.grid(grid)
plt.savefig(f"media/dispersion/collisions_single.png", dpi=dpi)
# plt.show()

plt.clf()
for i in range(len(modes)): 
  plt.semilogy(freqs * 1e-12, nlin_no_cross[i, :], label=mode_names[i], marker='x')
plt.xlabel('Frequency (THz)')
plt.ylabel('NLIN coeff')
plt.legend()
plt.grid(grid)
plt.tight_layout()
plt.savefig(f"media/dispersion/nlin_no_cross.png", dpi=dpi)
# plt.show()

plt.clf()
for i in range(4):
    plt.semilogy(freqs * 1e-12, nlin[i, :], label=mode_names[i], marker='x')
plt.xlabel('Frequency (THz)')
plt.ylabel('NLIN coeff')
plt.legend()
plt.grid(grid)
plt.tight_layout()
plt.savefig(f"media/dispersion/nlin.png", dpi=dpi)
# plt.show()

plt.clf()
plt.figure(figsize=(4.6, 4))
for i in range(4):
    plt.plot(freqs * 1e-12, beta1[i, :] * 1e9, label=mode_names[i])
minn = np.min(beta1)
maxx = np.max(beta1)

# plt.axvline(205.5, color="gray", ls="dashed" , lw=0.5)
# plt.axvline(196.07, color="gray", ls="dashed", lw=0.5)
# plt.axvline(191.69, color="gray", ls="dashed", lw=0.5)

plt.axvline(190.9, color="red", ls="dotted", lw=1.5)
plt.axvline(200.9, color="red", ls="dotted", lw=1.5)

# # plt.xticks([185, 193, 196, 206])
# freq_boundaries = [189, 192.7, 197, 206]
# for i, label in enumerate(['L', 'C', 'S', 'E']):
#     plt.text(freq_boundaries[i], 4.8924, label, ha='center', va='bottom')
    
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\beta_1$ (\mu s/km)')
plt.legend()
# plt.grid(grid)
plt.tight_layout()
plt.savefig(f"media/dispersion/beta1.png", dpi=dpi)

plt.clf()
plt.figure(figsize=(4.6, 4))

for i in range(4):
    plt.plot(freqs * 1e-12, beta2[i, :] * 1e27, label=mode_names[i])
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\beta_2$ (ps$^2$/km)')
plt.legend()
# plt.grid(grid)
plt.tight_layout()
plt.savefig(f"media/dispersion/beta2.png", dpi=300)

beta1_differences = np.abs(beta1[:, :, np.newaxis, np.newaxis] - beta1[np.newaxis, np.newaxis, :, :])
beta1_differences = beta1_differences[beta1_differences!= 0]

mask = (beta1_differences < 200 * 1e-12)
hist, edges = np.histogram(beta1_differences[mask]*1e12, bins=200)
hist = hist / 2.0
plt.clf()
plt.figure(figsize=(4, 3.5))
plt.bar(edges[:-1], hist, width=np.diff(edges), zorder=3)
plt.xlabel('DGD (ps/m)')
plt.ylabel('channel pair count')
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(f"media/dispersion/DGD_histogram.png", dpi=dpi)

# mask = (beta1_differences < 0.002 * 1e-12)
print("Average DGD: ", np.mean(beta1_differences * 1e12))
# hist, edges = np.histogram(beta1_differences[mask]*1e12, bins=200)
# hist = hist / 2.0
# plt.clf()
# plt.figure(figsize=(4, 3.5))
# plt.bar(edges[:-1], hist, width=np.diff(edges), zorder=3)
# plt.xlabel('DGD (ps/m)')
# plt.ylabel('channel pair count')
# plt.grid(axis='y', zorder=0)
# plt.tight_layout()
# plt.savefig(f"media/dispersion/DGD_histogram_zoom.png", dpi=dpi)

fig = plt.figure(figsize=(6, 6))  # Overall figure size
gs = GridSpec(nrows=3, ncols=1, height_ratios=[2, 1, 1])  # The height_ratios adjust the relative sizes

# Create subplots
ax1 = fig.add_subplot(gs[0])  # Top subplot (smaller)
ax2 = fig.add_subplot(gs[1])  # Bottom subplot (larger)
ax3 = fig.add_subplot(gs[2])  # Bottom subplot (larger)
# Plot histogram on the top subplot
ax1.bar(edges[:-1], hist, width=np.diff(edges), zorder=3)
ax1.set_ylabel('Frequency')
ax1.grid(axis='y', zorder=0)
ax2.plot(edges[:-1], edges[:-1]*L/T*1e-12, color='blue')
ax2.set_ylabel(r'$m_{\mathrm{max}}$')
ax3.semilogy(edges[:-1], L/T / edges[:-1], color='red')
ax3.set_ylabel('partial NLIN')
ax3.set_xlabel('DGD (ps/m)')
# ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"media/dispersion/DGD_collisions.png", dpi=dpi)

plt.clf()
plt.figure(figsize=(4.6, 4))
for i in range(4):
    plt.plot(freqs * 1e-12, (beta1[i, :] - beta1[1, :])
             * 1e12, label=mode_names[i])
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\Delta\beta_1$ (ps/m)')
plt.legend()
plt.grid(grid)
plt.tight_layout()
plt.savefig(f"media/dispersion/DMGD_LP01.png", dpi=dpi)