import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.graph_objects as go
import seaborn as sns
import pynlin.wdm
from pynlin.utils import nu2lambda
 
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


s_limit = 1460e-9
l_limit = 1625e-9
s_freq = 3e8/s_limit
l_freq = 3e8/l_limit

print(s_freq*1e-12)
print(l_freq*1e-12)
delta = (s_freq - l_freq) *1e-12
print(delta)
avg = print((s_freq+l_freq) *1e-12 /2)
beta_file = './results/fitBeta.mat'
mat = scipy.io.loadmat(beta_file)['fitParams'] * 1.0

print(mat)

# Load the WDM grid and fiber characteristic and compute the number of collisions


beta2 = -pynlin.utils.dispersion_to_beta2(
    dispersion, wavelength
)
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=center_frequency
)

freqs = wdm.frequency_grid()
# high = 214.2e12
# low  = 187.5e12

print(nu2lambda(np.max(freqs))*1e9)
print(nu2lambda(np.min(freqs))*1e9)
modes = [0, 1, 2, 3]

# from the Matlab file the fit is:
# 3 * fitresult.p1.*(omega_n).^2 + 2 * fitresult.p2.*omega_n +fitresult.p3)./std(omega)
# omega_n is the centered rescaled vector
omega = 2 * np.pi * freqs
omega_norm = scipy.io.loadmat(beta_file)['omega_std']
omega_n = (omega - scipy.io.loadmat(beta_file)['omega_mean']) / omega_norm
beta1 = np.zeros((4, len(freqs)))

for i in range(4):
    beta1[i, :] = (3 * mat[i, 0] * (omega_n ** 2) + 2 *
                   mat[i, 1] * omega_n + mat[i, 2]) / omega_norm

plt.clf()
sns.heatmap(beta1, cmap="coolwarm", square=False, xticklabels=freqs, yticklabels=modes)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Modes')
plt.title('Beta1 Heatmap')
plt.savefig("media/dispersion/disp.png")

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

nlin_single = np.zeros((1, len(freqs)))
for j in range(len(freqs)):
    nlin_single[0, j] = np.sum(L / (np.abs(beta1[0, :] - beta1[0, j])[(beta1[0, :] - beta1[0, j]) != 0] * T))
print("Unlucky channel has noise: ", np.min(nlin))
print("Lucky channel has noise: ", np.max(nlin))

# plt.clf(
# sns.heatmap(collisions, cmap="magma", square=False, xticklabels=freqs, yticklabels=modes)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Modes')
# plt.title('Total number of collision due to system')
# plt.savefig("media/dispersion/disp.png")
# plt.show()

plt.clf()
for i in range(4):
    plt.plot(freqs * 1e-12, collisions[i, :], label=f'Mode {modes[i]}')
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$m_{\mathrm{max}}$')
plt.legend()
plt.grid(True)
plt.savefig(f"media/dispersion/collisions.png")
# plt.show()

plt.clf()
plt.plot(freqs * 1e-12, collisions_single[0, :], label=f'Mode {modes[i]}')
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$m_{\mathrm{max}}$')
plt.legend()
plt.grid(True)
plt.savefig(f"media/dispersion/collisions_single.png")
# plt.show()

plt.clf()
plt.semilogy(freqs * 1e-12, nlin_single[0, :], label=f'Mode {modes[i]}', marker='x')
plt.xlabel('Frequency (THz)')
plt.ylabel('NLIN coeff')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"media/dispersion/nlin_single.png")
# plt.show()

plt.clf()
for i in range(4):
    plt.semilogy(freqs * 1e-12, nlin[i, :], label=f'Mode {modes[i]}', marker='x')
plt.xlabel('Frequency (THz)')
plt.ylabel('NLIN coeff')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"media/dispersion/nlin.png")
# plt.show()

plt.clf()
plt.figure(figsize=(4.6, 4))
for i in range(4):
    plt.plot(freqs * 1e-12, beta1[i, :] * 1e9, label=f'Mode {modes[i]}')
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\beta_1$ (ps/km)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"media/dispersion/beta1.png")

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
plt.savefig(f"media/dispersion/DGD_histogram.png")

mask = (beta1_differences < 0.002 * 1e-12)
hist, edges = np.histogram(beta1_differences[mask]*1e12, bins=200)
hist = hist / 2.0
plt.clf()
plt.figure(figsize=(4, 3.5))
plt.bar(edges[:-1], hist, width=np.diff(edges), zorder=3)
plt.xlabel('DGD (ps/m)')
plt.ylabel('channel pair count')
plt.grid(axis='y', zorder=0)
plt.tight_layout()
plt.savefig(f"media/dispersion/DGD_histogram_zoom.png")

plt.clf()
plt.figure(figsize=(4.6, 4))
for i in range(4):
    plt.plot(freqs * 1e-12, (beta1[i, :] - beta1[1, :])
             * 1e12, label=f'Mode {modes[i]}')
plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\Delta\beta_1$ (ps/m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"media/dispersion/DMGD_LP01.png")
