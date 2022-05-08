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
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.wdm import WDM
import pynlin.constellations

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '26'

average_energy = 1
power_dBm_list = np.linspace(-20, 0, 3)
arity_list = [16, 64, 256]

constellation_variance = []

for ar_idx, M in enumerate(arity_list):
            qam = pynlin.constellations.QAM(M)

            qam_symbols = qam.symbols()
            cardinality = len(qam_symbols)

            # assign specific average optical energy
            qam_symbols = qam_symbols / \
                np.sqrt(np.mean(np.abs(qam_symbols)**2)) * \
                np.sqrt(average_energy)

            constellation_variance.append(
                np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)

fig_arity = plt.figure(figsize=(10, 5))

# normalized to 16-QAM variance
plt.loglog(arity_list, constellation_variance/constellation_variance[0],
           marker='x', markersize=10, color='black')
plt.minorticks_off()
plt.grid()
plt.xlabel("QAM modulation arity")
plt.xticks(ticks=arity_list, labels=arity_list)
plt.ylabel(r"variance scale factor")
plt.yticks(ticks=constellation_variance/constellation_variance[0], labels=[1.0, 1.190, 1.235])

fig_arity.tight_layout()
fig_arity.savefig("arity_noise.pdf")
