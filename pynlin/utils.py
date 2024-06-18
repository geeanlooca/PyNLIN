from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from scipy.constants import speed_of_light as c0
from scipy.constants import nu2lambda, lambda2nu
import torch

def wavelength_to_frequency(lambdas):
    """Convert wavelength to frequency."""
    return scipy.constants.lambda2nu(lambdas)


def frequency_to_wavelength(freqs):
    """Convert frequency to wavelength."""
    return scipy.constants.nu2lambda(freqs)


def alpha_to_linear(alpha):
    """Convert attenuation constant from dB/m to neper/m."""
    return alpha * np.log(10) / 10


def alpha2linear(alpha):
    return alpha_to_linear(alpha)


def wavelength2frequency(lambdas):
    return wavelength_to_frequency(lambdas)


def frequency2wavelength(freqs):
    return frequency_to_wavelength(freqs)


def watt_to_dBm(power):
    return 10 * np.log10(power) + 30


def dBm_to_watt(power):
    return 10 ** ((power - 30) / 10)


def watt2dBm(power):
    return watt_to_dBm(power)


def dBm2watt(power):
    return dBm_to_watt(power)


def beta2_to_dispersion(beta2, wavelength):
    """Convert GVD (beta2) parameter to dispersion coefficient.
    Values are assumed in SI units.
    """
    return -2 * np.pi * c0 / (wavelength**2) * beta2


def dispersion_to_beta2(D, wavelength):
    """Convert the dispersion coefficient to GVD (beta2).
    Values are assumed in SI units.
    """
    return D * wavelength**2 / (-2 * np.pi * c0)


def oi_law(l1, l2, params):
  a1, b1, a2, b2, x, c = params
  return (a1 * l1**2 + b1 * l1 + a2 * l2**2 + b2 * l2 + x * l1 * l2 + c)

def oi_polynomial_expansion(wl, values):
  A1, B1, A2, B2, X, C = values
  M = A1.shape[0]
  N = wl.shape[1]
  OIa1 = torch.kron((wl**2).reshape(N, 1), A1).repeat(1, N)
  OIa2 = torch.kron(wl**2, A2).repeat(N, 1)
  OIb1 = torch.kron(wl.reshape(N, 1), B1).repeat(1, N)
  OIb2 = torch.kron(wl, B2).repeat(N, 1)
  OIx = torch.kron(torch.kron(wl.reshape(N, 1), X), wl)
  OIc = C.repeat((N, N))
  return (OIa1 + OIa2 + OIb1 + OIb2 + OIx + OIc).float()


def oi_law_fit(L, a1, b1, a2, b2, x, c):
  l1, l2 = L
  return (a1 * np.square(l1) + b1 * l1 + a2 * np.square(l2) + b2 * l2 + x * l1 * l2 + c).ravel()


class OpticalBands(Enum):
    """Class enumerating the optical transmission bandwidths.

    Data taken from: https://www.thefoa.org/tech/ref/basic/SMbands.html
    """

    O = (1260, 1360)
    E = (1360, 1460)
    S = (1460, 1530)
    C = (1530, 1565)
    L = (1565, 1625)
    U = (1625, 1675)

    def __add__(self, other):
        joint = self.value + other.value
        return (min(joint), max(joint))

    @staticmethod
    def plot(ax, xaxis="wavelength", **kwargs):
        for x, band in enumerate(OpticalBands):
            m, M = band.value

            if xaxis == "frequency":
                m = lambda2nu(m * 1e-9) * 1e-12
                M = lambda2nu(M * 1e-9) * 1e-12

            mid_point = (M + m) / 2
            name = band.name
            ax.axvspan(m, M, alpha=0.1, color=f"C{x}")
            ax.text(mid_point, 0.1, name)

        if xaxis == "frequency":
            ax.set_xlabel("Frequency [THz]")
        else:
            ax.set_xlabel("Wavelength [nm]")


"""
O-band 	1260 – 1360 nm 	Original band, PON upstream
E-band 	1360 – 1460 nm 	Water peak band
S-band 	1460 – 1530 nm 	PON downstream
C-band 	1530 – 1565 nm 	Lowest attenuation, original DWDM band, compatible with fiber amplifiers, CATV
L-band 	1565 – 1625 nm 	Low attenuation, expanded DWDM band
U-band 	1625 – 1675 nm 	Ultra-long wavelength
"""
