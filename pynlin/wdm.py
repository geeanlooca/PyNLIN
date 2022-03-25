from operator import mod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import nu2lambda

import pynlin.utils


class WDM:
    """This class is used to create a WDM object."""

    def __init__(
        self, spacing: float = 100, num_channels: int = 50, center_frequency=193.1
    ):
        """Initialize the WDM object.

        Params
        ------
        spacing: float
            The spacing between the center frequencies of the WDM channels (GHz).
        """
        self.spacing = spacing
        self.num_channels = num_channels
        self.central_frequency = center_frequency

    def frequency_grid(self) -> np.ndarray:
        """Generate the frequency grid.

        Returns
        -------
        freqs: ndarray
            The frequencies of the WDM grid in Hz.
        """

        num_channels = self.num_channels

        if num_channels % 2:
            freqs = np.arange(-(num_channels - 1) / 2, (num_channels - 1) / 2 + 1)
        else:
            freqs = np.arange(-num_channels / 2, num_channels / 2)

        return freqs * self.spacing * 1e9 + self.central_frequency * 1e12

    def wavelength_grid(self) -> np.ndarray:
        """Generate the wavelength grid.

        Returns
        -------
        wavelengths: ndarray
            The wavelengths of the WDM grid in meters.
        """
        return nu2lambda(self.frequency_grid())

    def plot(self, ax, xaxis="wavelength", **kwargs):
        pynlin.utils.OpticalBands.plot(ax, xaxis=xaxis)
        if xaxis == "wavelength":
            w = self.wavelength_grid() * 1e9
            ax.set_xlabel("Wavelength [nm]")
        else:
            w = self.wavelength_grid()
            w = nu2lambda(w) * 1e-12
            ax.set_xlabel("Frequency [THz]")

        ax.stem(w, np.ones_like(w), **kwargs)
