from numpy import polyval
from pynlin.utils import oi_law
import numpy as np


class Fiber:
    """Object collecting parameters of an optical fiber."""

    def __init__(
        self,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta2=20 * 1e-24 / 1e3,
        gamma=1.3 * 1e-3,
    ):
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.beta2 = beta2
        self.gamma = gamma

        if losses:
            try:
                self.losses = list(losses)
            except:
                self.losses = [losses]
        else:
            # coefficients of the [0, 1, 2]-th order coefficients of a quadratic fit in
            # powers of wavelength (in m). Result in units of dB/m
            self.losses = np.array([2.26786883e-06 * 1e18, -
                           7.12461042e-03 * 1e9, 5.78789219e00]) * 1e-3

        self.raman_efficiency = self.raman_coefficient / self.effective_area

        super().__init__()

    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class MMFiber:
    def __init__(
        self,
        losses=0.2 * 1e-3,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta2=20 * 1e-24 / 1e3,
        modes=1,
        overlap_integrals=None,
        overlap_integrals_avg=None,
        mode_names=None,
    ):
        print()
        """
        overlap_integrals     : 6 quadratic fit parameters for each mode family pair
        overlap_integrals_avg : 1 oi for each mode family pair
        """
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.beta2 = beta2
        try:
            self.losses = list(losses)
        except:
            self.losses = [losses]

        self.raman_efficiency = self.raman_coefficient / self.effective_area
        self.modes = modes

        # Structure of the overlap integrals
        # overlap_integrals[i, j] = [a1, b1, a2, b2, x, c]
        # i, j mode indexes,
        # all the quadratic fit parameters are used in oi_law

        self.overlap_integrals = overlap_integrals[:, :modes, :modes]
        try:
            self.overlap_integrals_avg = overlap_integrals_avg[:modes, :modes]
        except:
            self.overlap_integrals_avg = overlap_integrals_avg
        self.mode_names = mode_names

        super().__init__()

    """
    i, j are mode indexes
    wl1, wl2 are the respective wavelengths
    """

    def evaluate_oi(self, i, j, wavelengths):
        # print("____________")
        # print(np.shape(self.overlap_integrals[:, i, j]))
        # print(np.shape(wavelengths[1][None, :, :, :, :]))
        # print(np.shape(wavelengths[0][None, :, :, :, :]))
        # original data were in um
        return oi_law(wavelengths[0][None, :, :, :, :], wavelengths[1][None, :, :, :, :], self.overlap_integrals[:, i, j])

    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
