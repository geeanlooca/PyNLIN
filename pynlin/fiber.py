from scipy import polyval
from pynlin.utils import oi_law

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
            self.losses = [2.26786883e-06, -7.12461042e-03, 5.78789219e00]

        self.raman_efficiency = self.raman_coefficient / self.effective_area

        super().__init__()

    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/km) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths * 1e9)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
    
class MMFiber:
    def __init__(
        self,
        losses=0.2,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta2=20 * 1e-24 / 1e3,
        modes=1,
        overlap_integrals=None,
        mode_names=None,
    ):
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.beta2 = beta2
        try:
            self.losses = list(losses)
        except:
            self.losses = [losses]

        self.raman_efficiency = self.raman_coefficient / self.effective_area
        self.modes = modes

        ## Structure of the overlap integrals
        # overlap_integrals[i, j] = [a1, b1, a2, b2, x, c] 
        # i, j mode indexes, 
        # all the quadratic fit parameters are used in oi_law

        self.overlap_integrals = overlap_integrals
        self.mode_names = mode_names
        
        super().__init__()
    
    """
    i, j are mode indexes
    wl1, wl2 are the respective wavelengths
    TODO rename
    """
    def overlap_integral(self, i, j, wavelengths):
      print(self.overlap_integrals[i, j])
      return oi_law(wavelengths, *self.overlap_integrals[i, j]) # original data were in um
        
    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/km) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths * 1e9)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()