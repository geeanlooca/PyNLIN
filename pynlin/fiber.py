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
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta2=20 * 1e-24 / 1e3,
        modes=1,
        overlap_integrals=None,
        overlap_integrals_avg=None,
        mode_names=None,
    ):
        """
        
        Params
        
        ======
        
        overlap_integrals     : 6 quadratic fit parameters for each mode family pair
        
        overlap_integrals_avg : 1 oi for each mode family pair
        
        Strategy: if overlap_integrals is none, go to default case to overlap_integrals_avg
        both of them can be used with polyval (of course using the average is inefficient)
        
        Attributes
        
        =======
         
        self.overlap_integrals : (6, modes, modes) used only in the Numpy solver: can also contain also the average if needed!
        
        self.overlap_integrals_avg : (modes, modes), symmetric, used in the Torch solver. Torch solver do not support wavelength-dependent oi. 
        
        """
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.beta2 = beta2
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
        self.modes = modes

        # Structure of the overlap integrals
        # overlap_integrals[i, j] = [a1, b1, a2, b2, x, c]
        # i, j mode indexes,
        # all the quadratic fit parameters are used in oi_law
        
        if overlap_integrals is None:
          # default super-approximated case: all the modes overlap as the fundamental one.
          # No cross-overlap
          if overlap_integrals_avg is None: 
            # self.overlap_integrals_avg = 1/effective_area * np.identity(modes)   
            self.overlap_integrals_avg = np.array([[1. , 2.], [3. , 4.]])       
            self.overlap_integrals = self.overlap_integrals_avg[None, :, :].repeat(6, axis=0)
            
            for i in range(5):
              self.overlap_integrals[i, :, :] *= 0.0
          else:
            self.overlap_integrals_avg = overlap_integrals_avg
        
          self.overlap_integrals = self.overlap_integrals_avg[None, :, :].repeat(6, axis=0)
          for i in range(5):
            self.overlap_integrals[i, :, :] *= 0.0
        else:
          self.overlap_integrals = overlap_integrals
          self.overlap_integrals = self.overlap_integrals[:, :modes, :modes]
          self.overlap_integrals_avg = self.overlap_integrals[0, :, :]
          
        # adjust for mismatches of OI matrix and selected mode number 
        # print(np.shape(self.overlap_integrals))
        # print(np.shape(self.overlap_integrals_avg))  
        self.overlap_integrals_avg = self.overlap_integrals_avg[:modes, :modes]
        self.overlap_integrals = self.overlap_integrals[:, :modes, :modes]
        
        print("==========================")
        print(self.overlap_integrals[-1, :, :])
        
        self.mode_names = mode_names

        super().__init__()

    """
    i, j are mode indexes
    wl1, wl2 are the respective wavelengths
    """

    def evaluate_oi(self, i, j, wavelength_i, wavelength_j):
        # original data were in um
        return oi_law(wavelength_i, wavelength_j, self.overlap_integrals[:, i, j])
    
    def get_oi_matrix(self, modes, wavelengths):
      M = len(modes)
      W = len(wavelengths)
      mat = np.zeros((M*W, M*W))
      for n in range(M):
        for m in range(M):
          for wn in range(W):
            for wm in range(W):
              mat[n+(wn*M), m+(wm*M)] = self.evaluate_oi(n, m, wavelengths[wn], wavelengths[wm])
      return mat
    
    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
