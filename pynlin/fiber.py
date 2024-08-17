from dataclasses import dataclass
from numpy import polyval
from pynlin.utils import oi_polynomial_expansion, oi_law, beta1_polynomial_expansion
import numpy as np
import torch


class Fiber:
    """Object collecting parameters of a Single Mode optical fiber."""
    def __init__(
        self,
        fiber_type,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        gamma=1.3 * 1e-3,
        length = 100e3
    ):
        self.fiber_type = fiber_type 
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.gamma = gamma
        self.length = length
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
        
    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
  
  
class SMFiber(Fiber):
    """Object collecting parameters of a Single Mode optical fiber."""

    def __init__(
        self,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta2=20 * 1e-24 / 1e3,
        gamma=1.3 * 1e-3,
        length = 100e3
    ):
        super().__init__("SM", 
                         losses=losses, 
                         raman_coefficient=raman_coefficient, 
                         effective_area=effective_area, 
                         gamma=gamma, 
                         length=length)
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.beta2 = beta2
        self.gamma = gamma

        self.raman_efficiency = self.raman_coefficient / self.effective_area

    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


"""
OI fit storage and evaluation in a way which is compatible with torch
"""
@dataclass
class OICoefficients:
  values: list[torch.Tensor]

  def __init__(self, modes: int, input_values: np.ndarray):
    self.values = [torch.from_numpy(v[:modes, :modes]) for v in input_values]
    # self.num_modes, dim=2
    # ).float()

  def evaluate_oi_tensor(self, wavelengths: torch.Tensor) -> torch.Tensor:
      """Evaluate the overlap integral between the pump and signal modes.+

      Parameters
      ----------
      coefficients : OICoefficients
          The overlap integral coefficients. The `values` attribute should be a list of length 6
            of torch.Tensor elements, each of size (num_modes, num_modes).

      wavelengths : torch.Tensor (batch_dim, num_frequencies (pumps + signals))
        Tensor containing all the wavelengths involved in the system.
      """
      return oi_polynomial_expansion(wavelengths, self.values)
    
    
"""
Dispersion data storage and evaluation for the walkoff and collision evaluation
"""
@dataclass
class GroupDelay:
  values: list[np.array]

  def __init__(self, modes: int, input_values: np.ndarray):
    self.modes = modes
    self.values = input_values
  
  def evaluate_beta1(self, wavelengths: torch.Tensor) -> torch.Tensor:
    return beta1_polynomial_expansion(wavelengths, self.values)
  
  
class MMFiber:
    def __init__(
        self,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta2=20 * 1e-24 / 1e3,
        gamma=1.3 * 1e-3,
        length = 100e3,
        modes=1,
        overlap_integrals=None,
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
        """
        super().__init__("MM", 
                         losses=losses, 
                         raman_coefficient=raman_coefficient, 
                         effective_area=effective_area, 
                         gamma=gamma, 
                         length=length)
        self.beta2 = beta2 # to be modified
        self.raman_efficiency = self.raman_coefficient / self.effective_area
        self.modes = modes

        # Structure of the overlap integrals
        # overlap_integrals[i, j] = [a1, b1, a2, b2, x, c]
        # i, j mode indexes,
        # all the quadratic fit parameters are used in oi_polynomial_expansion
       
        self.overlap_integrals = overlap_integrals
        self.torch_oi = OICoefficients(self.modes, overlap_integrals)
        
        self.mode_names = mode_names

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
      mat[:, :]
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
