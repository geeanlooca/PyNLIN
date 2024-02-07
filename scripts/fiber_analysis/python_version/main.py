import logging  
from typing import List, Tuple
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.constants import lambda2nu, nu2lambda
import scipy.interpolate
from scipy.special import jv as J, kv as K
from scipy.optimize import fminbound
import scipy.integrate


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class PropagationConstant:
    v: NDArray
    b: NDArray
    v_range: Tuple[float, float] = field(init=False)
    spline: scipy.interpolate.CubicSpline = field(init=False)

    def __post_init__(self):

        # remove NaNs from b and the corresponding v values
        mask = np.isnan(self.b)
        self.b = self.b[~mask]
        self.v = self.v[~mask]

        self.spline = scipy.interpolate.CubicSpline(self.v, self.b, extrapolate=False)
        self.v_range = (self.v[0], self.v[-1])

    def __call__(self, v: NDArray) -> NDArray:
        return self.spline(v)

@dataclass
class LPMode:
    type: str
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"LP{self.azimuthal_order}{self.radial_order}"


@dataclass
class LPModeFamily:
    azimuthal_order: int
    radial_order: int
    cutoff_frequency: float
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"LP{self.azimuthal_order}{self.radial_order}"

    def propagation_constant(self, wavelength_range: Tuple[float, float]) -> PropagationConstant:
        v_min = normalized_frequency(self, wavelength_range[0])
        v_max = normalized_frequency(self, wavelength_range[1])
        return find_propagation_constant(self, v_min, v_max)


@dataclass
class ModeCollection:
    
    modes: List[LPModeFamily]

@dataclass
class StepIndexFiber:
    core_radius: float
    n_co: float
    n_cl: float

    def __post_init__(self):
        self.NA = np.sqrt(self.n_co**2 - self.n_cl**2)

    def modes_at(self, wavelength: float) -> List[LPModeFamily]:
        v_max = normalized_frequency(self, wavelength)

        logger.debug(f"Finding modes at wavelength {wavelength*1e6:.2f} um. Corresponding normalized frequency: {v_max:.3f}")
        modes = find_all_modes(v_max)

        logger.debug(f"Found {len(modes)} LP mode families")
        return modes


def normalized_frequency(fiber: StepIndexFiber, wavelength: float) -> float:
    return 2 * np.pi * fiber.core_radius / wavelength * fiber.NA

def denormalize_frequency(fiber: StepIndexFiber, v: float) -> float:
    wavelength = 2 * np.pi * fiber.core_radius * fiber.NA / v
    return lambda2nu(wavelength)


def denormalize_propagation_constant(fiber: StepIndexFiber, b: float, v: float) -> float:
    freq = denormalize_frequency(fiber, v)
    wavelength = nu2lambda(freq)
    k_cl = 2 * np.pi * fiber.n_cl / wavelength
    k_co = 2 * np.pi * fiber.n_co / wavelength
    return (b * (k_co**2 - k_cl**2) + k_cl**2) ** 0.5

def cof_lhs(v, n):
    return v * J(n + 1, v)


def cof_rhs(v, n):
    return 2 * n * J(n, v)


def cutoff_frequency_equation(v: float, n: int) -> float:
    """The characteristic equation for the cutoff frequency.

    Args:
        v (float): normalized frequency
        n (int): azimuthal order

    Returns:
        float: The difference between the lhs and rhs of the characteristic equation.
    """
    return cof_lhs(v, n) - cof_rhs(v, n)


def characteristic_equation(b: float, v: float, n: int) -> float:
    """Compute difference between lhs and rhs of the characteristic equation.

    Args:
        b (float): _description_
        v (float): the normalized frequency
        n (int): the azimuthal order

    Returns:
        float: _description_
    """
    v1b = v * (1 - b) ** 0.5
    vb = v * b**0.5
    lhs = v1b * J(n + 1, v1b) / J(n, v1b)
    rhs = vb * K(n + 1, vb) / K(n, vb)
    return lhs - rhs


def find_zeros(fun, v_min: float, v_max: float, debug: bool = False):
    num_intervals = 200
    interval = (v_max - v_min) / num_intervals

    zeros = []

    for i in range(num_intervals):
        v1 = v_min + i * interval
        v2 = v_min + (i + 1) * interval

        if fun(v1) * fun(v2) < 0:
            z, val, err, num_func = fminbound(
                lambda x: np.abs(fun(x)), v1, v2, full_output=True
            )
            zeros.append(z)

    if debug:
        plt.figure()
        plt.plot(np.linspace(v_min, v_max, 1000), fun(np.linspace(v_min, v_max, 1000)))
        plt.plot(zeros, np.zeros_like(zeros), "x")

    

    return sorted(zeros)



def find_all_modes(v_max: float) -> List[LPModeFamily]:
    m = 0
    first_mode = False
    modes = [LPModeFamily(azimuthal_order=0, radial_order=1, cutoff_frequency=0)]

    while True:
        zeros = find_zeros(lambda v: cutoff_frequency_equation(v, m), 0, v_max, debug=False)

        if len(zeros) == 0:
            if m != 0:
                break
            else:
                if first_mode:
                    break
                else:
                    first_mode = True
                    m += 1
                    continue


        for n, v_cutoff in enumerate(zeros):
            if m == 0:
                radial_order = n + 2
            else:
                radial_order = n + 1
            mode = LPModeFamily(
                azimuthal_order=m, radial_order=radial_order, cutoff_frequency=v_cutoff
            )
            modes.append(mode)

        m += 1

    return modes


def find_propagation_constant(
    mode: LPModeFamily, v_min: float, v_max: float
) -> PropagationConstant:
    dv = 0.01
    num_points = int((v_max - v_min) / dv)
    num_points = max(num_points, 256)
    v_ = np.linspace(v_min, v_max, num_points)
    b_ = np.full_like(v_, np.NaN)

    if mode.cutoff_frequency > v_max:
        return PropagationConstant(v_, b_)

    for i, v in enumerate(v_):
        if v <= mode.cutoff_frequency:
            continue

        db = 0.1

        lb = 0 if i == 0 else b_[i - 1]
        lb = 0 if np.isnan(lb) else lb
        ub = lb + db

        b, f_val, *_ = fminbound(
            lambda b: np.abs(characteristic_equation(b, v, mode.azimuthal_order)),
            lb,
            ub,
            full_output=True,
        )
        b_[i] = b


    return PropagationConstant(v_, b_)

def mode_intensity(fiber: StepIndexFiber, mode: LPModeFamily, b: float, v: float, type: str = "even", debug: bool = False):

    accepted_types = ("even", "odd")
    if type not in accepted_types:
        raise ValueError(f"Type must be one of {accepted_types}")
    
    azimuth_fn = np.cos if type == "even" else np.sin

    a = fiber.core_radius
    chi_co = v * (1-b) ** 0.5 / a
    chi_cl = v * b ** 0.5 / a

    x = np.linspace(-1.5 * a, 1.5 *a, 1000)
    y = np.linspace(-1.5 * a, 1.5 *a, 1000)

    xx, yy = np.meshgrid(x, y)
    rr = (xx**2 + yy**2) ** 0.5
    phi = np.arctan2(yy, xx)
    

    # compute the radial part of the mode
    n = mode.azimuthal_order

    index_cor = rr <= a
    index_clad = rr > a

    field = np.zeros_like(rr)
    field[index_cor] = J(n, chi_co * rr[index_cor]) / J(n, chi_co * a)
    field[index_clad] = K(n, chi_cl * rr[index_clad]) / K(n, chi_cl * a)

    field *= azimuth_fn(n * phi)

    core_circle = plt.Circle((0, 0), a, fill=False)

    if debug:
        plt.figure()
        plt.imshow(np.abs(field), extent=[-1.5*a, 1.5*a, -1.5*a, 1.5*a])
        plt.gca().add_patch(core_circle)

    return xx, yy, np.abs(field) ** 2




def double_integral(xx, yy, f) -> float:
    iy = np.trapz(f, axis=0)
    ix = np.trapz(iy, axis=0)
    return ix

def effective_mode_area(xx, yy, intensity) -> float:
    aeff =  double_integral(xx, yy, intensity) ** 2 / double_integral(xx, yy, intensity ** 2)
    return aeff
    
    



if __name__ == "__main__":
    fiber = StepIndexFiber(core_radius=8e-6, n_co=1.4485, n_cl=1.417)

    wavelengths = np.linspace(1.1e-6, 1.6e-6, 40)
    v_max = normalized_frequency(fiber, wavelengths[0])
    v_min = normalized_frequency(fiber, wavelengths[-1])
    print(v_max, v_min)

    modes = fiber.modes_at(v_max)
    print(modes)

    mode = modes[0]
    prop_const = find_propagation_constant(mode, 0, v_max)  

    plt.figure()
    plt.plot(prop_const.v, prop_const.b)
    plt.show()

    Aeff = np.zeros_like(wavelengths)
    b_s = np.zeros_like(wavelengths)

    print(mode)

    for i, wavelength in enumerate(wavelengths):
        v = normalized_frequency(fiber, wavelength)
        print(wavelength, v)
        b = prop_const(v)
        b_s[i] = b
        xx, yy, intensity = mode_intensity(fiber, mode, b, v, debug=True)
        plt.title(f"Mode intensity at wavelength {wavelength*1e9:.2f} nm")

        Aeff[i] = effective_mode_area(xx, yy, intensity)

    plt.figure()
    plt.plot(wavelengths*1e9, Aeff*1e12)


    plt.figure()
    plt.plot(wavelengths*1e9, b_s)




    
    plt.show()
