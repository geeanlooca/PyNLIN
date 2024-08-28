import functools
import math
from typing import Tuple, List
from numba import jit

import h5py
import numpy as np
import scipy.integrate
import tqdm
from scipy.constants import nu2lambda
from tqdm.contrib.concurrent import process_map
from itertools import product

from pynlin.fiber import Fiber, SMFiber, MMFiber
from pynlin.pulses import Pulse, RaisedCosinePulse, GaussianPulse, NyquistPulse
from pynlin.wdm import WDM
from pynlin.collisions import get_interfering_frequencies, get_m_values, get_interfering_channels, get_frequency_spacing, get_collision_location, get_z_walkoff, get_dgd, get_gvd


# TODO needs to be generalized
def apply_chromatic_dispersion(
        b_chan: Tuple[int, int], pulse: Pulse, fiber: Fiber, wdm: WDM, z: float, delay: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the propagated pulse shape.
    Optionally apply a delay in time.
    """

    g, t = pulse.data()
    dt = t[1] - t[0]
    nsamples = len(g)
    beta2 = get_gvd(b_chan, fiber, wdm)
    freq = np.fft.fftfreq(nsamples, d=dt)
    omega = 2 * np.pi * freq
    omega = np.fft.fftshift(omega)
    gf = np.fft.fftshift(np.fft.fft(g))

    propagator = -1j * beta2 / 2 * omega**2 * z
    delay = np.exp(-1j * delay * omega)

    gf_propagated = gf * np.exp(propagator) * delay
    g_propagated = np.fft.ifft(np.fft.fftshift(gf_propagated))

    return g_propagated


def get_interfering_channels(a_chan: Tuple, wdm: WDM, fiber: Fiber):
    b_chans = list(product(range(fiber.n_modes), range(wdm.num_channels)))
    b_chans.remove(a_chan)
    return b_chans


def iterate_time_integrals(
    wdm: WDM,
    fiber: Fiber,
    a_chan: Tuple,  # WDM index and mode
    pulse: Pulse,
    filename: str,
    overwrite=False,
    **compute_collisions_kwargs,
) -> None:
    """Compute the inner time integral of the expression for the XPM
    coefficients Xhkm for each combination of frequencies in the supplied WDM
    grid."""
    if isinstance(fiber, SMFiber):
      assert (a_chan[0] == 0)

    append_write = "a"
    found = False
    try:
        with h5py.File(filename, 'r') as file:
            for gg in file["time_integrals"]:
                print(gg)

            if f"a_chan_{a_chan}" in file["time_integrals"]:
                print("A-chan group already present on file. Nothing to do.")
                found = True
    except FileNotFoundError:
        print(f"File {filename} not found. Creating a new file.")
        append_write = "w"

    if overwrite and found:
        print(
            "\033[91m warn: \033[0m overwriting by deleting and rewriting all the results file!")
        append_write = "w"
    elif found and not overwrite:
        return -1

    print("No groups found for this A channel, calculating...")
    file = h5py.File(filename, append_write)

    frequency_grid = wdm.frequency_grid()
    a_freq = frequency_grid[a_chan[1]]
    group_name = f"time_integrals/a_chan_{a_chan}/"
    group = file.create_group(group_name)
    group.attrs["mode"] = a_chan[0]
    group.attrs["frequency"] = a_freq

    b_channels = get_interfering_channels(
        a_chan, wdm, fiber,
    )

    # iterate over all channels of the WDM grid
    for b_num, b_chan in enumerate(b_channels):
        # set up the progress bar to iterate over
        # all interfering channels of the current channel of interest
        pbar = tqdm.tqdm(b_channels)
        pbar.set_description(
            f"A-chan: {a_chan}, B-chan = {b_chan}"
        )

        z, I, M = compute_all_collisions_time_integrals(
            a_chan,
            b_chan,
            fiber,
            wdm,
            pulse,
            **compute_collisions_kwargs,
        )

        # in each COI group, create a group for each interfering channel
        # and store the z-array (positions inside the fiber)
        # and the time integrals for each collision.
        b_freq = frequency_grid[b_chan[1]]
        interferer_group_name = group_name + \
            f"b_chan_{b_chan}/"
        interferer_group = file.create_group(interferer_group_name)
        interferer_group.attrs["frequency"] = b_freq
        interferer_group.attrs["mode"] = b_chan[0]

        file.create_dataset(interferer_group_name + "z", data=z)
        file.create_dataset(interferer_group_name + "m", data=M)
        # integrals_group_name = interferer_group_name + "/integrals/"
        # for x, integral in enumerate(I_list):
        file.create_dataset(
            interferer_group_name + f"integrals",
            data=I,
            compression="gzip",
            compression_opts=9,
        )


def compute_all_collisions_time_integrals(
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    fiber: Fiber,
    wdm: WDM,
    pulse: Pulse,
    points_per_collision: int = 10,
    use_multiprocessing: bool = False,
    partial_collisions_margin: int = 10,
    speedup_pulse_propagation=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the integrals for all the collisions for the specified pair of
    channels.

    Returns
    -------
    z: np.ndarray
        The sampling points along the fiber
    time_integrals: np.ndarray
        Time integral as a function of the fiber position, one for each collision
    m: np.ndarray
        Array of collision indeces
    """
    T = 1 / pulse.baud_rate
    f_grid = wdm.frequency_grid()
    m_list = get_m_values(
        fiber,
        wdm,
        a_chan,
        b_chan,
        T,
        partial_collisions_margin,
    )[::-1]
    z_axis_list = []
    z_walkoff = get_z_walkoff(fiber, wdm, a_chan, b_chan, pulse)
    print("==============")
    print(
        f"a: {a_chan}, b: {b_chan}, a_freq:{f_grid[a_chan[1]]:.5e}, b_freq:{f_grid[b_chan[1]]:.5e}")
    print(f"dgd: {get_dgd(a_chan, b_chan, fiber, wdm)}, z_w: {z_walkoff}, lenght/z_w:{fiber.length/z_walkoff:.5e}")
    print("==============")

    n_rough_grid = 200
    spacing = get_frequency_spacing(a_chan, b_chan, wdm)

    def i_func(m, z): return m_th_time_integral_Gaussian(
        pulse,
        fiber,
        wdm,
        a_chan,
        b_chan,
        spacing,
        m,
        z
    )
    print(m_list)

    # Estimate the significant range of each collision:
    #   fill the z_axis_grid with good estimates of
    #   where the pulses start and end the collision
    #   print(get_frequency_spacing(a_chan, b_chan, wdm))
    n_z_points = 200
    margin = 10
    if isinstance(pulse, NyquistPulse):
      print("\033[91m warn: \033[0m The pulse is Nyquist (long-tailed): overriding the number of points!")
      n_z_points = 1000
      margin = 100
    for m in m_list:
        # print(f"m={m}")
        z_m = get_collision_location(m, fiber, wdm, a_chan, b_chan, pulse)
        z_min = z_m - z_walkoff / 2 * margin
        z_max = z_m + z_walkoff / 2 * margin
        i_sample_z_m = i_func(m, z_m)
        dz = fiber.length / n_rough_grid
        threshold = i_sample_z_m / 100
        z = 0
        i_sample = i_func(m, z)
        i_sample = i_func(m, z_min)
        while i_sample > threshold and z_min > 0:
            z_min -= dz
            i_sample = i_func(m, z_min)

        i_sample = i_func(m, z_max)
        while i_sample > threshold and z_max < fiber.length:
            z_max += dz
            i_sample = i_func(m, z_max)
        z_min = max(z_min, 0)
        z_max = min(z_max, fiber.length)
        if z_min > z_max:
            z_axis_list.append(None)
        else:
            z_axis_list.append(np.linspace(z_min, z_max, n_z_points))

    if pulse.num_symbols != n_z_points:
      print("\033[91m warn: \033[0m overriding the pulse number of samples!")
      pulse.num_symbols = n_z_points

    if use_multiprocessing:
        # build a partial function otherwise multiprocessing
        # complains about not being able to pickle stuff
        partial_function = functools.partial(
            m_th_time_integral, pulse, fiber, wdm, a_chan, b_chan,
        )
        # def partial_function(m, z): return m_th_time_integral(pulse, fiber, wdm, a_chan, b_chan, m, z)
        integrals_list = process_map(
            partial_function, m_list, z_axis_list, leave=False, chunksize=1
        )
    # convert the list of arrays in a 2d array, since the shape is the same
    z_axis_list_2d = np.stack(z_axis_list)
    integrals_list_2d = np.stack(integrals_list)
    return z_axis_list_2d, integrals_list_2d, m_list


# Multiprocessing wrapper
def m_th_time_integral(
    pulse: Pulse,
    fiber: Fiber,
    wdm: WDM,
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    m: int,
    z: List[float]
):
    freq_spacing = get_frequency_spacing(a_chan, b_chan, wdm)
    if isinstance(pulse, NyquistPulse):
        return m_th_time_integral_general(pulse, fiber, wdm, a_chan, b_chan, freq_spacing, m, z)
        raise NotImplementedError("no Nyquist Pulse yet")
    elif isinstance(pulse, GaussianPulse):
        # return m_th_time_integral_general(pulse, fiber, wdm, a_chan, b_chan, freq_spacing, m, z)
        return m_th_time_integral_Gaussian(pulse, fiber, wdm, a_chan, b_chan, freq_spacing, m, z)
    else:
        return m_th_time_integral_general(pulse, fiber, wdm, a_chan, b_chan, freq_spacing, m, z)


# @jit
def m_th_time_integral_Gaussian(
    pulse: Pulse,
    fiber: Fiber,
    wdm: WDM,
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    freq_spacing: float,
    m: int,
    z: List[float]
) -> float:
    # Apply the fully analytical formula
    if isinstance(fiber, SMFiber):
        l_d = 1 / (np.abs(fiber.beta2) * (pulse.baud_rate)**2)
        dgd = fiber.beta2 * 2 * np.pi * (freq_spacing)
        factor1 = pulse.baud_rate / (np.sqrt(2 * np.pi))
        factor2 = 1 / np.sqrt(1 + (z / l_d)**2)
        # print(f"=== z: {z:.3e}, exponent: {m + pulse.baud_rate * dgd * z:.3e}")
        exponent = -((m + pulse.baud_rate * dgd * z) ** 2) / (2 * (1 + (z / l_d)**2))
        # print(f"m/pulse.baud_rate : {m:.5e}, dgd z: {pulse.baud_rate * dgd * z:.5e}, exponent : {exponent:.5e}")
        return factor1 * factor2 * np.exp(exponent)
    if isinstance(fiber, MMFiber):
        l_da = 1 / \
            (fiber.group_delay.evaluate_beta2(
                a_chan[0], wdm.frequency_grid()[a_chan[1]])(pulse.baud_rate)**2)
        l_db = 1 / \
            (fiber.group_delay.evaluate_beta2(
                b_chan[0], wdm.frequency_grid()[b_chan[1]])(pulse.baud_rate)**2)
        dgd = fiber.group_delay.evaluate_beta1(b_chan[0], wdm.frequency_grid(
        )[b_chan[1]]) - fiber.group_delay.evaluate_beta1(a_chan[0], wdm.frequency_grid()[a_chan[1]])
        avg_l_d = (l_da * l_db) / (l_da + l_db) / 2
        factor1 = pulse.baud_rate / (np.sqrt(2 * np.pi))
        factor2 = 1 / np.sqrt(1 + (z / avg_l_d)**2)
        exponent = -((m / pulse.baud_rate + dgd * z)**2) / (2 * (1 + (z / avg_l_d)**2))
        return factor1 * factor2 * np.exp(exponent)


def m_th_time_integral_Nyquist(
    pulse: Pulse,
    fiber: Fiber,
    wdm: WDM,
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    freq_spacing: float,
    m: int,
    z: float
) -> float:
    # Integrate in spectral domain?
    if isinstance(fiber, SMFiber):
        l_d = 1 / (np.abs(fiber.beta2) * (pulse.baud_rate)**2)
        dgd = fiber.beta2 * 2 * np.pi * (freq_spacing)
        factor1 = pulse.baud_rate / (np.sqrt(2 * np.pi))
        factor2 = 1 / np.sqrt(1 + (z / l_d)**2)
        exponent = -((m + pulse.baud_rate * dgd * z) ** 2) / (2 * (1 + (z / l_d)**2))
        return factor1 * factor2 * np.exp(exponent)
    if isinstance(fiber, MMFiber):
        l_da = 1 / \
            (fiber.group_delay.evaluate_beta2(
                a_chan[0], wdm.frequency_grid()[a_chan[1]])(pulse.baud_rate)**2)
        l_db = 1 / \
            (fiber.group_delay.evaluate_beta2(
                b_chan[0], wdm.frequency_grid()[b_chan[1]])(pulse.baud_rate)**2)
        dgd = fiber.group_delay.evaluate_beta1(b_chan[0], wdm.frequency_grid(
        )[b_chan[1]]) - fiber.group_delay.evaluate_beta1(a_chan[0], wdm.frequency_grid()[a_chan[1]])
        avg_l_d = (l_da * l_db) / (l_da + l_db) / 2
        factor1 = pulse.baud_rate / (np.sqrt(2 * np.pi))
        factor2 = 1 / np.sqrt(1 + (z / avg_l_d)**2)
        exponent = -((m / pulse.baud_rate + dgd * z)**2) / (2 * (1 + (z / avg_l_d)**2))
        return factor1 * factor2 * np.exp(exponent)


def m_th_time_integral_general(
    pulse: Pulse,
    fiber: Fiber,
    wdm: WDM,
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    freq_spacing: float,
    m: int,
    z_axis: List[float]
) -> float:
    i_list = []
    dt = pulse.T0/pulse.samples_per_symbol
    for z in z_axis:
      delay = m / pulse.baud_rate + get_dgd(a_chan, b_chan, fiber, wdm) * z
      g1 = apply_chromatic_dispersion(b_chan, pulse, fiber, wdm, z, 0.0)
      g2 = np.conj(g1)
      g3 = apply_chromatic_dispersion(b_chan, pulse, fiber, wdm, z, delay)
      g4 = np.conj(g3)
      i_list.append(scipy.integrate.trapezoid(g1 * g2 * g3 * g4, dx=dt))
    return i_list


# def X0mm_time_integral_multiprocessing_wrapper(
#     pulse: Pulse, fiber: Fiber, z: np.ndarray, channel_spacing: float, m: int
# ):
#     """A wrapper for the `X0mm_time_integral` function to easily
#     `functool.partial` and enable multiprocessing."""
#     return X0mm_time_integral(pulse, fiber, z, channel_spacing, m)


# def X0mm_time_integral_precomputed_multiprocessing_wrapper(
#     pulse_matrix: np.ndarray, fiber: Fiber, z: np.ndarray, t: np.ndarray, channel_spacing: float, m: int, T: float
# ):
#     """A wrapper for the `X0mm_time_integral` function to easily
#     `functool.partial` and enable multiprocessing."""
#     return X0mm_time_integral_precomputed(pulse_matrix, fiber, z, t, channel_spacing, m, T)
# Numerical or semianalytical methods for time integrals


def m_th_time_integral_Nyquist(
    pulse: Pulse,
    fiber: Fiber,
    z: np.ndarray,
    channel_spacing: float,
    m: int,
) -> np.ndarray:
    # apply the formula from Nakazawa
    pass


# def X0mm_time_integral_precomputed(
#     pulse_matrix: np.ndarray,
#     fiber: Fiber,
#     z: np.ndarray,
#     t: np.ndarray,
#     channel_spacing: float,
#     m: int,
#     T: float,
# ) -> np.ndarray:
#     """Compute the inner time integral of the expression for the XPM
#     coefficients Xhkm for the specified channel spacing.
#     Use a pulse matrix to exploit the precomputation of the propagation"""
#     npoints_z = len(z)
#     dt = t[1] - t[0]
#     O = 2 * np.pi * channel_spacing
#     beta2 = fiber.beta2
#     time_integrals = np.zeros((npoints_z,), dtype=complex)
#     for i, L in enumerate(z):
#         g1 = pulse_matrix[:, i]
#         delay = (- m * T - beta2 * O * L)
#         shift = int(np.round((delay / dt)))
#         g3 = np.roll(pulse_matrix[:, i], shift)
#         integrand = np.conj(g1) * g1 * np.conj(g3) * g3
#         time_integrals[i] = scipy.integrate.trapezoid(integrand, t)
#     return time_integrals


####################################
# Other methods, inefficient
####################################
# def X0mm_time_integral(
#     pulse: Pulse,
#     fiber: Fiber,
#     z: np.ndarray,
#     channel_spacing: float,
#     m: int,
# ) -> np.ndarray:
#     """Compute the inner time integral of the expression for the XPM
#     coefficients Xhkm for the specified channel spacing.
#     Propagate all the pulses one by one"""
#     npoints_z = len(z)
#     g, t = pulse.data()
#     dt = t[1] - t[0]
#     nsamples = len(g)
#     freq = np.fft.fftfreq(nsamples, d=dt)
#     omega = 2 * np.pi * freq
#     omega = np.fft.fftshift(omega)
#     O = 2 * np.pi * channel_spacing
#     T = pulse.T0
#     beta2 = fiber.beta2
#     time_integrals = np.zeros((npoints_z,), dtype=complex)
#     gf = np.fft.fftshift(np.fft.fft(g))

#     for i, L in enumerate(z):
#         # for each point in space, calculate the corresponding time integral
#         propagator = -1j * beta2 / 2 * omega**2 * L
#         delay = np.exp(-1j * (m * T + beta2 * O * L) * omega)

#         gf_propagated_1 = gf * np.exp(propagator)
#         gf_propagated_3 = gf_propagated_1 * delay

#         g1 = np.fft.ifft(np.fft.fftshift(gf_propagated_1))
#         g3 = np.fft.ifft(np.fft.fftshift(gf_propagated_3))

#         integrand = np.conj(g1) * g1 * np.conj(g3) * g3
#         time_integrals[i] = scipy.integrate.trapezoid(integrand, t)

#     return time_integrals


# def Xhkm_time_integral_multiprocessing_wrapper(
#     pulse: Pulse, fiber: Fiber, z: np.ndarray, channel_spacing: float, m: int
# ):
#     """A wrapper for the `Xhkm_time_integral` function to easily
#     `functool.partial` and enable multiprocessing."""
#     return Xhkm_time_integral(pulse, fiber, z, channel_spacing, 0, m, m)


# # Currently unused methods for nondegenerate FWM noise
# def Xhkm_time_integral(
#     pulse: Pulse,
#     fiber: Fiber,
#     z: np.ndarray,
#     channel_spacing: float,
#     h: int,
#     k: int,
#     m: int,
# ) -> np.ndarray:
#     """Compute the inner time integral of the expression for the XPM
#     coefficients Xhkm for the specified channel spacing."""
#     npoints_z = len(z)
#     g, t = pulse.data()
#     O = 2 * np.pi * channel_spacing
#     T = pulse.T0
#     beta2 = fiber.beta2
#     time_integrals = np.zeros((npoints_z,), dtype=complex)

#     for i, L in enumerate(z):
#         # for each point in space, calculate the corresponding time integral
#         g1 = apply_chromatic_dispersion(pulse, fiber, L, delay=0)
#         g2 = apply_chromatic_dispersion(pulse, fiber, L, delay=h * T)
#         g3 = apply_chromatic_dispersion(pulse, fiber, L, delay=k * T + beta2 * O * L)
#         g4 = apply_chromatic_dispersion(pulse, fiber, L, delay=m * T + beta2 * O * L)

#         integrand = np.conj(g1) * g2 * np.conj(g3) * g4
#         time_integrals[i] = scipy.integrate.trapezoid(integrand, t)

#     return time_integrals


def X0mm_space_integral(
        z: np.ndarray, time_integrals, amplification_function: np.ndarray = None, axis=-1) -> np.ndarray:
    """Compute the X0mm XPM coefficients specifying the inner time integral as
    input.

    Useful to compare different amplification schemes without re-
    computing the time integral.
    """
    if type(amplification_function) is not np.ndarray:
        # if the amplification function is not supplied, assume perfect distributed
        # amplification
        amplification_function = np.ones_like(z)
    else:
        print(
            '\033[2;31;43m WARN \033[0;0m need to implement scaling of f(z) w.r.t. the given z axis.')
    X = scipy.integrate.trapezoid(time_integrals * amplification_function, z, axis=axis)
    return X


# def Xhkm(
#     pulse: Pulse,
#     fiber: Fiber,
#     amplification_function: np.ndarray,
#     z: np.ndarray,
#     channels_spacing: float,
#     h: int,
#     k: int,
#     m: int,
# ) -> np.ndarray:
#     """Compute the Xhkm XPM coefficients."""
#     time_integrals = Xhkm_time_integral(pulse, fiber, z, channels_spacing, h, k, m)

#     # integrate in space
#     X = scipy.integrate.trapezoid(time_integrals * amplification_function, z)
#     return X


# # we can design a single function that takes in input Delta beta1, beta2,
# # check if the pulse is special, and call a specialized function or a standard
# # one (with propagation).

# # wdm + fiber ->
# # (coi[freq, mode], interf[freq, mode]) channel couples ->
# # {X0mm},  collision to be computed ITER ->
# # two integrals to compute ->
# # integrand evaluation: manual propagation or analytics
