import functools
import math
from typing import Tuple, List

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
from pynlin.collisions import get_interfering_frequencies, get_m_values, get_interfering_channels, get_frequency_spacing


def apply_chromatic_dispersion(
    pulse: Pulse, fiber: Fiber, z: float, delay: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the propagated pulse shape.

    Optionally apply a delay in time.
    """

    g, t = pulse.data()

    dt = t[1] - t[0]
    nsamples = len(g)
    beta2 = fiber.beta2

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
    b_chans = list(product(range(wdm.num_channels), range(fiber.n_modes)))
    b_chans.remove(a_chan)
    return b_chans


def get_dgd(a_chan, b_chan, fiber, wdm):
    freq_grid = wdm.frequency_grid()
    if isinstance(fiber, SMFiber):
        assert (a_chan[1] == 0 and b_chan[1] == 0)
        return fiber.beta2 * (freq_grid(b_chan[0]) - freq_grid(a_chan[0]))
    elif isinstance(fiber, MMFiber):
        return fiber.group_delay.evaluate_beta1(b_chan[0], freq_grid[b_chan[1]]) - fiber.group_delay.evaluate_beta1(a_chan[0], freq_grid[a_chan[1]])


def get_gvd(b_chan, fiber, wdm):
    if isinstance(fiber, SMFiber):
        return fiber.beta2
    elif isinstance(fiber, MMFiber):
        return fiber.group_delay.evaluate_beta2(b_chan[0], wdm.frequenc_grid()[b_chan[1]])
    pass


def iterate_time_integrals(
    wdm: WDM,
    fiber: Fiber,
    a_chan: Tuple,  # WDM index and mode
    pulse: Pulse,
    filename: str,
    **compute_collisions_kwargs,
) -> None:
    """Compute the inner time integral of the expression for the XPM
    coefficients Xhkm for each combination of frequencies in the supplied WDM
    grid."""
    assert (isinstance(fiber, SMFiber) and a_chan[1] == 0)

    # open the file to save data to disk
    with h5py.File('results/results.hdf5', 'a') as file:
        if f"time_integrals/channel_{a_chan}/" in file:
            print("A chan group already present on file. Nothing to do.")
            return -1

    print("No groups found for this A channel, calculating...")
    file = h5py.File(filename, "a")

    frequency_grid = wdm.frequency_grid()
    b_channels = get_interfering_channels(
        a_chan, wdm, fiber,
    )

    # iterate over all channels of the WDM grid
    for b_num, b_chan in enumerate(b_channels):
        # set up the progress bar to iterate over all interfering channels
        # of the current channel of interest
        interfering_frequencies_pbar = tqdm.tqdm(b_channels)
        interfering_frequencies_pbar.set_description(
            f"Interfering frequencies of channel {b_chan}"
        )

        z, I, M = compute_all_collisions_time_integrals(
            a_chan,
            b_chan,
            fiber,
            wdm,
            pulse,
            **compute_collisions_kwargs,
        )
        # saving the result
        # each Channel of interest gets its own group with its attributes
        a_freq = frequency_grid[a_chan[1]]
        b_freq = frequency_grid[b_chan[1]]
        group_name = f"time_integrals/channel_{a_chan}/"
        group = file.create_group(group_name)
        group.attrs["mode"] = a_chan[0]
        group.attrs["frequency"] = a_freq

        # in each COI group, create a group for each interfering channel
        # and store the z-array (positions inside the fiber)
        # and the time integrals for each collision.
        interferer_group_name = group_name + \
            f"interfering_channel_{b_chan}/"
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

    M = get_m_values(
        fiber,
        wdm,
        a_chan,
        b_chan,
        T,
        partial_collisions_margin,
    )
    # first, create the Pulse object with the appropriate parameters
    # compute the maximum delay between pulses and use it to set
    # the number of symbols in the rcos signal
    # tau_max = fiber.beta2 * fiber.length * 2 * np.pi * channel_spacing
    # num_symbols = np.abs(4 * math.ceil(tau_max / T))

    npoints_z = math.ceil(len(M) * points_per_collision)
    z = np.linspace(0, fiber.length, npoints_z)

    # channel_spacing_GHz = channel_spacing * 1e-9
    # interfering_frequency_THz = interfering_frequency * 1e-12
    # coi_frequency_THz = frequency_of_interest * 1e-12
    freq_spacing = get_frequency_spacing(a_chan, b_chan, wdm)

    if speedup_pulse_propagation:
        g, t = pulse.data()
        dt = t[1] - t[0]
        nsamples = len(g)
        freq = np.fft.fftfreq(nsamples, d=dt)
        omega = 2 * np.pi * freq
        omega = np.fft.fftshift(omega)
        beta2 = fiber.beta2
        gf = np.fft.fftshift(np.fft.fft(g))
        pulse_matrix = np.zeros((len(t), len(z)), dtype=complex)
        for i, L in enumerate(z):
            propagator = -1j * beta2 / 2 * omega**2 * L
            gf_propagated_1 = gf * np.exp(propagator)
            pulse_matrix[:, i] = np.fft.ifft(np.fft.fftshift(gf_propagated_1))
    pbar_description = (
        f"Collisions between ({a_chan}"
        + f" and {b_chan}) THz, spacing {freq_spacing} GHz"
    )

    if use_multiprocessing:
        # build a partial function otherwise multiprocessing complains about
        # not being able to pickle stuff
        if speedup_pulse_propagation:
            partial_function = functools.partial(
                X0mm_time_integral_multiprocessing_wrapper, pulse, fiber, wdm, a_chan, b_chan, z
            )
            time_integrals = process_map(
                partial_function, M, leave=False, desc=pbar_description, chunksize=1
            )
        else:
            partial_function = functools.partial(
                X0mm_time_integral_precomputed_multiprocessing_wrapper, pulse_matrix, fiber, z, t, channel_spacing, pulse.T0
            )
            time_integrals = process_map(
                partial_function, M, leave=False, desc=pbar_description, chunksize=1
            )
    else:
        collisions_pbar = tqdm.tqdm(M, leave=False)
        collisions_pbar.set_description(pbar_description)
        time_integrals = []
        for m in collisions_pbar:
            if speedup_pulse_propagation:
                I = X0mm_time_integral_precomputed(
                    pulse_matrix,
                    fiber,
                    z,
                    t,
                    channel_spacing,
                    m,
                    pulse.T0)
            else:
                I = X0mm_time_integral(
                    pulse,
                    fiber,
                    z,
                    channel_spacing,
                    m)
            time_integrals.append(I)

    # convert the list of arrays in a 2d array, since the shape is the same
    I = np.stack(time_integrals)
    return z, time_integrals, M


# Multiprocessing wrapper
def X0mm_time_integral_multiprocessing_wrapper(
    pulse: Pulse,
    fiber: Fiber,
    wdm: WDM,
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    m: int,
    z: float
):
    freq_spacing = get_frequency_spacing(a_chan, b_chan, wdm)
    if isinstance(pulse, NyquistPulse):
        raise NotImplementedError("no Nyquist Pulse yet")
    elif isinstance(pulse, GaussianPulse):
        return X0mm_time_integral_Gaussian(pulse, fiber, wdm, a_chan, b_chan, freq_spacing, m, z)
    else:
        raise NotImplementedError("no generic Pulse yet")


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
def X0mm_time_integral_Gaussian(
    pulse: Pulse,
    fiber: Fiber,
    wdm: WDM,
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    freq_spacing: float,
    m: int,
    z: float
) -> np.ndarray:
    # Apply the fully analytical formula
    if isinstance(fiber, SMFiber):
        l_d = 1 / (np.abs(fiber.beta2) * (pulse.baud_rate)**2)
        factor1 = pulse.baud_rate / (np.sqrt(2 * np.pi))
        factor2 = 1 / np.sqrt(1 + (z / l_d)**2)
        exponent = -((m + fiber.beta2 * (2 * np.pi * freq_spacing) * z)
                     ** 2) / (2 * (1 + (z / l_d)**2))
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
        exponent = -((m + dgd * z)**2) / (2 * (1 + (z / avg_l_d)**2))
        return factor1 * factor2 * np.exp(exponent)
    pass


def X0mm_time_integral_Nyquist(
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


# def Xhkm_precomputed(
#     z: np.ndarray, time_integrals, amplification_function: np.ndarray = None, axis=-1
# ) -> np.ndarray:
#     """Compute the Xhkm XPM coefficients specifying the inner time integral as
#     input.

#     Useful to compare different amplification schemes without re-
#     computing the time integral.
#     """

#     if type(amplification_function) is not np.ndarray:
#         # if the amplification function is not supplied, assume perfect distributed
#         # amplification
#         amplification_function = np.ones_like(z)

#     X = scipy.integrate.trapezoid(time_integrals * amplification_function, z, axis=axis)

#     return X


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
