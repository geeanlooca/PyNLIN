import functools
import math
from typing import Generator, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import tqdm
from scipy.constants import nu2lambda
from tqdm.contrib.concurrent import process_map

from pynlin.constellations import Constellation
from pynlin.fiber import Fiber
from pynlin.pulses import Pulse, RaisedCosinePulse, GaussianPulse, NyquistPulse
from pynlin.wdm import WDM


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


def get_interfering_frequencies(
    channel_of_interest: float,
    frequency_grid: np.ndarray,
) -> List[float]:
    """Given a channel frequency and an  iterable of frequencies, generate a
    list of interfering channel frequencies."""
    combinations = []
    for x in frequency_grid:
        if x != channel_of_interest:
            combinations.append(x)
    return combinations


def X0mm_time_integral_WDM_grid(
    baud_rate: float,
    wdm: WDM,
    fiber: Fiber,
    fiber_length: float,
    filename: str,
    **compute_collisions_kwargs,
) -> None:
    """Compute the inner time integral of the expression for the XPM
    coefficients Xhkm for each combination of frequencies in the supplied WDM
    grid."""
    T = 1 / baud_rate
    beta2 = fiber.beta2

    # open the file to save data to disk
    file = h5py.File(filename, "w")

    # get the WDM comb frequencies and set up the progress bar
    frequency_grid = wdm.frequency_grid()
    print(frequency_grid)
    coi_frequency_grid_pbar = tqdm.tqdm([frequency_grid[0]])
    coi_frequency_grid_pbar.set_description("WDM Channels")

    # iterate over all channels of the WDM grid
    for coi_number, coi_frequency in enumerate(coi_frequency_grid_pbar):
        interfering_frequencies = get_interfering_frequencies(
            coi_frequency, frequency_grid
        )

        z_list = []
        integrals_list = []
        m_list = []

        # set up the progress bar to iterate over all interfering channels
        # of the current channel of interest
        interfering_frequencies_pbar = tqdm.tqdm(interfering_frequencies)
        interfering_frequencies_pbar.set_description(
            f"Interfering frequencies of channel {coi_number}"
        )
        for intererer_number, interfering_frequency in enumerate(
            interfering_frequencies_pbar
        ):

            # compute the frequency separation between the coi and the
            # interfering channel
            channel_spacing = interfering_frequency - coi_frequency

            # compute the time integrals for each complete collision
            z, I, M = compute_all_collisions_X0mm_time_integrals(
                coi_frequency,
                interfering_frequency,
                baud_rate,
                fiber,
                fiber_length,
                **compute_collisions_kwargs,
            )

            z_list.append(z)
            integrals_list.append(I)
            m_list.append(M)

        # Save the results to file

        # each Channel of interest gets its own group with its attributes
        group_name = f"time_integrals/channel_{coi_number}/"
        group = file.create_group(group_name)
        group.attrs["frequency"] = coi_frequency
        group.attrs["frequency_THz"] = coi_frequency * 1e-12
        group.attrs["wavelength"] = nu2lambda(coi_frequency) * 1e9

        # in each COI group, create a group for each interfering channel
        # and store the z-array (positions inside the fiber)
        # and the time integrals for each collision.
        for interf_number, (z, integral, m, interf_freq) in enumerate(
            zip(z_list, integrals_list, m_list, interfering_frequencies)
        ):

            interferer_group_name = group_name + f"interfering_channel_{interf_number}/"
            interferer_group = file.create_group(interferer_group_name)
            interferer_group.attrs["frequency"] = interf_freq
            interferer_group.attrs["frequency_THz"] = interf_freq * 1e-12
            interferer_group.attrs["wavelength"] = nu2lambda(interf_freq) * 1e9

            dset = file.create_dataset(interferer_group_name + "z", data=z)
            dset = file.create_dataset(interferer_group_name + "m", data=m)
            # integrals_group_name = interferer_group_name + "/integrals/"
            # for x, integral in enumerate(I_list):
            file.create_dataset(
                interferer_group_name + f"integrals",
                data=integral,
                compression="gzip",
                compression_opts=9,
            )


def compute_all_collisions_X0mm_time_integrals(
    frequency_of_interest: float,
    interfering_frequency: float,
    baud_rate: float,
    fiber: Fiber,
    fiber_length: float,
    pulse_shape: str,
    rolloff_factor: float = 0.1,
    samples_per_symbol=5,
    points_per_collision: int = 10,
    use_multiprocessing: bool = False,
    partial_collisions_start: int = 10,
    partial_collisions_end: int = 10,
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
    T = 1 / baud_rate
    channel_spacing = interfering_frequency - frequency_of_interest
    M = get_m_values(
        fiber,
        fiber_length,
        channel_spacing,
        T,
        partial_collisions_end=partial_collisions_end,
        partial_collisions_start=partial_collisions_start,
    )
    # first, create the Pulse object with the appropriate parameters

    # compute the maximum delay between pulses and use it to set
    # the number of symbols in the rcos signal
    tau_max = fiber.beta2 * fiber_length * 2 * np.pi * channel_spacing
    num_symbols = np.abs(4 * math.ceil(tau_max / T))

    if pulse_shape == "RaisedCosine":
        pulse = RaisedCosinePulse(baud_rate=baud_rate,
                                  samples_per_symbol=samples_per_symbol,
                                  num_symbols=num_symbols,
                                  rolloff=rolloff_factor,)
    elif pulse_shape == "Nyquist":
        pulse = NyquistPulse(baud_rate=baud_rate,
                             samples_per_symbol=samples_per_symbol,
                             num_symbols=num_symbols,
                             rolloff=rolloff_factor,)
    elif pulse_shape == "Gaussian":
        pulse = GaussianPulse(baud_rate=baud_rate,
                              samples_per_symbol=samples_per_symbol,
                              num_symbols=num_symbols,
                              rolloff=rolloff_factor,)
    else:
        print("Invalid pulse_shape, using Raised Cosine...")
        pulse = RaisedCosinePulse(baud_rate=baud_rate,
                                  samples_per_symbol=samples_per_symbol,
                                  num_symbols=num_symbols,
                                  rolloff=rolloff_factor,)

    npoints_z = math.ceil(len(M) * points_per_collision)
    z = np.linspace(0, fiber_length, npoints_z)

    channel_spacing_GHz = channel_spacing * 1e-9
    interfering_frequency_THz = interfering_frequency * 1e-12
    coi_frequency_THz = frequency_of_interest * 1e-12

    pbar_description = (
        f"Collisions between channels at ({coi_frequency_THz}"
        + f" and {interfering_frequency_THz}) THz, spacing {channel_spacing_GHz} GHz"
    )
    if use_multiprocessing:
        # build a partial function otherwise multiprocessing complains about
        # not being able to pickle stuff
        partial_function = functools.partial(
            X0mm_time_integral_multiprocessing_wrapper, pulse, fiber, z, channel_spacing
        )
        time_integrals = process_map(
            partial_function, M, leave=False, desc=pbar_description, chunksize=1
        )

    else:
        collisions_pbar = tqdm.tqdm(M, leave=False)
        collisions_pbar.set_description(pbar_description)
        time_integrals = []
        for m in collisions_pbar:
            I = X0mm_time_integral(pulse, fiber, z, channel_spacing, m)
            time_integrals.append(I)

    # convert the list of arrays in a 2d array, since the shape is the same
    I = np.stack(time_integrals)
    return z, time_integrals, M


def X0mm_time_integral_multiprocessing_wrapper(
    pulse: Pulse, fiber: Fiber, z: np.ndarray, channel_spacing: float, m: int
):
    """A wrapper for the `X0mm_time_integral` function to easily
    `functool.partial` and enable multiprocessing."""
    return X0mm_time_integral(pulse, fiber, z, channel_spacing, m)


def Xhkm_time_integral_multiprocessing_wrapper(
    pulse: Pulse, fiber: Fiber, z: np.ndarray, channel_spacing: float, m: int
):
    """A wrapper for the `Xhkm_time_integral` function to easily
    `functool.partial` and enable multiprocessing."""
    return Xhkm_time_integral(pulse, fiber, z, channel_spacing, 0, m, m)


def get_m_values(
    fiber: Fiber,
    fiber_length: float,
    channel_spacing: float,
    T: float,
    partial_collisions_start=10,
    partial_collisions_end=10,
) -> np.ndarray:
    """Get values of the m indeces to compute the X0mm XPM coefficients for.

    Computes those indeces for which the collisions fall inside the
    fiber. By default, 10 extra partial collisions at each end of the
    fiber are computed. This parameter can be controlled by the
    `partial_collisions_start` and `partial_collisions_end` kwargs.
    """
    m_max = -fiber_length * fiber.beta2 * 2 * math.pi * channel_spacing / T

    if m_max < 0:
        m_max = math.ceil(m_max)
        return np.arange(m_max - partial_collisions_start, partial_collisions_end + 1)
    else:
        m_max = math.floor(m_max)
        return np.arange(-partial_collisions_start, m_max + partial_collisions_end + 1)


def get_collision_location(m, fiber, channel_spacing, T) -> float:
    """For the specified index m, compute the position of the corresponding
    complete collision."""
    return -m * T / (fiber.beta2 * 2 * math.pi * channel_spacing)


def X0mm_time_integral(
    pulse: Pulse,
    fiber: Fiber,
    z: np.ndarray,
    channel_spacing: float,
    m: int,
) -> np.ndarray:
    """Compute the inner time integral of the expression for the XPM
    coefficients Xhkm for the specified channel spacing."""
    npoints_z = len(z)
    g, t = pulse.data()
    dt = t[1] - t[0]
    nsamples = len(g)
    freq = np.fft.fftfreq(nsamples, d=dt)
    omega = 2 * np.pi * freq
    omega = np.fft.fftshift(omega)
    O = 2 * np.pi * channel_spacing
    T = pulse.T0
    beta2 = fiber.beta2
    time_integrals = np.zeros((npoints_z,), dtype=np.complex)
    gf = np.fft.fftshift(np.fft.fft(g))
    max_prop = 0
    for i, L in enumerate(z):
        # for each point in space, calculate the corresponding time integral
        propagator = -1j * beta2 / 2 * omega**2 * L
        delay = np.exp(-1j * (m * T + beta2 * O * L) * omega)

        gf_propagated_1 = gf * np.exp(propagator)
        gf_propagated_3 = gf_propagated_1 * delay

        g1 = np.fft.ifft(np.fft.fftshift(gf_propagated_1))
        g3 = np.fft.ifft(np.fft.fftshift(gf_propagated_3))

        integrand = np.conj(g1) * g1 * np.conj(g3) * g3
        time_integrals[i] = scipy.integrate.trapezoid(integrand, t)

    return time_integrals


def Xhkm_time_integral(
    pulse: Pulse,
    fiber: Fiber,
    z: np.ndarray,
    channel_spacing: float,
    h: int,
    k: int,
    m: int,
) -> np.ndarray:
    """Compute the inner time integral of the expression for the XPM
    coefficients Xhkm for the specified channel spacing."""
    npoints_z = len(z)
    g, t = pulse.data()
    O = 2 * np.pi * channel_spacing
    T = pulse.T0
    beta2 = fiber.beta2
    time_integrals = np.zeros((npoints_z,), dtype=np.complex)

    for i, L in enumerate(z):
        # for each point in space, calculate the corresponding time integral
        g1 = apply_chromatic_dispersion(pulse, fiber, L, delay=0)
        g2 = apply_chromatic_dispersion(pulse, fiber, L, delay=h * T)
        g3 = apply_chromatic_dispersion(pulse, fiber, L, delay=k * T + beta2 * O * L)
        g4 = apply_chromatic_dispersion(pulse, fiber, L, delay=m * T + beta2 * O * L)

        integrand = np.conj(g1) * g2 * np.conj(g3) * g4
        time_integrals[i] = scipy.integrate.trapezoid(integrand, t)

    return time_integrals


def Xhkm_precomputed(
    z: np.ndarray, time_integrals, amplification_function: np.ndarray = None, axis=-1
) -> np.ndarray:
    """Compute the Xhkm XPM coefficients specifying the inner time integral as
    input.

    Useful to compare different amplification schemes without re-
    computing the time integral.
    """

    if type(amplification_function) is not np.ndarray:
        # if the amplification function is not supplied, assume perfect distributed
        # amplification
        amplification_function = np.ones_like(z)

    X = scipy.integrate.trapezoid(time_integrals * amplification_function, z, axis=axis)

    return X


def Xhkm(
    pulse: Pulse,
    fiber: Fiber,
    amplification_function: np.ndarray,
    z: np.ndarray,
    channels_spacing: float,
    h: int,
    k: int,
    m: int,
) -> np.ndarray:
    """Compute the Xhkm XPM coefficients."""
    time_integrals = Xhkm_time_integral(pulse, fiber, z, channels_spacing, h, k, m)

    # integrate in space
    X = scipy.integrate.trapezoid(time_integrals * amplification_function, z)
    return X
