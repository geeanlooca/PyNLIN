import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy.constants import epsilon_0, lambda2nu

from pynlin.utils import c0

positions = (
    np.array(
        [
            56.25,
            100.00,
            231.25,
            362.50,
            463.00,
            497.00,
            611.50,
            691.67,
            793.67,
            835.50,
            930.00,
            1080.00,
            1215.00,
        ]
    )
    * 1e2
)

intensities = np.array(
    [1.00, 11.40, 36.67, 67.67, 74.00, 4.50, 6.80, 4.60, 4.20, 4.50, 2.70, 3.10, 3.00]
)

g_fwhm = (
    np.array(
        [
            52.10,
            110.42,
            175.00,
            162.50,
            135.33,
            24.50,
            41.50,
            155.00,
            59.50,
            64.30,
            150.00,
            91.00,
            160.00,
        ]
    )
    * 1e2
)

l_fwhm = (
    np.array(
        [
            17.37,
            38.81,
            58.33,
            54.17,
            45.11,
            8.17,
            13.83,
            51.67,
            19.83,
            21.43,
            50.00,
            30.33,
            53.33,
        ]
    )
    * 1e2
)

# Vibrational frequencies
omega_v = 2 * np.pi * c0 * positions

# Lorentzian FWHM
gamma = np.pi * c0 * l_fwhm

# Gaussian FWHM
Gamma = np.pi * c0 * g_fwhm

# Amplitude
amplitudes = intensities * omega_v

# Number of vibrational components
num_components = len(positions)


def impulse_response(duration, fs, normalize=False, num_samples=None):
    """Compute the Raman impulse response of silica.

    Parameters
    ----------
    duration : float
        The duration of the impulse response in seconds.
    fs : float
        The sampling frequency.
    normalize : bool, optional
        Normalize to its maximum value, by default False
    num_samples : int, optional
        If specified, forces the number of samples, by default None

    Returns
    -------
    response : array_like
        The impulse response samples.
    t : array_like
        The time samples.
    """

    dt = 1 / fs

    if num_samples is None:
        num_samples = np.math.ceil(duration / dt)

    t = np.arange(num_samples) * dt

    modes = (
        np.reshape(amplitudes / omega_v, (num_components, 1))
        * np.exp(-np.outer(gamma, t))
        * np.exp(-np.outer(Gamma**2, t**2 / 4))
        * np.sin(np.outer(omega_v, t))
    )

    response = np.sum(modes, axis=0)

    if normalize:
        response /= np.max(np.abs(response))

    return response, t


def gain_spectrum(frequencies, spacing=20e9, normalize=False, spline_order=4):
    """Compute the Raman gain spectrum of silica.

    Parameters
    ----------
    frequencies : array_like
        The frequency points at which the Raman gain must be computed.
    spacing : float, optional
        Desired frequency resolution in Hertz, by default 20e9
    normalize : bool, optional
        Normalize the gain profile to its maximum, by default False
    spline_order : int, optional
        The order of the spline used to interpolate the spectrum, by default 4

    Returns
    -------
    gains : array_like
        The interpolated gain values computed at frequencies.
    gain : array_like
        The entire Raman gain spectrum.
    f : array_like
        The frequency axis.
    """

    # Set the sampling frequency based on the maximum frequency requested
    fs = np.max(np.abs(frequencies)) * 10

    #negative_idx = np.argwhere()

    # fs = max(fs, 80e12)

    num_samples = np.math.ceil(fs / spacing)

    # get the frequency response
    response, t = impulse_response(None, fs, num_samples=num_samples)

    # compute the frequency axis
    dt = 1 / fs
    duration = (num_samples - 1) * dt
    dF = fs / num_samples

    # ! This is not correct: at f = 0, gain is != 0
    # ! Maybe take the modulus of the negative frequencies and switch the sign
    # ! of the gain to force simmetry around f = 0
    f = (np.arange(num_samples)) * dF - (num_samples - 1) * dF / 2

    # obtain the Raman gain as the imaginary part of
    # the spectrum
    gain = -np.imag(np.fft.fft(response))
    gain = np.fft.fftshift(gain)

    if normalize:
        gain /= np.max(np.abs(gain))

    # interpolate at the desired frequency spacings
    spline = scipy.interpolate.splrep(f, gain, k=spline_order)
    gains = scipy.interpolate.splev(frequencies, spline)

    return gains, gain, f


def agrawal_copolarized_response(duration, fs):
    dt = 1 / fs

    num_samples = np.math.ceil(duration / dt)

    t = np.arange(num_samples) * dt

    tau1 = 12.2e-15
    tau2 = 32e-15

    response = (
        tau1
        * (1 / (tau1**2) + 1 / (tau2**2))
        * np.exp(-t / tau2)
        * np.sin(t / tau1)
    )

    return response, t


def agrawal_crosspolarized_response(duration, fs):
    dt = 1 / fs

    num_samples = np.math.ceil(duration / dt)

    t = np.arange(num_samples) * dt

    tau1 = 12.2e-15
    tau2 = 32e-15
    taub = 96e-15

    response = (2 * taub - t) / (taub**2) * np.exp(-t / taub)

    return response, t


if __name__ == "__main__":

    wavelength = 514e-9
    fs = 1e15
    duration = 1e-11
    fa, fb, fc = 0.75, 0.21, 0.04
    fR = 0.245

    n2 = 2.6e-20
    gamma = n2 * lambda2nu(wavelength) * 2 * np.pi / c0
    print(gamma)

    copolarized, t = agrawal_copolarized_response(duration, fs)
    crosspolarized, t = agrawal_crosspolarized_response(duration, fs)

    nsamp = len(t)
    f = np.arange(nsamp) * fs / nsamp

    copolarized *= fa + fc
    crosspolarized *= fb

    total_response = fR * (copolarized + crosspolarized)
    total_response_spectrum = np.fft.fft(total_response) / fs
    total_gain_spectrum = 2 * gamma * np.imag(total_response_spectrum)

    co_spectrum = -np.imag(np.fft.fft(copolarized)) / fs
    cross_spectrum = -np.imag(np.fft.fft(crosspolarized)) / fs
    real_co = np.real(np.fft.fft(copolarized)) / fs
    real_cross = np.real(np.fft.fft(crosspolarized)) / fs

    a_tilde = np.fft.fft(copolarized) / fs
    b_tilde = np.fft.fft(crosspolarized) / fs

    # in cm/W
    co_gain = 2 * gamma * fR * co_spectrum * epsilon_0 * 1e2
    cross_gain = 2 * gamma * fR * cross_spectrum * epsilon_0 * 1e2

    # at 526 nm
    freq_scaling = lambda2nu(526e-9) / lambda2nu(wavelength)

    plt.figure()
    plt.plot(f * 1e-12, -total_gain_spectrum * 1e2, label="795.5 nm")
    plt.plot(f * 1e-12, -total_gain_spectrum * 1e2 * freq_scaling, label="526 nm")
    plt.xlabel("Frequency [THz]")
    plt.ylabel("Total gain spectrum [cm/W]")
    plt.legend()
    plt.xlim(0, 40)
    plt.ylim(0, 2e-11)
    plt.minorticks_on()
    plt.grid(which="both")

    plt.figure()
    plt.plot(f * 1e-12, np.real(total_response_spectrum), label="Real")
    plt.plot(f * 1e-12, np.imag(total_response_spectrum), label="Imaginary")
    plt.xlabel("Frequency [THz]")
    plt.ylabel("Total response spectrum")
    plt.legend()
    plt.xlim(0, 40)
    plt.minorticks_on()
    plt.grid(which="both")

    # Real and imaginary parts of A(f), B(f)
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(f * 1e-12, np.real(a_tilde), label=r"$\tilde{a}$, real")
    ax[0].plot(f * 1e-12, np.imag(a_tilde), label=r"$\tilde{a}$, imag")
    ax[0].set_xlabel("frequency [thz]")
    ax[0].set_xlim(0, 40)
    ax[0].legend()

    ax[1].plot(f * 1e-12, np.real(b_tilde), label=r"$\tilde{b}$, real")
    ax[1].plot(f * 1e-12, np.imag(b_tilde), label=r"$\tilde{b}$, imag")
    ax[1].set_xlabel("frequency [thz]")
    ax[1].set_xlim(0, 40)
    ax[1].legend()

    plt.show()
