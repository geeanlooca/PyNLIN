import numpy as np
import matplotlib.pyplot as plt
import torch
from pynlin.fiber import Fiber
from pynlin.raman.pytorch.solvers import RamanAmplifier
from pynlin.raman.solvers import RamanAmplifier as NumpyRamanAmplifier
from pynlin.wdm import WDM
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.raman.pytorch.gain_optimizer import CopropagatingOptimizer

from scipy.constants import lambda2nu, nu2lambda


def test_pytorch_optimizer():
    wdm = WDM(num_channels=50, spacing=200)

    fiber = Fiber()
    fiber_length = 50e3
    integration_steps = 100
    num_pumps = 12
    pump_band_b = lambda2nu(1500e-9)
    pump_band_a = lambda2nu(1390e-9)
    initial_pump_frequencies = np.linspace(pump_band_a, pump_band_b, num_pumps)

    power_per_channel = dBm2watt(-5)
    power_per_pump = dBm2watt(0)
    signal_wavelengths = wdm.wavelength_grid()
    pump_wavelengths = nu2lambda(initial_pump_frequencies) * 1e9
    num_pumps = len(pump_wavelengths)

    signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
    pump_powers = np.ones_like(pump_wavelengths) * power_per_pump

    torch_amplifier = RamanAmplifier(
        fiber_length,
        integration_steps,
        num_pumps,
        signal_wavelengths,
        power_per_channel,
        fiber,
    )

    optimizer = CopropagatingOptimizer(
        torch_amplifier,
        torch.from_numpy(pump_wavelengths),
        torch.from_numpy(pump_powers),
    )

    input_power, output_spectrum, pump_wavelengths, pump_powers = optimizer.optimize(
        epochs=1000
    )
    amplifier = NumpyRamanAmplifier(fiber)

    z = np.linspace(0, fiber_length, integration_steps)

    pump_solution, signal_solution = amplifier.solve(
        signal_powers, signal_wavelengths, pump_powers, pump_wavelengths, z
    )

    plt.figure()
    plt.plot(z, watt2dBm(pump_solution), color="red")
    plt.plot(z, watt2dBm(signal_solution), color="black")
    plt.xlabel("Position [m]")
    plt.ylabel("Power [dBm]")

    plt.figure()
    plt.stem(lambda2nu(pump_wavelengths) * 1e-12, watt2dBm(pump_powers), markerfmt=".")
    plt.stem(
        lambda2nu(signal_wavelengths) * 1e-12,
        watt2dBm(signal_solution[-1]),
        markerfmt=".",
    )

    plt.figure()
    plt.plot(signal_wavelengths * 1e9, (input_power))
    plt.plot(signal_wavelengths * 1e9, (output_spectrum))
    plt.plot(signal_wavelengths * 1e9, watt2dBm(signal_solution[-1]))
    plt.show()
