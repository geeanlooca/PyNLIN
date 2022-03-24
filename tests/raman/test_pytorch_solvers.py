import numpy as np
import matplotlib.pyplot as plt
import torch
from pynlin.fiber import Fiber
from pynlin.raman.pytorch.solvers import RamanAmplifier
from pynlin.raman.solvers import RamanAmplifier as NumpyRamanAmplifier
from pynlin.wdm import WDM
from pynlin.utils import dBm2watt, watt2dBm


def test_pytorch_copropagating_solver_no_raman():
    wdm = WDM(num_channels=50)
    fiber = Fiber(raman_coefficient=0)
    fiber_length = 50e3
    integration_steps = 1000
    z = np.linspace(0, fiber_length, integration_steps)

    power_per_channel = dBm2watt(-5)
    power_per_pump = dBm2watt(10)
    num_pumps = 4
    signal_wavelengths = wdm.wavelength_grid()
    pump_wavelengths = np.array([1450, 1470, 1480, 1490]) * 1e-9

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

    numpy_amplifier = NumpyRamanAmplifier(fiber)
    pump_solution, signal_solution = numpy_amplifier.solve(
        signal_powers, signal_wavelengths, pump_powers, pump_wavelengths, z
    )
    signal_spectrum_numpy = signal_solution[-1]

    torch_input = torch.from_numpy(
        np.concatenate((pump_wavelengths, pump_powers))
    ).float()
    signal_spectrum_torch = torch_amplifier(torch_input.view(1, -1)).numpy().squeeze()

    assert np.allclose(signal_spectrum_numpy, signal_spectrum_torch)


def test_pytorch_copropagating_solver():
    wdm = WDM(num_channels=50)

    fiber = Fiber()
    fiber_length = 50e3
    integration_steps = 1000
    z = np.linspace(0, fiber_length, integration_steps)

    power_per_channel = dBm2watt(-5)
    power_per_pump = dBm2watt(10)
    num_pumps = 4
    signal_wavelengths = wdm.wavelength_grid()
    pump_wavelengths = np.array([1450, 1470, 1480, 1490]) * 1e-9

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

    numpy_amplifier = NumpyRamanAmplifier(fiber)
    pump_solution, signal_solution = numpy_amplifier.solve(
        signal_powers, signal_wavelengths, pump_powers, pump_wavelengths, z
    )
    signal_spectrum_numpy = signal_solution[-1]

    torch_input = torch.from_numpy(
        np.concatenate((pump_wavelengths, pump_powers))
    ).float()
    signal_spectrum_torch = torch_amplifier(torch_input.view(1, -1)).numpy().squeeze()

    assert np.allclose(signal_spectrum_numpy, signal_spectrum_torch)


def test_pytorch_counterpropagating_solver():
    wdm = WDM(num_channels=50)

    fiber = Fiber()
    fiber_length = 50e3
    integration_steps = 1000
    z = np.linspace(0, fiber_length, integration_steps)

    power_per_channel = dBm2watt(-5)
    power_per_pump = dBm2watt(0)
    num_pumps = 4
    signal_wavelengths = wdm.wavelength_grid()
    pump_wavelengths = np.array([1450, 1470, 1480, 1490]) * 1e-9

    signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
    pump_powers = np.ones_like(pump_wavelengths) * power_per_pump

    torch_amplifier = RamanAmplifier(
        fiber_length,
        integration_steps,
        num_pumps,
        signal_wavelengths,
        power_per_channel,
        fiber,
        pump_direction=-1,
    )

    numpy_amplifier = NumpyRamanAmplifier(fiber)
    pump_solution, signal_solution = numpy_amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers,
        pump_wavelengths,
        z,
        pump_direction=-1,
        use_power_at_fiber_start=True,
    )
    signal_spectrum_numpy = signal_solution[-1]

    torch_input = torch.from_numpy(
        np.concatenate((pump_wavelengths, pump_powers))
    ).float()
    signal_spectrum_torch = torch_amplifier(torch_input.view(1, -1)).numpy().squeeze()

    assert np.allclose(signal_spectrum_numpy, signal_spectrum_torch)
