import matplotlib.pyplot as plt
import numpy as np
import pytest

import pynlin.fiber
import pynlin.raman.pytorch
import pynlin.raman.pytorch.solvers
import pynlin.raman.solvers
from pynlin.raman.pytorch.solvers import MMFRamanAmplifier
import pynlin.utils
import pynlin.wdm
from pynlin.utils import dBm2watt, watt2dBm

def test_raman_amplifier():
    Ps0 = pynlin.utils.dBm2watt(-5)
    Pp0 = 500e-3
    fiber = pynlin.fiber.Fiber()
    fiber_length = 50e3
    num_points = 1000

    wdm = pynlin.wdm.WDM(num_channels=50, spacing=200)
    signal_wavelengths = wdm.wavelength_grid()
    signal_powers = np.ones_like(signal_wavelengths) * Ps0
    pump_wavelengths = (
        np.array([1411.0216, 1418.8397, 1429.0806, 1445.8669, 1458.7216, 1476.1899])
        * 1e-9
    )
    pump_powers = np.ones_like(pump_wavelengths) * Pp0
    pump_powers = np.abs(np.array([0.0896, 0.0550, -0.0677, 0.0381, -0.0095, -0.0623]))
    amplifier = pynlin.raman.solvers.RamanAmplifier(fiber)

    z = np.linspace(0, fiber_length, num_points)

    pump_solution, signal_solution, ase_solution = amplifier.solve(
        signal_powers, signal_wavelengths, pump_powers, pump_wavelengths, z
    )
    pump_solution_off, signal_solution_off, ase_solution = amplifier.solve(
        signal_powers, signal_wavelengths, 0 * pump_powers, pump_wavelengths, z
    )


def test_raman_amplifier_counterpumping_with_power_at_z0():
    Ps0 = pynlin.utils.dBm2watt(-5)
    Pp0 = dBm2watt(5)
    fiber = pynlin.fiber.Fiber()
    fiber_length = 50e3
    num_points = 1000

    wdm = pynlin.wdm.WDM(num_channels=50, spacing=200)
    signal_wavelengths = wdm.wavelength_grid()
    signal_powers = np.ones_like(signal_wavelengths) * Ps0
    pump_wavelengths = np.array([1411.0216, 1418.8397, 1429.0806, 1445.8669]) * 1e-9
    pump_powers = np.ones_like(pump_wavelengths) * Pp0
    amplifier = pynlin.raman.solvers.RamanAmplifier(fiber)

    z = np.linspace(0, fiber_length, num_points)

    pump_solution, signal_solution, ase_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers,
        pump_wavelengths,
        z,
        pump_direction=-1,
        use_power_at_fiber_start=True,
    )

    # plt.figure()
    # plt.plot(z, watt2dBm(signal_solution), color="k")
    # plt.plot(z, watt2dBm(pump_solution), color="r")

    # plt.figure()
    # plt.plot(signal_wavelengths, watt2dBm(signal_solution[-1]), color="k")
    # plt.show()


def test_raman_amplifier_counterpumping_with_power_at_z0_exception():
    Ps0 = pynlin.utils.dBm2watt(-5)
    Pp0 = dBm2watt(5)
    fiber = pynlin.fiber.Fiber()
    fiber_length = 50e3
    num_points = 1000

    wdm = pynlin.wdm.WDM(num_channels=50, spacing=200)
    signal_wavelengths = wdm.wavelength_grid()
    signal_powers = np.ones_like(signal_wavelengths) * Ps0
    pump_wavelengths = np.array([1411.0216, 1418.8397, 1429.0806, 1445.8669]) * 1e-9
    pump_powers = np.ones_like(pump_wavelengths) * Pp0
    amplifier = pynlin.raman.solvers.RamanAmplifier(fiber)

    z = np.linspace(0, fiber_length, num_points)

    with pytest.raises(NotImplementedError):
        pump_solution, signal_solution, ase_solution = amplifier.solve(
            signal_powers,
            signal_wavelengths,
            pump_powers,
            pump_wavelengths,
            z,
            pump_direction=-1,
            use_power_at_fiber_start=False,
        )

def test_raman_amplifier_MMF():
    Ps0 = pynlin.utils.dBm2watt(-5)
    Pp0 = 500e-3

    fiber = pynlin.fiber.MMFiber(modes=4,
        overlap_integrals=np.array([1e-9, 0, 0])[:, None, None].repeat(4, axis=1).repeat(4, axis=2),
        mode_names=None,)
    fiber_length = 50e3
    num_points = 1000

    wdm = pynlin.wdm.WDM(num_channels=50, spacing=200)
    signal_wavelengths = wdm.wavelength_grid()
    signal_powers = np.ones_like(signal_wavelengths) * Ps0
    pump_wavelengths = (
        np.array([1411.0216, 1418.8397, 1429.0806, 1445.8669, 1458.7216, 1476.1899])
        * 1e-9
    )
    pump_powers = np.ones_like(pump_wavelengths) * Pp0
    pump_powers = np.abs(np.array([0.0896, 0.0550, -0.0677, 0.0381, -0.0095, -0.0623]))
    
    ## change into MMFAmplifier
    amplifier = pynlin.raman.solvers.RamanAmplifier(fiber)

    z = np.linspace(0, fiber_length, num_points)

    pump_solution, signal_solution, ase_solution = amplifier.solve(
        signal_powers, signal_wavelengths, pump_powers, pump_wavelengths, z
    )
    pump_solution_off, signal_solution_off, ase_solution = amplifier.solve(
        signal_powers, signal_wavelengths, 0 * pump_powers, pump_wavelengths, z
    )

def test_torch_raman_amplifier():
  pass
    # Ps0 = pynlin.utils.dBm2watt(-5)
    # Pp0 = 500e-3

    # fiber = pynlin.fiber.MMFiber(modes=4,
    #     overlap_integrals=oi,
    #     mode_names=None,)
    # fiber_length = 50e3
    # num_points = 1000

    # wdm = pynlin.wdm.WDM(num_channels=50, spacing=200)
    # signal_wavelengths = wdm.wavelength_grid()
    # signal_powers = np.ones_like(signal_wavelengths) * Ps0
    # pump_wavelengths = (
    #     np.array([1411.0216, 1418.8397, 1429.0806, 1445.8669, 1458.7216, 1476.1899])
    #     * 1e-9
    # )
    # pump_powers = np.ones_like(pump_wavelengths) * Pp0
    # pump_powers = np.abs(np.array([0.0896, 0.0550, -0.0677, 0.0381, -0.0095, -0.0623]))
    
    # ## change into MMFAmplifier

   
    # amplifier = pynlin.raman.pytorch.solvers.RamanAmplifier(
    #      fiber=fiber,
    #      integration_steps=100, 
    #      num_pumps=10,
    #      signal_wavelengths=1000)

    # z = np.linspace(0, fiber_length, num_points)

    # pump_solution, signal_solution, ase_solution = amplifier.solve(
    #     signal_powers, signal_wavelengths, pump_powers, pump_wavelengths, z
    # )
    # pump_solution_off, signal_solution_off, ase_solution = amplifier.solve(
    #     signal_powers, signal_wavelengths, 0 * pump_powers, pump_wavelengths, z
    # )