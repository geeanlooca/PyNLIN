import numpy as np
import matplotlib.pyplot as plt
import pynlin.fiber
import pynlin.raman.solvers
import pynlin.wdm
import pynlin.utils


def test_raman_amplifier():
    Ps0 = pynlin.utils.dBm2watt(-6)
    Pp0 = 500e-3
    fiber = pynlin.fiber.Fiber()
    fiber_length = 50e3
    num_points = 1000

    wdm = pynlin.wdm.WDM(num_channels=50)
    signal_wavelengths = wdm.wavelength_grid()
    signal_powers = np.ones_like(signal_wavelengths) * Ps0
    pump_wavelengths = np.array([1450, 1470, 1480, 1490]) * 1e-9
    pump_powers = np.ones_like(pump_wavelengths) * Pp0
    amplifier = pynlin.raman.solvers.RamanAmplifier(fiber)

    z = np.linspace(0, fiber_length, num_points)

    pump_solution, signal_solution = amplifier.solve(
        signal_powers, signal_wavelengths, pump_powers, pump_wavelengths, z
    )
    pump_solution_off, signal_solution_off = amplifier.solve(
        signal_powers, signal_wavelengths, 0 * pump_powers, pump_wavelengths, z
    )

    plt.figure()
    plt.plot(z, pynlin.utils.watt2dBm(pump_solution), color="red")
    plt.plot(z, pynlin.utils.watt2dBm(signal_solution), color="black")
    plt.xlabel("Position [m]")
    plt.ylabel("Power [dBm]")

    plt.figure()
    plt.plot(
        signal_wavelengths * 1e9,
        10 * np.log10(signal_solution[-1] / signal_solution_off[-1]),
        label="On-off gain",
    )
    plt.plot(
        signal_wavelengths * 1e9,
        10 * np.log10(signal_solution[-1] / signal_solution[0]),
        label="Net gain",
    )
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Gain [dB]")
    plt.legend()

    plt.show()


def test_raman_amplifier_counterpump_jiang():
    Ps0 = pynlin.utils.dBm2watt(-6)
    fiber = pynlin.fiber.Fiber()
    fiber_length = 50e3
    num_points = 1000

    wdm = pynlin.wdm.WDM(num_channels=50)
    signal_wavelengths = wdm.wavelength_grid()
    signal_powers = np.ones_like(signal_wavelengths) * Ps0
    pump_wavelengths = np.array([1450, 1470, 1480, 1490]) * 1e-9
    pump_powers = np.array([500, 900, 300, 200]) * 1e-3
    amplifier = pynlin.raman.solvers.RamanAmplifier(fiber)

    z = np.linspace(0, fiber_length, num_points)

    pump_solution, signal_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers,
        pump_wavelengths,
        z,
        pump_direction=-1,
        shooting_solver="scipy",
    )

    plt.figure()
    plt.subplot(121)
    plt.plot(z, pynlin.utils.watt2dBm(pump_solution), color="red")
    plt.plot(z, pynlin.utils.watt2dBm(signal_solution), color="black")

    for x in pump_powers:
        plt.axhline(pynlin.utils.watt2dBm(x), color="green")
    plt.xlabel("Position [m]")
    plt.ylabel("Power [dBm]")

    plt.subplot(122)
    plt.plot(
        signal_wavelengths * 1e9,
        10 * np.log10(signal_solution[-1] / signal_solution[0]),
        label="Net gain",
    )

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Gain [dB]")
    plt.legend()

    plt.show()
