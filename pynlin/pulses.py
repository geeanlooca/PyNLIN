from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy.integrate


class Pulse(ABC):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
    ):
        self.baud_rate = baud_rate
        self.num_symbols = num_symbols
        self.samples_per_symbol = samples_per_symbol
        self.T0 = 1 / self.baud_rate

    @abstractmethod
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the pulse shape and the time axis."""
        pass


class RaisedCosinePulse(Pulse):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
        rolloff: float = 0.1,
    ):
        super().__init__(baud_rate, num_symbols, samples_per_symbol)
        self.rolloff = rolloff
        self._generate()

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.g, self.t

    def _generate(self):
        dt = self.T0 / self.samples_per_symbol
        Ndt = self.samples_per_symbol * self.num_symbols
        t = np.arange(-Ndt / 2, Ndt / 2) * dt
        df = 1 / (max(t) - min(t))
        dw = 2 * np.pi * df
        f = np.arange(-Ndt / 2, Ndt / 2) * df

        rof = self.rolloff
        R = rof
        rate = 1 / self.T0
        freq = f
        wind1 = np.zeros_like(f)
        wind1[np.abs(freq) <= rate * (1 + R) / 2] = 1
        wind2 = np.zeros_like(f)
        wind2[np.abs(freq) >= rate * (1 - R) / 2] = 1
        wind = wind1 * wind2
        # wind1[np.abs(freq) >= R * (1 - R) / 2] = 1
        gf = (1 - wind) + wind * 0.5 * (
            1 + np.cos(np.pi / R / rate * (np.abs(freq) - rate * (1 - R) / 2))
        )
        gf[np.abs(freq) > rate * (1 + R) / 2] = 0

        gt = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(gf))))

        energy = scipy.integrate.trapezoid(np.abs(gt) ** 2, t)
        gt = gt / np.sqrt(energy)

        self.t = t
        self.g = gt

class NyquistPulse(Pulse):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
        rolloff: float = 0.1,
    ):
        super().__init__(baud_rate, num_symbols, samples_per_symbol)
        self.rolloff = rolloff
        self._generate()

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.g, self.t

    def _generate(self):
        dt = self.T0 / self.samples_per_symbol
        Ndt = self.samples_per_symbol * self.num_symbols
        t = np.arange(-Ndt / 2, Ndt / 2) * dt

        gt = np.sinc(t/self.T0)/np.sqrt(self.T0)

        # Correct analytical normalization (finiteness of interval make it imprecise)
        energy = scipy.integrate.trapezoid(np.abs(gt) ** 2, t)
        gt = gt / np.sqrt(energy)

        self.t = t
        self.g = gt

class GaussianPulse(Pulse):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
        rolloff: float = 0.1,
    ):
        super().__init__(baud_rate, num_symbols, samples_per_symbol)
        self.rolloff = rolloff
        self._generate()

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.g, self.t

    def _generate(self):
        dt = self.T0 / self.samples_per_symbol
        Ndt = self.samples_per_symbol * self.num_symbols
        t = np.arange(-Ndt / 2, Ndt / 2) * dt

        gt = np.exp(-t**2/(2*(self.T0**2))) / np.sqrt((np.sqrt(np.pi) * self.T0))
        
        # Correct analytical normalization (finiteness of interval make it imprecise)
        energy = scipy.integrate.trapezoid(np.abs(gt) ** 2, t)
        gt = gt / np.sqrt(energy)

        self.t = t
        self.g = gt
