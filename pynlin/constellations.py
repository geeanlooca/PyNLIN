from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Constellation(ABC):
    @abstractmethod
    def symbols(self) -> np.ndarray:
        pass

    @staticmethod
    def average_power(symbols: np.ndarray) -> float:
        return (np.abs(symbols) ** 2).mean()


class PSK(Constellation):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def symbols(self) -> np.ndarray:
        """
        Generate M-PSK symbols
        Parameters
        ----------
        M : int
            PSK order
        Returns
        -------
        symbols: array_like
            PSK symbols  normalised  to power of 1
        """
        M = self.M
        if M == 4:  # QPSK is rotated by pi/4 in contrast to others
            return np.exp(1j * (np.arange(M) * 2 * np.pi / M + np.pi / M))
        else:
            return np.exp(2j * np.arange(M) * np.pi / M)


class QPSK(PSK):
    def __init__(self):
        super().__init__(4)


class QAM(Constellation):
    def __init__(self, M):
        super().__init__()
        self.M = M

        self.symbol_generating_fn = (
            self._symbols_cross_qam
            if np.log2(M) % 2 > 0.5
            else self._symbols_square_qam
        )

    def symbols(self) -> np.ndarray:
        """Generate the symbols on the constellation diagram for M-QAM."""
        symbols = self.symbol_generating_fn()
        scale = Constellation.average_power(symbols) ** 0.5
        return symbols / scale

    def _symbols_square_qam(self) -> np.ndarray:
        """Generate the symbols on the constellation diagram for square
        M-QAM."""
        qam = np.mgrid[
            -(2 * np.sqrt(self.M) / 2 - 1) : 2 * np.sqrt(self.M) / 2
            - 1 : 1.0j * np.sqrt(self.M),
            -(2 * np.sqrt(self.M) / 2 - 1) : 2 * np.sqrt(self.M) / 2
            - 1 : 1.0j * np.sqrt(self.M),
        ]
        return (qam[0] + 1.0j * qam[1]).flatten()

    def _symbols_cross_qam(self) -> np.ndarray:
        """Generate the symbols on the constellation diagram for non-square
        (cross) M-QAM."""

        N = (np.log2(self.M) - 1) / 2
        s = 2 ** (N - 1)
        rect = np.mgrid[
            -(2 ** (N + 1) - 1) : 2 ** (N + 1) - 1 : 1.0j * 2 ** (N + 1),
            -(2**N - 1) : 2**N - 1 : 1.0j * 2**N,
        ]
        qam = rect[0] + 1.0j * rect[1]
        idx1 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) > s))
        idx2 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) <= s))
        qam[idx1] = np.sign(qam[idx1].real) * (abs(qam[idx1].real) - 2 * s) + 1.0j * (
            np.sign(qam[idx1].imag) * (4 * s - abs(qam[idx1].imag))
        )
        qam[idx2] = np.sign(qam[idx2].real) * (4 * s - abs(qam[idx2].real)) + 1.0j * (
            np.sign(qam[idx2].imag) * (abs(qam[idx2].imag) + 2 * s)
        )
        return qam.flatten()
