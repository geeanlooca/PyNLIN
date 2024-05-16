from typing import Tuple

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import MSELoss
from torch.optim.adam import Adam
from pynlin.utils import dBm2watt
from pynlin.raman.pytorch.solvers import RamanAmplifier


def dBm(x: torch.Tensor) -> torch.Tensor:
    """Convert a tensor from Watt to dBm."""
    return 10 * torch.log10(x) + 30


class GainOptimizer(nn.Module):
    """PyTorch module for finding the power and wavelength of each pump of a
    Raman amplifier to obtain a target output signal spectrum."""

    def __init__(
        self,
        raman_torch_solver: RamanAmplifier,
        initial_pump_wavelengths: torch.Tensor,
        initial_pump_powers: torch.Tensor,
    ):
        super(GainOptimizer, self).__init__()
        self.raman_solver = raman_torch_solver
        scaled_wavelengths, self.wavelength_scaling = self.scale(
            initial_pump_wavelengths.float()
        )
        self.pump_powers = nn.Parameter(initial_pump_powers.float())
        self.pump_wavelengths = nn.Parameter(scaled_wavelengths)

    def forward(self, wavelengths: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
        """Compute the output spectrum of the Raman amplifier given pump
        parameters."""
        x = torch.cat((wavelengths, powers)).view(1, -1).float()
        return dBm(self.raman_solver(x).float())

    def scale(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Scale the input tensor to the range [0, 1]."""
        m = torch.min(x)
        d = torch.max(x) - m
        return (x - m) / d, (m, d)

    def unscale(self, x: torch.Tensor, m: float, d: float) -> torch.Tensor:
        """Unscale the input tensor to the original range."""
        return (x * d) + m

    def optimize(
        self,
        target_spectrum: np.ndarray = None,
        epochs: int = 100,
        learning_rate: float = 2e-2,
        lock_wavelengths: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the optimization algorithm."""

        if target_spectrum is None:
            _target_spectrum = dBm(
                self.raman_solver.signal_power
                * torch.ones_like(self.raman_solver.signal_wavelengths).view(1, -1)
            ).float()
        else:
            print("Overriding the loading of the target spectrum")
            _target_spectrum = torch.from_numpy(target_spectrum).float()
            # _target_spectrum = torch.from_numpy(target_spectrum).view(1, -1).float()

        torch_optimizer = Adam(self.parameters(), lr=learning_rate)
        loss_function = MSELoss()

        best_loss = torch.inf
        best_wavelengths = torch.clone(self.pump_wavelengths)
        best_powers = torch.clone(self.pump_powers)

        # do not optimize wavelengths for the first `lock_wavelengths` epochs
        self.pump_wavelengths.requires_grad = False
        reg_lambda = 0.0
        # pbar = tqdm.trange(epochs)
        pbar = tqdm.trange(epochs)
        for epoch in pbar:
            if best_loss > 1 or flatness > 1:
                if epoch > lock_wavelengths:
                    self.pump_wavelengths.requires_grad = True
                pump_wavelengths = self.unscale(
                    self.pump_wavelengths, *self.wavelength_scaling
                )
                # TODO add the power dependency on the modes? No, we do not tune the modes 
                # TODO adapt this whole method to MMF
                # print("pump_powers: ", self.pump_powers)
                signal_spectrum = self.forward(
                    pump_wavelengths, self.pump_powers) + reg_lambda * torch.sum(dBm2watt(self.pump_powers[4:])*1e3)
                
                loss = loss_function(signal_spectrum, _target_spectrum)
                loss.backward()
                torch_optimizer.step()
                torch_optimizer.zero_grad()

                with torch.no_grad():
                    flatness = (
                        torch.max(signal_spectrum) - torch.min(signal_spectrum)
                    ).item()
                pbar.set_description(
                    f"Loss: {loss.item():.4f}"
                    + f"\tBest Loss: {best_loss:.4f}"
                    + f"\tFlatness: {flatness:.2f} dB"
                )
                # print(f"\nWavel. : {pump_wavelengths}"+f"\nPow. : {self.pump_powers}")
                # pbar.set_description(
                #     f"Loss: {loss.item():.4f}"
                #     + f"\tBest Loss: {best_loss:.4f}"
                #     + f"\tFlatness: {flatness:.2f} dB"
                # )

                if loss.item() < best_loss:
                    pump_wavelengths = self.unscale(
                        self.pump_wavelengths, *self.wavelength_scaling
                    )
                    best_wavelengths = torch.clone(pump_wavelengths)
                    best_powers = torch.clone(self.pump_powers)
                    best_loss = loss.item()

        # print(f"\nFlatness: {flatness:.2f} dB")
        return (
            best_wavelengths.detach().numpy().squeeze(),
            torch.abs(best_powers).detach().numpy().squeeze(),
        )
