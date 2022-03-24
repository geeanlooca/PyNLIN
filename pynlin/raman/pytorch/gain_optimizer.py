from pynlin.raman.pytorch.solvers import RamanAmplifier
import torch
import tqdm
from torch import nn
from torch.optim.adam import Adam
from torch.nn import MSELoss


def dBm(x):
    return 10 * torch.log10(x) + 30


class CopropagatingOptimizer(nn.Module):
    def __init__(
        self,
        raman_torch_solver: RamanAmplifier,
        initial_pump_wavelengths,
        initial_pump_powers,
    ):
        super(CopropagatingOptimizer, self).__init__()
        self.raman_solver = raman_torch_solver
        self.pump_powers = nn.Parameter(initial_pump_powers)
        self.pump_wavelengths = nn.Parameter(initial_pump_wavelengths)

    def forward(self, x):
        return self.raman_solver(x).float()

    def optimize(self, epochs=100, learning_rate=1e-1):
        self.input_power = dBm(
            self.raman_solver.signal_power
            * torch.ones_like(self.raman_solver.signal_wavelengths).view(1, -1)
        ).float()

        torch_optimizer = Adam(self.parameters(), lr=0.1)
        best_loss = torch.inf
        best_wavelengths = torch.clone(self.pump_wavelengths)
        best_powers = torch.clone(self.pump_powers)

        pbar = tqdm.tqdm(range(epochs))
        loss_function = MSELoss()
        for epoch in pbar:
            x = (
                torch.cat((self.pump_wavelengths * 1e-9, self.pump_powers))
                .view(1, -1)
                .float()
            )
            signal_spectrum = dBm(self.forward(x))
            loss = loss_function(signal_spectrum, self.input_power)
            loss.backward()
            torch_optimizer.step()
            torch_optimizer.zero_grad()

            pbar.set_description(f"Loss: {loss.item():.4f}")

            if loss.item() < best_loss:
                best_wavelengths = torch.clone(self.pump_wavelengths)
                best_powers = torch.clone(self.pump_powers)
                best_loss = loss.item()

            print(self.pump_wavelengths)
            print(self.pump_powers)

        return (
            self.input_power.detach().numpy().squeeze(),
            signal_spectrum.detach().numpy().squeeze(),
            best_wavelengths.detach().numpy().squeeze() * 1e-9,
            torch.abs(best_powers).detach().numpy().squeeze(),
        )
