from scipy import polyval


class Fiber:
    """Object collecting parameters of an optical fiber."""

    def __init__(
        self,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta2=20 * 1e-24 / 1e3,
        gamma=1.3 * 1e-3,
    ):
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.beta2 = beta2
        self.gamma = gamma

        if losses:
            try:
                self.losses = list(losses)
            except:
                self.losses = [losses]
        else:
            self.losses = [2.26786883e-06, -7.12461042e-03, 5.78789219e00]

        self.raman_efficiency = self.raman_coefficient / self.effective_area

        super().__init__()

    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/km) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths * 1e9)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
