import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass

@dataclass
class Material:
    A: ArrayLike
    B: ArrayLike


class NBK7(Material):
    def __init__(self):
        """NBK7 glass material.

        Reference: https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
        """
        self.A = np.array([1.03961212, 0.231792344, 1.01046945])
        self.B = np.array([0.00600069867, 0.0200179144, 103.560653])



def sellmeier(wavelength: ArrayLike | float, A: ArrayLike, B: ArrayLike) -> ArrayLike | float:
    """Compute the refractive index of a material using the Sellmeier equation."""

    n2 = 1
    for b,c in zip(A,B):
        n2 += b *wavelength**2 / (wavelength **2 - c)

    return np.sqrt(n2)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage of the sellmeier function
    wavelength = np.linspace(0.4, 2.5, 1000)
    A = np.array([0.6961663, 0.4079426, 0.8974794])
    B = np.array([0.0684043**2, 0.1162414**2, 9.896161**2])

    nbk7 = NBK7()

    n = sellmeier(wavelength, nbk7.A, nbk7.B)
    n_1 = sellmeier(1.5, nbk7.A, nbk7.B)
    assert np.isclose(n_1, 1.5013, atol=1e-4), f"Sellmeier equation is not working properly: {n_1} != 1.5013"

    plt.figure()
    plt.plot(wavelength, n)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Refractive index")
    plt.ylim(0, 1.75)
    plt.show()
