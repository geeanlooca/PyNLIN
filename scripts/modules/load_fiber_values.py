import numpy as np
import pynlin.utils
import scipy
from scipy.optimize import curve_fit
from pynlin.utils import *


def convert_coefficients(p1, p2, p3):
    beta_file = './results/fitBeta.mat'
    std = scipy.io.loadmat(beta_file)['omega_std'][0][0]
    avg = scipy.io.loadmat(beta_file)['omega_mean'][0][0]
    p1_new = 3 * p1 / (std**3)
    p2_new = 2 * p2 / (std**2) - 6 * p1 * avg / (std**3)
    p3_new = 3 * (avg**2) / (std**3) * p1 - 2 * p2 * avg / (std**2) + p3 / std
    return [p1_new * (2 * np.pi)**2, p2_new * 2 * np.pi, p3_new]


def load_group_delay() -> np.array:
    beta_file = './results/fitBeta.mat'
    mat = scipy.io.loadmat(beta_file)['fitParams'] * 1.0
    for i in range(4):
      # print(mat[i, 0], mat[i, 1], mat[i, 2])
      # print(convert_coefficients(mat[i, 0], mat[i, 1], mat[i, 2]))
      mat[i, :] = convert_coefficients(mat[i, 0], mat[i, 1], mat[i, 2])
    return mat


def load_gvd() -> np.array:
    beta_file = './results/fitBeta.mat'
    mat = scipy.io.loadmat(beta_file)['fitParams'] * 1.0

    for i in range(4):
        mat[:, i] = convert_coefficients
    return mat

def load_oi() -> np.array:
    oi_file = 'oi.mat'
    mat = scipy.io.loadmat(oi_file)
    oi_full = mat['OI'] * 1e12
    wl = mat['wavelenght_array'][0] * 1e-9
    # average over the polarizations
    oi = np.ndarray((21, 21, 4, 4))
    oi_avg = np.ndarray((4, 4))
    oi_max = np.zeros_like(oi_avg)
    oi_min = np.zeros_like(oi_avg)

    def polix(f):
        ll = [2, 4, 4, 2]
        st = [0, 2, 6, 10]
        return (st[f], st[f] + ll[f])

    for i in range(4):
        for j in range(4):
            oi[:, :, i, j] = np.mean(oi_full[:, :, polix(i)[0]:polix(i)[
                1], polix(j)[0]:polix(j)[1]], axis=(2, 3))
            oi_avg[i, j] = np.mean(oi[:, :, i, j])
            oi_max[i, j] = np.max(oi[:, :, i, j])
            oi_min[i, j] = np.min(oi[:, :, i, j])

    np.save('oi.npy', oi)
    np.save('oi_avg.npy', oi_avg)
    np.save('oi_max.npy', oi_max)
    np.save('oi_min.npy', oi_min)

    # quadratic fit of the OI in frequency
    oi_fit = np.ndarray((6, 4, 4))

    x, y = np.meshgrid(wl, wl)
    for i in range(4):
        for j in range(4):
            oi_fit[:, i, j] = curve_fit(
                oi_law_fit, (x, y), oi[:, :, i, j].ravel(), p0=[1e10, 1e10, 1e10, 1e10, 0, 1])[0].T
    np.save('oi_fit.npy', oi_fit)
    return oi_fit
