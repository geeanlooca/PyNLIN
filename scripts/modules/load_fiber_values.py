import numpy as np
import pynlin.utils
import scipy
from scipy.optimize import curve_fit
from pynlin.utils import *


def init_omega_norm():
    std = scipy.io.loadmat(beta_file)['omega_std']
    avg = scipy.io.loadmat(beta_file)['omega_mean']

    def omega_norm(freq) -> np.array:
        omega = 2 * np.pi * freq
        omega_n = (omega - avg) / std
        return omega_n

    return omega_norm

def load_group_delay() -> np.array:
    # s_limit = 1460e-9
    # l_limit = 1625e-9
    # s_freq = 3e8/s_limit
    # l_freq = 3e8/l_limit

    # print(s_freq*1e-12)
    # print(l_freq*1e-12)
    # delta = (s_freq - l_freq) *1e-12
    # print(delta)
    # avg = print((s_freq+l_freq) *1e-12 /2)
    beta_file = './results/fitBeta.mat'
    mat = scipy.io.loadmat(beta_file)['fitParams'] * 1.0

    # modes = [0, 1, 2, 3]

    # from the Matlab file the fit is:
    # 3 * fitresult.p1.*(omega_n).^2 + 2 * fitresult.p2.*omega_n +fitresult.p3)./std(omega)
    # omega_n is the centered rescaled vector
    # omega = 2 * np.pi * freqs
    # omega_norm = scipy.io.loadmat(beta_file)['omega_std']
    # omega_n = (omega - scipy.io.loadmat(beta_file)['omega_mean']) / omega_norm
    # beta1 = np.zeros((4, len(freqs)))

    # for i in range(4):
    #     beta1[i, :] = (3 * mat[i, 0] * (omega_n ** 2) + 2 *
    #                   mat[i, 1] * omega_n + mat[i, 2]) / omega_norm
    return mat


s_limit = 1460e-9
l_limit = 1625e-9
s_freq = 3e8 / s_limit
l_freq = 3e8 / l_limit

print(s_freq * 1e-12)
print(l_freq * 1e-12)
delta = (s_freq - l_freq) * 1e-12
print(delta)
avg = print((s_freq + l_freq) * 1e-12 / 2)
beta_file = './results/fitBeta.mat'
mat = scipy.io.loadmat(beta_file)['fitParams'] * 1.0

omega = 2 * np.pi * freqs
omega_norm = scipy.io.loadmat(beta_file)['omega_std']
omega_n = (omega - scipy.io.loadmat(beta_file)['omega_mean']) / omega_norm
beta1 = np.zeros((4, len(freqs)))

for i in range(4):
    beta1[i, :] = (3 * mat[i, 0] * (omega_n ** 2) + 2 *
                   mat[i, 1] * omega_n + mat[i, 2]) / omega_norm


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
    return
