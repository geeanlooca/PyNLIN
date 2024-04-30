import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)

oi_file='oi.mat'
mat = scipy.io.loadmat(oi_file)
oi_full = mat['OI']
wl = mat['wavelenght_array'][0]
# average over the polarizations
oi = np.ndarray((21, 21, 4, 4))

# return starting and ending index of the polarization
def polix(f):
  ll = [2, 4, 4, 2]
  st = [0, 2, 6, 10]
  return (st[f], st[f]+ll[f])

for i in range(4):
  for j in range(4):
    oi[:, :, i, j] = np.average(oi_full[:, :, polix(i)[0]:polix(i)[1], polix(j)[0]:polix(j)[1]], axis=(2, 3))
np.save('oi.npy', oi)

plot=True
if plot:
  modes = ["01", "11", "21","02"]
  for i in range(len(oi[0, 0, :, 0])):
    plt.plot(wl, oi[0, :, i, :], label=modes) 
    plt.legend()
    plt.title(str(modes[i])+" @ 1400nm")
    plt.show()
    plt.savefig('media/OI_'+str(i)+'_pump-like.pdf')
  
  for i in range(len(oi[0, 0, :, 0])):
    plt.plot(wl, oi[-1, :, i, :], label=modes) 
    plt.legend()
    plt.title(str(modes[i])+" @ 1600nm")
    plt.show()
    plt.savefig('media/OI_'+str(i)+'sign-like.pdf')
