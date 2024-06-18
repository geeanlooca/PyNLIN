import scipy.io
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.graph_objects as go
from pynlin.utils import oi_law_fit, oi_law
from matplotlib.cm import viridis
rc('text', usetex=False)

beta_file = 'fitBeta.mat'
mat = scipy.io.loadmat(beta_file)

print(mat)

## Load the WDM grid and fiber characteristic and compute the number of collisions

