

# A workaround for setting the correct workind dir.
# beware to run only once per session!
# is there a better way? :)

from space_integrals_general import *
from time_integrals import do_time_integrals
import matplotlib.pyplot as plt

# Load the simulation parameters
with open("./scripts/sim_config.json") as f:
  data = json.load(f)

  # we use a setup of a low populated WDM grid: 
  # just a few (<10) channels in the beginning of the grid 
  num_channels = data["num_channels"] 
  length = data["fiber_length"][0]

  # write the results file in ../results/general_results.h5 with the correct time integrals
  # the file contains, for each interferent channel, the values (z, m, I) of the z
  # channel of interest is set to channel 0, and interferent channel index start from 0 for simplicity
  do_time_integrals(length, pulse_shape="Gaussian")
  compare_interferent(interfering_channels=[0, 1, 2, 3])