from space_integrals_general import get_space_integrals
from time_integrals import do_time_integrals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os.path

class ScalarFormatterClass1(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

class ScalarFormatterClass2(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.0f"

length_list = 1e3 * 100 * np.array([1, 10])
X_mid = np.zeros_like(length_list)
X_ana = np.zeros_like(length_list)
Omega = 0.0

for il, length in enumerate(length_list):
  do_time_integrals(length)
  fname = str(length)
  if os.path.isfile("X_"+fname+".npy"):
    X0mm = np.load()
  X0mm, X0mm_ana = get_space_integrals()
  allofthem = sum(X0mm**2)
  numbs = len(X0mm)
  X_mid[il] = allofthem
  X_ana[il] = sum(X0mm_ana**2)

plt.clf()
plt.figure(figsize=(8, 5))
plt.plot(length_list, np.abs(X_mid-X_ana)/X_mid)
ax = plt.gca()
yScalarFormatter = ScalarFormatterClass1(useMathText=True)
yScalarFormatter.set_powerlimits((0,0))
xScalarFormatter = ScalarFormatterClass2(useMathText=True)
xScalarFormatter.set_powerlimits((3,3))
ax.yaxis.set_major_formatter(yScalarFormatter)
ax.xaxis.set_major_formatter(xScalarFormatter)
plt.plot(length_list, X_ana*0.0, color="gray", ls="dashed")
plt.xlabel(r"$L$")
plt.ylabel(r"$\varepsilon_R$")
plt.tight_layout()
plt.savefig("media/convergence.pdf")