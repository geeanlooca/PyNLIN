import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
npg = np.load("np_gains.npy")
tcg = np.load("tc_gains.npy")

plt.clf()
sns.heatmap(npg[:, None], cmap="coolwarm",  square=True)
# plt.show()
plt.savefig("media/gain/npy_gain.pdf")
plt.clf()
sns.heatmap(npg[:, None], cmap="coolwarm", square=True)
# plt.show()
plt.savefig("media/gain/tch_gain.pdf")
plt.clf()
sns.heatmap((npg[:, None]-tcg[:, None])/((npg[:, None]+tcg[:, None])/2), cmap="coolwarm", square=True)
# plt.show()
plt.savefig("media/gain/diff_gain.pdf")