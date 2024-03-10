import os

import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science'])
plt.rcParams['font.size'] = 11
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (4.5, 3.2)

sigmas = np.load("data/paper_comparison/sigmas_simple.npy")
current_means_simple = np.load("data/paper_comparison/mean_currents_simple.npy")
current_means_gradient = np.load("data/paper_comparison/mean_currents_gradient.npy")
sigmas_classical = np.load("../TASEP Classical (no RL)/data/fixed/different_speeds/sigma_vs_current_sigmas_128x32.npy")[100:]
current_means_classical = np.load("../TASEP Classical (no RL)/data/fixed/different_speeds/sigma_vs_current_128x32.npy")[100:]

plt.plot(sigmas_classical, current_means_classical, label="Classical")
plt.plot(sigmas, current_means_simple, label="Simple Policy")
plt.plot(sigmas, current_means_gradient, label="Speed Gradient Policy")
plt.xscale("log")
plt.xlabel("Sigma")
plt.ylabel("Steady state current")
plt.legend()
os.makedirs("plots/paper_comparison", exist_ok=True)
plt.savefig("plots/paper_comparison/plot.pdf")
