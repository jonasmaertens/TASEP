import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science'])
plt.rcParams['font.size'] = 11
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (4.5, 3.2)


# Load data
name = "uniform_speeds"
timesteps = np.load(f"data/{name}/timesteps.npy")
currents = np.load(f"data/{name}/currents.npy")

plot_interval = 2000
plt.xlabel("Time steps")
plt.ylabel(f"Current")
plt.plot(timesteps, currents, label=f"Data (last {plot_interval} steps averaged)")
plt.plot(timesteps, np.ones(len(timesteps)) * np.mean(currents[125:]),
         label=f"Average (first {plot_interval * 125} steps excluded)")
plt.legend()
plt.savefig(f"plots/scientific/{name}.pdf")