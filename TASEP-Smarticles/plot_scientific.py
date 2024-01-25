import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science'])
plt.rcParams['font.size'] = 11
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (4.5, 3.2)


# Load data
name = "lanes_1"
timesteps = np.load(f"data/{name}/timesteps.npy")
currents = np.load(f"data/{name}/currents.npy")

# smooth currents
currents = np.convolve(currents, np.ones(20)/20, mode='valid')
timesteps = timesteps[9:-10]

plot_interval = 5000
plt.xlabel("Time steps")
plt.ylabel(f"Current")
plt.plot(timesteps, currents, label=f"Data (100,000 steps moving avg.)")
plt.plot(timesteps, np.ones(len(timesteps)) * np.mean(currents),
         label=f"Average")
plt.legend()
plt.savefig(f"plots/scientific/{name}.pdf")
# plt.show()
