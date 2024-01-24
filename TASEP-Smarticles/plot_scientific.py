import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science'])
plt.rcParams['font.size'] = 11
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (4.5, 3.2)


# Load data
name = "speed_grad_4"
timesteps = np.load(f"data/{name}/timesteps.npy")
currents = np.load(f"data/{name}/currents.npy")

# smooth currents
currents = np.convolve(currents, np.ones(10)/10, mode='valid')
timesteps = timesteps[4:-5]

plot_interval = 5000
plt.xlabel("Time steps")
plt.ylabel(f"Current")
plt.plot(timesteps, currents, label=f"Data (moving average of 50,000 steps)")
plt.plot(timesteps, np.ones(len(timesteps)) * np.mean(currents[-300:]),
         label=f"Average over last 1,500,000 steps")
plt.legend()
plt.savefig(f"plots/scientific/{name}.pdf")
plt.show()