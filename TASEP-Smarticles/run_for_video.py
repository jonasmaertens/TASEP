# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sigma = 5
    average_window = 3000
    plt.rcParams["figure.figsize"] = (4, 1.6)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 4
    plt.style.use("dark_background")
    # hide controls
    plt.rcParams["toolbar"] = "None"
    plt.title(f"Smart TASEP for sigma = {sigma}")
    plt.xlabel("Time")
    plt.ylabel(f"Current (last {average_window} moves averaged)")

    trainer = Trainer.load(do_plot=True, render_start=0, wait_initial=True, total_steps=400000, sigma=sigma,
                           average_window=average_window, window_height=300, moves_per_timestep=400)
    trainer.run()
