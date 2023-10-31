# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
import matplotlib.pyplot as plt
from Trainer import Trainer, EnvParams


if __name__ == '__main__':
    sigma = 50
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
    model = f"models/different_speeds/individual_sigmas/model_100000_steps_sigma_1.00e+01_20231028170511.pt"
    env_params = EnvParams(render_mode="human",
                           length=128,
                           width=32,
                           moves_per_timestep=400,
                           window_height=300,
                           observation_distance=2,
                           initial_state_template="checkerboard",
                           distinguishable_particles=True,
                           use_speeds=True,
                           sigma=sigma,
                           average_window=average_window)
    trainer = Trainer(env_params, model=model, total_steps=300_000, do_plot=True, plot_interval=average_window,
                      progress_bar=True, wait_initial=True)
    trainer.run()
