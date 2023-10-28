# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
import glob
import os

import matplotlib.pyplot as plt

from Trainer import Trainer, Hyperparams, EnvParams
import numpy as np


def train_model():
    for sigma in sigmas:
        print(f"sigma = {sigma:.2e}")
        env_params = EnvParams(render_mode=None,
                               length=128,
                               width=32,
                               moves_per_timestep=20,
                               window_height=256,
                               observation_distance=2,
                               initial_state_template="checkerboard",
                               distinguishable_particles=True,
                               use_speeds=True,
                               sigma=sigma,
                               average_window=3500)
        hyperparams = Hyperparams(BATCH_SIZE=512,
                                  GAMMA=0.9,
                                  EPS_START=0.9,
                                  EPS_END=0.01,
                                  EPS_DECAY=40000,
                                  TAU=0.0001,
                                  LR=0.005,
                                  MEMORY_SIZE=100000)

        trainer = Trainer(env_params, hyperparams, reset_interval=40000,
                          total_steps=155_000, do_plot=False, plot_interval=3500)

        trainer.train()

        trainer.save_plot(f"plots/different_speeds/individual_sigmas/long/long_sigma_{sigma:.2e}.png")

        trainer.save(f"models/different_speeds/individual_sigmas/long/model_long_151_000_steps_sigma_{sigma:.2e}.pt")


def evaluate():
    # check if we find a model for each sigma
    # all_files = os.listdir("models/different_speeds/individual_sigmas")
    # current_means = []
    # for sigma in sigmas:
    #     possible_files = glob.glob(
    #         f"models/different_speeds/individual_sigmas/model_100000_steps_sigma_{sigma:.2e}*.pt")
    #     if len(possible_files) != 1:
    #         print(f"Could not find model for sigma = {sigma}")
    #     file = possible_files[0]
    #     print(f"Found model for sigma = {sigma:2e}")
    #     env_params = EnvParams(render_mode=None,
    #                            length=128,
    #                            width=32,
    #                            moves_per_timestep=300,
    #                            window_height=256,
    #                            observation_distance=2,
    #                            initial_state_template="checkerboard",
    #                            distinguishable_particles=True,
    #                            use_speeds=True,
    #                            sigma=sigma,
    #                            average_window=3000)
    #     currents = np.zeros(49)
    #     for i in range(10):
    #         trainer = Trainer(env_params, model=file, total_steps=150_000, do_plot=False, plot_interval=3000)
    #         trainer.run()
    #         currents += np.array(trainer.currents)
    #     currents /= 10
    #     ################# Fix this plot
    #     np.save(f"data/different_speeds/individual_sigmas/currents_sigma_{sigma:.2e}.npy", currents)
    #     plt.plot(np.arange(49), currents, label=f"sigma = {sigma:.2e}")
    #     current_means.append(np.mean(currents[-30:]))
    # np.save("data/different_speeds/current_means.npy", current_means)
    for sigma in sigmas[12:-3]:
        currents = np.load(f"data/different_speeds/individual_sigmas/currents_sigma_{sigma:.2e}.npy")
        plt.plot(np.arange(49), currents * 3 / 10, label=f"sigma = {sigma:.2e}")
    current_means = np.load("data/different_speeds/current_means.npy")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(f"Current (last 2000 moves averaged)")
    plt.title(f"Average current over 10 runs")
    plt.savefig(f"plots/different_speeds/individual_sigmas/currents_fixed_sigma.png")
    plt.show()
    plt.cla()
    plt.plot(sigmas, current_means)
    plt.xlabel("Sigma")
    plt.ylabel(f"Steady state current")
    plt.title(f"Average current over 10 runs")
    plt.xscale("log")
    plt.savefig(f"plots/different_speeds/steady_state_current_log.png")


if __name__ == '__main__':
    sigmas = np.logspace(-3.3, 1.3, 60, dtype=np.float32)
    print([f"{sigma:.2e}" for sigma in sigmas])
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 7
    train_model()



