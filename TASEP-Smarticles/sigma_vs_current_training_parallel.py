# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
import glob
import os
import multiprocessing as mp

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


def run_trainer(sigma):
    possible_files = glob.glob(
        f"models/different_speeds/individual_sigmas/model_100000_steps_sigma_1.00e+01*.pt")
    if len(possible_files) != 1:
        print(f"Could not find model for sigma = {sigma}")
    file = possible_files[0]
    # print(f"Found model for sigma = {sigma:2e}")
    env_params = EnvParams(render_mode=None,
                           length=128,
                           width=32,
                           moves_per_timestep=300,
                           window_height=256,
                           observation_distance=2,
                           initial_state_template="checkerboard",
                           distinguishable_particles=True,
                           use_speeds=True,
                           sigma=sigma,
                           average_window=5000)
    currents = np.zeros(40)
    for i in range(20):
        print(f"Run {i+1}/20 for sigma = {sigma:.2e}")
        trainer = Trainer(env_params, model=file, total_steps=201_000, do_plot=False, plot_interval=5000, progress_bar=False)
        trainer.run()
        currents += np.array(trainer.currents)
    currents /= 20
    np.save(f"data/different_speeds/individual_sigmas/currents_onlysigma10_sigma_{sigma:.2e}.npy", currents)
    return np.mean(currents[-31:])


if __name__ == '__main__':
    sigmas = np.logspace(-1.25, 1.3, 55, dtype=np.float32)
    print([f"{sigma:.2e}" for sigma in sigmas])
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 7
    pool = mp.Pool(processes=10)  # Change the number of processes as needed
    current_means = pool.map(run_trainer, sigmas)
    np.save("data/different_speeds/current_means_onlysigma10.npy", current_means)



