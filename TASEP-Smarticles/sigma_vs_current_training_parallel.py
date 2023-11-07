# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
import multiprocessing as mp

import matplotlib.pyplot as plt

from Trainer import Trainer
import numpy as np
from functools import partial


def run_trainer(sigma, model_id, runsNumber, len_currs, steps):
    currents = np.zeros(len_currs)
    for i in range(runsNumber):
        print(f"Run {i + 1}/{runsNumber} for sigma = {sigma:.2e}")
        trainer = Trainer.load(model_id, total_steps=steps, sigma=sigma)
        trainer.run()
        currents += np.array(trainer.currents)
    currents /= runsNumber
    np.save(f"data/different_speeds/individual_sigmas/currents_fixed_2_sigma_{sigma:.2e}.npy", currents)
    return np.mean(currents[-int(len_currs * 3 / 4):])


def plot_results(runsNumber):
    sigmas = np.load("data/different_speeds/sigmas_tot.npy")
    current_means = np.load("data/different_speeds/current_means_fixed_tot.npy")
    plt.plot(sigmas[5:-12], current_means[5:-12])
    plt.xlabel("Sigma")
    plt.ylabel(f"Steady state current")
    plt.title(f"Average current over {runsNumber} runs of smart TASEP (128x32)")
    plt.xscale("log")
    plt.savefig(f"plots/different_speeds/steady_state_current_log_fixed.png")


if __name__ == '__main__':
    sigmas1 = np.logspace(-1.5, 1.5, 50, dtype=np.float32)
    sigmas2 = np.logspace(-3.5, -1, 10, dtype=np.float32)
    runsNumber = 10
    steps = 250000
    print([f"{sigma:.2e}" for sigma in sigmas2])
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 7
    # model_id = 1  # Trainer.choose_model()
    # trainer = Trainer.load(model_id, total_steps=steps)
    # trainer.run()
    # len_currs = len(trainer.currents)
    # print(f"len_currs = {len_currs}")
    # run_trainer_partial = partial(run_trainer, model_id=model_id, runsNumber=runsNumber, len_currs=len_currs,
    #                               steps=steps)
    # pool = mp.Pool(processes=10)  # Change the number of processes as needed
    # current_means = pool.map(run_trainer_partial, sigmas)
    #np.save("data/different_speeds/current_means_fixed_2.npy", current_means)

    plot_results(runsNumber)
