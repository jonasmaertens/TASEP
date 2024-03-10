import multiprocessing as mp
import matplotlib.pyplot as plt
from smarttasep import Trainer
import numpy as np
from functools import partial


def run_trainer(sigma, model_id, runsNumber, len_currs, steps):
    currents = np.zeros(len_currs)
    for i in range(runsNumber):
        trainer = Trainer.load(model_id, total_steps=steps, sigma=sigma, do_plot=False, progress_bar=False,
                               do_render=False)
        trainer.env.unwrapped.prepare_lane_gradient = True
        trainer.env.unwrapped.inflate_speeds = True
        trainer.run(reset_stats=True)
        currents += np.array(trainer.currents)
        print(
            f"Run {i + 1}/{runsNumber} for sigma = {sigma:.2e}, mean current = {np.mean(trainer.currents[-int(len_currs * 3 / 4):]):.2f}")
    currents /= runsNumber
    np.save(f"data/paper_comparison/currents_gradient_{sigma:.2e}.npy", currents)
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
    sigmas = np.logspace(-1.5, 1.5, 30, dtype=np.float32)
    # sigmas2 = np.logspace(-3.5, -1, 10, dtype=np.float32)
    runsNumber = 5
    steps = 250000
    print([f"{sigma:.2e}" for sigma in sigmas])
    # plt.rcParams["figure.figsize"] = (8, 4)
    # plt.rcParams["figure.dpi"] = 300
    # plt.rcParams["font.size"] = 7
    model_id = 7  # Trainer.choose_model()
    trainer = Trainer.load(model_id, total_steps=steps, do_plot=False, do_render=False)
    trainer.env.unwrapped.prepare_lane_gradient = True
    trainer.env.unwrapped.inflate_speeds = True
    trainer.env.unwrapped.sigma = 10
    trainer.run(reset_stats=True)
    len_currs = len(trainer.currents)
    print(f"len_currs = {len_currs}")
    run_trainer_partial = partial(run_trainer, model_id=model_id, runsNumber=runsNumber, len_currs=len_currs,
                                  steps=steps)
    pool = mp.Pool(processes=10)  # Change the number of processes as needed
    current_means = pool.map(run_trainer_partial, sigmas)
    np.save("data/paper_comparison/mean_currents_gradient.npy", current_means)
    np.save("data/paper_comparison/sigmas_gradient.npy", sigmas)

    plt.plot(sigmas, current_means)
    plt.show()
