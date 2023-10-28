import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit
def truncated_normal_single(mean, std_dev):
    sample = -1
    while sample < 0 or sample > 1:
        sample = np.random.normal(mean, std_dev)
    return sample


@njit
def truncated_normal(mean, std_dev, size):
    samples = np.zeros(size, dtype=np.float32)
    for i in range(size):
        samples[i] = truncated_normal_single(mean, std_dev)
    return samples


@njit
def simulate(sigma, log=True):
    currents_arrays = []
    for iwalk in range(runsNumber):
        if log:
            print(f"Run number: {iwalk}/{runsNumber}\r")
        # Clearing arrays from the last run ////
        system = np.zeros(Lx * Ly, dtype=np.float32)
        # Filling the lattice with particles alternatively
        system[::2] = 1
        # Give each particle a random speed (between 0 and 1)
        # from a uniform distribution
        speeds = truncated_normal(0.5, sigma, N)
        system[system == 1] = speeds
        system = system.reshape((Lx, Ly))
        currents = []
        # Beginning of a single run ////
        for steps_done in range(1, totalMCS):
            total_forward = 0
            for move_attempt in range(N):
                dice = np.random.randint(Lx * Ly)  # Picks the random spin in the array
                x = dice // Ly
                y = dice - x * Ly
                while system[x, y] == 0:
                    dice = np.random.randint(Lx * Ly)
                    x = dice // Ly
                    y = dice - x * Ly
                # Simple implementation of Periodic boundary conditions
                x_prev = Lx - 1 if x == 0 else x - 1
                x_next = 0 if x == Lx - 1 else x + 1
                y_next = 0 if y == Ly - 1 else y + 1
                # Simulating exchange dynamics
                dice = np.random.randint(4)
                if dice == 0:  # hop forward
                    # check if there is a particle in front
                    if system[x, y_next] == 0:
                        # throw second dice from truncated normal that has to be less than the speed
                        dice2 = np.random.random()
                        if dice2 < system[x, y]:
                            # if system[x,y] < 0.1:
                            # print("slow particle moved")
                            system[x, y], system[x, y_next] = system[x, y_next], system[x, y]
                            total_forward += 1
                elif dice == 2:  # hop up
                    # check if there is a particle above
                    if system[x_prev, y] == 0:
                        system[x, y], system[x_prev, y] = system[x_prev, y], system[x, y]
                elif dice == 3:  # hop down
                    # check if there is a particle below
                    if system[x_next, y] == 0:
                        system[x, y], system[x_next, y] = system[x_next, y], system[x, y]
                if move_attempt % current_averaging_time == 0 and move_attempt > 0:
                    currents.append(total_forward / current_averaging_time)
                    total_forward = 0

        currents_arrays.append(currents)
    return currents_arrays


@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@njit
def simulate_sigma_vs_steady_state_current(sigmas):
    currents = []
    i = 1
    for sigma in sigmas:
        print("Sigma", i)
        i += 1
        currents_arrays = simulate(sigma, log=False)
        currents_single = np_mean(np.array(currents_arrays), axis=0)
        # print(len(currents_single))
        currents.append(np.mean(currents_single[-int(len(currents_single)/3):]))
    return currents


def evaluate():
    currents = np.load("data/sigma_vs_current_128x32.npy")
    sigmas = np.load("data/sigma_vs_current_sigmas_128x32.npy")
    print(len(currents))
    plt.cla()
    plt.plot(sigmas, currents)
    plt.xlabel("Sigma (log scale)")
    plt.ylabel(f"Steady state current")
    plt.title(f"Average current over {runsNumber} runs")
    plt.xscale("log")
    plt.savefig(f"plots/different_speeds/steady_state_current_log_128x32.png")
    plt.cla()
    plt.xlabel("Sigma")
    plt.xscale("linear")
    plt.ylabel(f"Steady state current")
    plt.title(f"Average current over {runsNumber} runs")
    plt.plot(sigmas[70:], currents[70:])
    plt.savefig(f"plots/different_speeds/steady_state_current_128x32.png")
    # take log(1/currents-1/0.0585)
    plt.cla()
    plt.plot(sigmas, np.log(1 / currents - 1 / 0.0585))
    plt.xlabel("Sigma")
    plt.ylabel(f"ln(1/I - 1/I_0)")
    plt.title(f"Average current over {runsNumber} runs")
    plt.savefig(f"plots/different_speeds//ln_fixed_sigma_128x32.png")


def calc_sigma_vs_current():
    plt.cla()
    sigmas = np.logspace(-4, 1, 150, dtype=np.float32)
    print(sigmas)
    currents = simulate_sigma_vs_steady_state_current(sigmas)
    # save data
    np.save("data/sigma_vs_current_128x32.npy", currents)
    np.save("data/sigma_vs_current_sigmas_128x32.npy", sigmas)


def calc_indivual_sigmas():
    for sigma in [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 5]:
        currents_arrays = simulate(sigma)
        times = np.arange(len(currents_arrays[0]))
        # average over all runs
        currents = np.mean(currents_arrays, axis=0)
        # plot
        plt.plot(times[10:], currents[10:], label=f"sigma = {sigma}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(f"Current (last {current_averaging_time} moves averaged)")
    plt.title(f"Average current over {runsNumber} runs")
    plt.savefig(f"plots/different_speeds/individual_sigmas/currents_fixed_sigma_128x32.png")


if __name__ == '__main__':
    totalMCS = 70  # Total number of Monte Carlo steps per single run
    runsNumber = 200  # Number of runs to average over
    Lx = 32  # Number of rows , width
    Ly = 128  # Number of columns , length
    N = Lx * Ly // 2
    size = Lx * Ly
    current_averaging_time = int(N / 2) - 1  # Number of MCS to average over

    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 11

    calc_indivual_sigmas()
    calc_sigma_vs_current()
    evaluate()






