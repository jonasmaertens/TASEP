import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import os


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
        # Clearing arrays from the last run
        system = np.zeros((Lx, Ly), dtype=np.float32)
        # Filling the lattice as checkerboard
        system[::2, ::2] = 1
        system[1::2, 1::2] = 1
        # Flatten the system in order to be able to assign speeds without
        # multidimensional indexing (which is not supported by numba)
        system = system.reshape((Lx * Ly))
        # Give each particle a random speed (between 0 and 1)
        # from a uniform distribution
        speeds = truncated_normal(0.5, sigma, N)
        system[system == 1] = speeds
        # Reshape the system back to 2D
        system = system.reshape((Lx, Ly))
        currents = []
        # Beginning of a single run
        total_forward = 0
        for move_attempt in range(N * totalMCS):
            # print("\n\n")
            # print(system)
            # print(move_attempt)
            if move_attempt % current_averaging_time == 0 and move_attempt > 0:
                currents.append(total_forward / current_averaging_time)
                total_forward = 0
            dice = np.random.randint(Lx * Ly)  # Picks the random spin in the array
            x = dice // Ly
            y = dice - x * Ly
            while system[x, y] == 0:
                dice = np.random.randint(Lx * Ly)
                x = dice // Ly
                y = dice - x * Ly
            # print(f"Chosen particle: {x}, {y}")
            # throw second dice from truncated normal that has to be less than the speed
            dice2 = np.random.random()
            if dice2 > system[x, y]:
                # print("no move")
                continue
            # Simple implementation of Periodic boundary conditions
            y_next = 0 if y == Ly - 1 else y + 1
            # Simulating exchange dynamics
            # hop forward with probability 1
            # check if there is a particle in front
            if system[x, y_next] == 0:
                system[x, y], system[x, y_next] = system[x, y_next], system[x, y]
                total_forward += 1
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
        currents.append(np.mean(currents_single[-int(len(currents_single) / 3):]))
    return currents


def evaluate():
    currents = np.load(f"data/maxcurr/sigma_vs_current_{Ly}x{Lx}.npy")
    sigmas = np.load(f"data/maxcurr/sigma_vs_current_sigmas_{Ly}x{Lx}.npy")
    print(len(currents))
    plt.cla()
    plt.plot(sigmas, currents)
    plt.xlabel("Sigma (log scale)")
    plt.ylabel(f"Steady state current")
    plt.title(f"Average current over {runsNumber} runs ({Ly}x{Lx})")
    plt.xscale("log")
    plt.savefig(f"plots/maxcurr/different_speeds/steady_state_current_log_{Ly}x{Lx}.png")


def calc_sigma_vs_current():
    plt.cla()
    sigmas = np.logspace(-4, 1, 150, dtype=np.float32)
    print(sigmas)
    currents = simulate_sigma_vs_steady_state_current(sigmas)
    # save data
    np.save(f"data/maxcurr/sigma_vs_current_{Ly}x{Lx}.npy", currents)
    np.save(f"data/maxcurr/sigma_vs_current_sigmas_{Ly}x{Lx}.npy", sigmas)


if __name__ == '__main__':
    totalMCS = 70  # Total number of Monte Carlo steps per single run
    runsNumber = 500  # Number of runs to average over
    Lx = 32  # Number of rows , width
    Ly = 128  # Number of columns , length
    N = Lx * Ly // 2
    size = Lx * Ly
    current_averaging_time = int(N / 2) - 1  # Number of MCS to average over

    plt.rcParams["figure.figsize"] = (4, 2)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 6

    # create all directories if they don't exist
    os.makedirs("plots/maxcurr/different_speeds/individual_sigmas", exist_ok=True)
    os.makedirs("data/maxcurr", exist_ok=True)

    calc_sigma_vs_current()
    evaluate()

