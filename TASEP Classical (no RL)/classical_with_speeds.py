import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.special import erf
import scienceplots

plt.style.use(['science'])
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (4.5, 3.2)  # (3,4) for truncnorm


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
    print(always_forward)
    for iwalk in range(runsNumber):
        if log:
            print(f"Run number: {iwalk}/{runsNumber}\r")
        # Clearing arrays from the last run
        system = np.zeros((Lx, Ly), dtype=np.float32)
        # Filling the lattice as checkerboard
        system[::2, ::2] = 1
        system[1::2, 1::2] = 1
        # print density
        # print(np.sum(system) / Lx / Ly)
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
        for move_attempt in range(totalSteps):
            # print("\n\n")
            # print(system)
            # print(move_attempt)
            if move_attempt % current_averaging_time == 0 and move_attempt > 0:
                currents.append(total_forward / current_averaging_time)
                total_forward = 0
            dice = np.random.randint(Lx * Ly)  # Picks the random spin in the array
            x = dice // Ly
            y = dice - x * Ly
            if system[x, y] == 0:
                continue
            # print(f"Chosen particle: {x}, {y}")
            # throw second dice from truncated normal that has to be less than the speed
            dice2 = np.random.random()
            if dice2 > system[x, y]:
                # print("no move")
                continue
            # Simple implementation of Periodic boundary conditions
            x_prev = Lx - 1 if x == 0 else x - 1
            x_next = 0 if x == Lx - 1 else x + 1
            y_next = 0 if y == Ly - 1 else y + 1
            # Simulating exchange dynamics
            dice = np.random.randint(4) if not always_forward else 0
            if dice <= 1:  # hop forward with probability 1/2
                # check if there is a particle in front
                if system[x, y_next] == 0:
                    system[x, y], system[x, y_next] = system[x, y_next], system[x, y]
                    total_forward += 1
                    # print("forward")
            elif dice == 2:  # hop up
                # check if there is a particle above
                if system[x_prev, y] == 0:
                    system[x, y], system[x_prev, y] = system[x_prev, y], system[x, y]
                    # print("hop up")
            elif dice == 3:  # hop down
                # check if there is a particle below
                if system[x_next, y] == 0:
                    system[x, y], system[x_next, y] = system[x_next, y], system[x, y]
                    # print("hop down")
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
        currents.append(np.mean(currents_single[-int(len(currents_single) / 4):]))
    return currents


def evaluate():
    plt.cla()
    currents = np.load(f"data/{path}sigma_vs_current_{Ly}x{Lx}.npy")
    current_always_forward = np.load(f"data/{path}/always_forward/sigma_vs_current_{Ly}x{Lx}.npy")
    # current_128x6 = np.load(f"data/{path}sigma_vs_current_128x6.npy")
    # current_128x3 = np.load(f"data/{path}sigma_vs_current_128x3.npy")
    # current_128x4 = np.load(f"data/{path}sigma_vs_current_128x4.npy")
    # current_128x5 = np.load(f"data/{path}sigma_vs_current_128x5.npy")
    # current_128x2 = np.load(f"data/{path}sigma_vs_current_128x2.npy")
    # current_128x1 = np.load(f"data/{path}sigma_vs_current_128x1.npy")
    sigmas = np.load(f"data/{path}sigma_vs_current_sigmas_{Ly}x{Lx}.npy")
    print(len(currents))
    # plt.figure(figsize=(6, 4))
    # plt.plot(sigmas, currents, label="128x32")
    plt.plot(sigmas, currents, label="Random moves")
    plt.plot(sigmas, current_always_forward, label="Always forward")
    # plt.plot(sigmas, current_128x6, label="128x6")
    # plt.plot(sigmas, current_128x5, label="128x5")
    # plt.plot(sigmas, current_128x4, label="128x4")
    # plt.plot(sigmas, current_128x3, label="128x3")
    # plt.plot(sigmas, current_128x2, label="128x2")
    # plt.plot(sigmas, current_128x1, label="128x1")
    plt.legend()
    plt.xlabel(r"Standard deviation $\sigma$ (log scale)")
    plt.ylabel(f"Steady state current")
    # plt.title(f"Average current over {runsNumber} runs ({Ly}x{Lx})")
    plt.xscale("log")
    # plt.ylim(0.0485, 0.066)
    plt.savefig(f"plots/{path}steady_state_current_both_log.pdf")


def calc_sigma_vs_current():
    plt.cla()
    sigmas = np.logspace(-4, 1, 200, dtype=np.float32)
    print(sigmas)
    currents = simulate_sigma_vs_steady_state_current(sigmas)
    # save data
    np.save(f"data/{path}sigma_vs_current_{Ly}x{Lx}.npy", currents)
    np.save(f"data/{path}sigma_vs_current_sigmas_{Ly}x{Lx}.npy", sigmas)


def calc_indivual_sigmas():
    plt.cla()
    for sigma in [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 5]:
        print("Sigma", sigma)
        if os.path.exists(f"data/{path}currents_fixed_sigma_{Ly}x{Lx}_{sigma}.npy"):
            currents = np.load(f"data/{path}currents_fixed_sigma_{Ly}x{Lx}_{sigma}.npy")
        else:
            currents_arrays = simulate(sigma, log=False)
            # average over all runs
            currents = np_mean(np.array(currents_arrays), axis=0)
            print(len(currents))
            np.save(f"data/{path}currents_fixed_sigma_{Ly}x{Lx}_{sigma}.npy", currents)
        # moving average over 10 points
        currents = np.convolve(currents, np.ones((5000,)) / 5000, mode='valid')
        times = np.arange(len(currents)) + 5000
        # plot
        plt.plot(times, currents, label=r"$\sigma$=" + f"{sigma}")
    plt.legend(ncol=2)
    plt.ylim(0.038, 0.088)
    plt.xlabel("Timesteps")
    plt.ylabel(f"Current (moving avg.)")
    # plt.title(f"Average current over {runsNumber} runs")
    plt.savefig(f"plots/{path}currents_fixed_sigma_{Ly}x{Lx}.pdf", bbox_inches='tight')


def test_truncated_normal():
    # samples = truncated_normal(0.5, sigma, 1000000)
    # Plot the histogram with density=True
    # plt.hist(samples, bins=60, density=True, label=r"Normalized histogram of samples", opacity=0.5)
    # Generate the x values for the truncated normal distribution
    x = np.linspace(0, 1, 1000)
    for sigma in [5, 0.3, 0.1]:
        # Calculate the y values for the truncated normal distribution
        y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - 0.5) ** 2 / (2 * sigma ** 2))
        # norm of the truncated normal distribution
        norm = 1 / 2 * (erf((1 - 0.5) / (np.sqrt(2) * sigma)) + erf(0.5 / (np.sqrt(2) * sigma)))
        # normalize the y values
        y /= norm
        # Plot the truncated normal distribution as a PDF
        p = plt.plot(x, y, label=r"$\sigma$=" + str(sigma))
        # plot vertical lines that truncate the normal distribution
        plt.vlines(x=[0, 1], color=p[-1].get_color(), ymin=0, ymax=y[0])
    plt.legend()
    # plt.ylim(0, 4.5)
    # plt.xticks(np.arange(0, 1.01, 0.1))
    plt.xlabel("Speed")
    plt.ylabel("Probability density")
    plt.savefig(f"plots/{path}truncated_normal_small.pdf")


if __name__ == '__main__':
    totalSteps = 140 * 128 * 32
    totalMCS = 140  # Total number of Monte Carlo steps per single run
    runsNumber = 800  # Number of runs to average over
    Lx = 32  # Number of rows , width
    Ly = 128  # Number of columns , length
    N = Lx * Ly // 2
    size = Lx * Ly
    current_averaging_time = int(N / 2) - 1  # Number of MCS to average over
    always_forward = False
    path = "fixed/different_speeds/longer/"
    if always_forward:
        path += "always_forward/"
    os.makedirs(f"plots/{path}", exist_ok=True)
    os.makedirs(f"data/{path}", exist_ok=True)
    # current_averaging_time = 1
    # calc_indivual_sigmas()
    # calc_sigma_vs_current()
    evaluate()
    # test_truncated_normal()
