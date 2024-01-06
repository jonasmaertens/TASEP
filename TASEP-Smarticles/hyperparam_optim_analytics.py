import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt


def sort_and_plot_globs(globs, average=10):
    title = globs[0].split("/")[2]
    plt.title(title)
    labels = [float(file.split("/")[3]) for file in globs]
    sorted_globs = [x for _, x in sorted(zip(labels, globs))]
    for file in sorted_globs:
        label = float(file.split("/")[3])
        rewards = np.load(file)
        # plot with moving average of 10
        plt.plot(np.convolve(rewards, np.ones((average,)) / average, mode='valid'), label=f"{label}")

    plt.legend()
    plt.show()


# eps_decay
sort_and_plot_globs(glob.glob("data/hyperparam_optim/eps_decay/*/rewards.npy"), 10)

# batch_size
sort_and_plot_globs(glob.glob("data/hyperparam_optim/batch_size/*/rewards.npy"), 10)

# gamma
sort_and_plot_globs(glob.glob("data/hyperparam_optim/gamma/*/rewards.npy"), 10)

# lr
sort_and_plot_globs(glob.glob("data/hyperparam_optim/lr/*/rewards.npy"), 10)

