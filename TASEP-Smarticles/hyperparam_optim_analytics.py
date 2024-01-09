import numpy as np
import glob
import os
import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def sort_and_plot_globs(globs, ax, average=10, legend_inside=False, title=""):
    # title = globs[0].split("/")[2]
    # ax.set_title(title)
    num_colors = len(globs)
    ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])
    try:
        labels = [eval(file.split("/")[3]) for file in globs]
    except NameError:
        labels = [file.split("/")[3] for file in globs]
    sorted_globs = [(label, x) for label, x in sorted(zip(labels, globs))]
    for label, file in sorted_globs:
        rewards = np.load(file)
        # plot with moving average
        averaged_rewards = np.convolve(rewards, np.ones((8,)) / 8, mode='valid')
        timesteps = np.arange(len(averaged_rewards)) * 2500
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.plot(timesteps, averaged_rewards, label=f"{label}")
    ax.set_xlabel("Training steps")
    # set fixed y axis limits
    if legend_inside:
        ax.legend(title=title)
        ax.set_ylabel("Average reward")
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=title)
        # no y axis labels
        ax.set_yticklabels([])
    # get row and column index of subplot
    row, col = np.where(axs == ax)
    if col == 1:
        # remove y axis labels
        ax.set_yticklabels([])
    if row != 3:
        # remove x axis labels
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if row == 0:
        ax.set_ylim(0.12, 0.42)
    elif row == 1:
        ax.set_ylim(-0.22, 0.42)
    elif row == 2:
        ax.set_ylim(0.01, 0.42)
    elif row == 3:
        ax.set_ylim(0.19, 0.395)
    if row == 3 and col == 0:
        ax.legend(ncol=2, title=r"Target network update rate $\tau$")


# Create a 3x2 grid of subplots
fig, axs = plt.subplots(4, 2, figsize=(12, 15), constrained_layout=True)
cm = plt.get_cmap('gist_rainbow')

# eps_decay
sort_and_plot_globs(glob.glob("data/hyperparam_optim/eps_decay/*/rewards.npy"), axs[0, 0], 10, legend_inside=True,
                    title="Epsilon decay constant")

# batch_size
sort_and_plot_globs(glob.glob("data/hyperparam_optim/batch_size/*/rewards.npy"), axs[2, 0], 10, legend_inside=True,
                    title="Batch size")

# gamma
sort_and_plot_globs(glob.glob("data/hyperparam_optim/gamma/*/rewards.npy"), axs[0, 1], 10,
                    title=r"Discount factor $\gamma$")

# lr
sort_and_plot_globs(glob.glob("data/hyperparam_optim/lr/*/rewards.npy"), axs[1, 1], 10, title="Learning rate")

# tau
sort_and_plot_globs(glob.glob("data/hyperparam_optim/tau/*/rewards.npy"), axs[3, 0], 10, legend_inside=True)

# memory_size
sort_and_plot_globs(glob.glob("data/hyperparam_optim/memory_size/*/rewards.npy"), axs[2, 1], 10, title="Memory size")

# hidden_layer_sizes
sort_and_plot_globs(glob.glob("data/hyperparam_optim/hidden_layer_sizes/*/rewards.npy"), axs[3, 1], 10,
                    title="Hidden layer sizes")

# activation_function
sort_and_plot_globs(glob.glob("data/hyperparam_optim/activation_function/*/rewards.npy"), axs[1, 0], 10,
                    legend_inside=True, title="Activation function")

# Adjust layout for better appearance
# plt.tight_layout()
plt.savefig("../Thesis/img/impl/hyperparam_optim_8.pdf")
