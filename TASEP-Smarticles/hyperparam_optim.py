# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
import os
from tqdm import tqdm
import multiprocessing as mp
import torch.nn as nn

import matplotlib.pyplot as plt

from Trainer import Trainer, Hyperparams, EnvParams
import numpy as np


def train_for_hidden_layer_size(hidden_layer_size):
    envParams = EnvParams(render_mode=None,
                          length=100,
                          width=10,
                          moves_per_timestep=200,
                          window_height=100,
                          observation_distance=2,
                          initial_state_template="checkerboard",
                          use_speeds=True,
                          distinguishable_particles=True,
                          sigma=5,
                          social_reward=True,
                          allow_wait=True,
                          invert_speed_observation=True)

    hyperparams = Hyperparams(BATCH_SIZE=32,
                              GAMMA=0.9,
                              EPS_START=0.9,
                              EPS_END=0.05,
                              EPS_DECAY=50_000,
                              TAU=0.005,
                              LR=0.005,
                              MEMORY_SIZE=1_000_000)
    trainer = Trainer(envParams, hyperparams,
                      total_steps=5_000,
                      do_plot=False,
                      plot_interval=5000,
                      hidden_layer_sizes=hidden_layer_size,
                      progress_bar=False)
    rewards = []
    for i in range(100):
        trainer.total_steps = 10000
        trainer.run(reset_stats=True)
        rewards.append(np.mean(trainer.all_rewards))
        trainer.total_steps = 2500
        trainer.train()
    dir = f"data/hyperparam_optim/hidden_layer_sizes/{hidden_layer_size}"
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/rewards.npy", "wb") as f:
        np.save(f, np.array(rewards))
    return rewards, hidden_layer_size


def train_for_activation_function(activation_function):
    envParams = EnvParams(render_mode=None,
                          length=100,
                          width=10,
                          moves_per_timestep=200,
                          window_height=100,
                          observation_distance=2,
                          initial_state_template="checkerboard",
                          use_speeds=True,
                          distinguishable_particles=True,
                          sigma=5,
                          social_reward=True,
                          allow_wait=True,
                          invert_speed_observation=True)

    hyperparams = Hyperparams(BATCH_SIZE=32,
                              GAMMA=0.9,
                              EPS_START=0.9,
                              EPS_END=0.05,
                              EPS_DECAY=50_000,
                              TAU=0.005,
                              LR=0.005,
                              MEMORY_SIZE=1_000_000)
    trainer = Trainer(envParams, hyperparams,
                      total_steps=5_000,
                      do_plot=False,
                      plot_interval=5000,
                      activation_function=activation_function,
                      new_model=True,
                      progress_bar=False)

    rewards = []
    for i in range(100):
        trainer.total_steps = 10000
        trainer.run(reset_stats=True)
        rewards.append(np.mean(trainer.all_rewards))
        print(trainer.all_rewards)
        trainer.total_steps = 2500
        trainer.train()
    dir = f"data/hyperparam_optim/activation_function/{activation_function.__class__.__name__}"
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/rewards.npy", "wb") as f:
        np.save(f, np.array(rewards))
    return rewards, activation_function


if __name__ == '__main__':
    # eps_decays = [0.01, 1000, 10_000, 100_000, 1_000_000]
    # for eps_decay in eps_decays:
    #     envParams = EnvParams(render_mode=None,
    #                           length=100,
    #                           width=10,
    #                           moves_per_timestep=200,
    #                           window_height=100,
    #                           observation_distance=2,
    #                           initial_state_template="checkerboard",
    #                           use_speeds=True,
    #                           distinguishable_particles=True,
    #                           sigma=5,
    #                           social_reward=True,
    #                           allow_wait=True,
    #                           invert_speed_observation=True)
    #
    #     hyperparams = Hyperparams(BATCH_SIZE=64,
    #                               GAMMA=0.8,
    #                               EPS_START=0.9,
    #                               EPS_END=0.05,
    #                               EPS_DECAY=eps_decay,
    #                               TAU=0.005,
    #                               LR=0.005,
    #                               MEMORY_SIZE=1_000_000)
    #     trainer = Trainer(envParams, hyperparams,
    #                       total_steps=5_000,
    #                       do_plot=False,
    #                       plot_interval=5000,
    #                       new_model=False)
    #     rewards = []
    #     for i in range(100):
    #         trainer.total_steps = 10000
    #         trainer.run(reset_stats=True)
    #         rewards.append(np.mean(trainer.all_rewards))
    #         trainer.total_steps = 2500
    #         trainer.train()
    #     dir = f"data/hyperparam_optim/eps_decay/{eps_decay}"
    #     os.makedirs(dir, exist_ok=True)
    #     with open(f"{dir}/rewards.npy", "wb") as f:
    #         np.save(f, np.array(rewards))
    #     plt.plot(rewards, label=f"eps_decay={eps_decay}")

    # batch_sizes = [8, 32, 64, 256, 1024]
    # for batch_size in batch_sizes:
    #     envParams = EnvParams(render_mode=None,
    #                           length=100,
    #                           width=10,
    #                           moves_per_timestep=200,
    #                           window_height=100,
    #                           observation_distance=2,
    #                           initial_state_template="checkerboard",
    #                           use_speeds=True,
    #                           distinguishable_particles=True,
    #                           sigma=5,
    #                           social_reward=True,
    #                           allow_wait=True,
    #                           invert_speed_observation=True)
    #
    #     hyperparams = Hyperparams(BATCH_SIZE=batch_size,
    #                               GAMMA=0.8,
    #                               EPS_START=0.9,
    #                               EPS_END=0.05,
    #                               EPS_DECAY=10_000,
    #                               TAU=0.005,
    #                               LR=0.005,
    #                               MEMORY_SIZE=1_000_000)
    #     trainer = Trainer(envParams, hyperparams,
    #                       total_steps=5_000,
    #                       do_plot=False,
    #                       plot_interval=5000,
    #                       new_model=False)
    #     rewards = []
    #     for i in range(100):
    #         trainer.total_steps = 10000
    #         trainer.run(reset_stats=True)
    #         rewards.append(np.mean(trainer.all_rewards))
    #         trainer.total_steps = 2500
    #         trainer.train()
    #     dir = f"data/hyperparam_optim/batch_size/{batch_size}"
    #     os.makedirs(dir, exist_ok=True)
    #     with open(f"{dir}/rewards.npy", "wb") as f:
    #         np.save(f, np.array(rewards))
    # plt.plot(rewards, label=f"batch_size={batch_size}")

    # gammas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # for gamma in gammas:
    #     envParams = EnvParams(render_mode=None,
    #                           length=100,
    #                           width=10,
    #                           moves_per_timestep=200,
    #                           window_height=100,
    #                           observation_distance=2,
    #                           initial_state_template="checkerboard",
    #                           use_speeds=True,
    #                           distinguishable_particles=True,
    #                           sigma=5,
    #                           social_reward=True,
    #                           allow_wait=True,
    #                           invert_speed_observation=True)
    #
    #     hyperparams = Hyperparams(BATCH_SIZE=64,
    #                               GAMMA=gamma,
    #                               EPS_START=0.9,
    #                               EPS_END=0.05,
    #                               EPS_DECAY=10_000,
    #                               TAU=0.005,
    #                               LR=0.005,
    #                               MEMORY_SIZE=1_000_000)
    #     trainer = Trainer(envParams, hyperparams,
    #                       total_steps=5_000,
    #                       do_plot=False,
    #                       plot_interval=5000,
    #                       new_model=False)
    #     rewards = []
    #     for i in range(100):
    #         trainer.total_steps = 10000
    #         trainer.run(reset_stats=True)
    #         rewards.append(np.mean(trainer.all_rewards))
    #         trainer.total_steps = 2500
    #         trainer.train()
    #     dir = f"data/hyperparam_optim/gamma/{gamma}"
    #     os.makedirs(dir, exist_ok=True)
    #     with open(f"{dir}/rewards.npy", "wb") as f:
    #         np.save(f, np.array(rewards))
    # plt.plot(rewards, label=f"gamma={gamma}")

    # lrs = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    # for lr in lrs:
    #     envParams = EnvParams(render_mode=None,
    #                           length=100,
    #                           width=10,
    #                           moves_per_timestep=200,
    #                           window_height=100,
    #                           observation_distance=2,
    #                           initial_state_template="checkerboard",
    #                           use_speeds=True,
    #                           distinguishable_particles=True,
    #                           sigma=5,
    #                           social_reward=True,
    #                           allow_wait=True,
    #                           invert_speed_observation=True)
    #
    #     hyperparams = Hyperparams(BATCH_SIZE=8,
    #                               GAMMA=0.9,
    #                               EPS_START=0.9,
    #                               EPS_END=0.05,
    #                               EPS_DECAY=10_000,
    #                               TAU=0.005,
    #                               LR=lr,
    #                               MEMORY_SIZE=1_000_000)
    #     trainer = Trainer(envParams, hyperparams,
    #                       total_steps=5_000,
    #                       do_plot=False,
    #                       plot_interval=5000,
    #                       new_model=False)
    #     rewards = []
    #     for i in range(100):
    #         trainer.total_steps = 10000
    #         trainer.run(reset_stats=True)
    #         rewards.append(np.mean(trainer.all_rewards))
    #         trainer.total_steps = 2500
    #         trainer.train()
    #     dir = f"data/hyperparam_optim/lr/{lr}"
    #     os.makedirs(dir, exist_ok=True)
    #     with open(f"{dir}/rewards.npy", "wb") as f:
    #         np.save(f, np.array(rewards))
    #     plt.plot(rewards, label=f"lr={lr}")

    # taus = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    # taus = [0.05]
    # for tau in tqdm(taus):
    #     envParams = EnvParams(render_mode=None,
    #                           length=100,
    #                           width=10,
    #                           moves_per_timestep=200,
    #                           window_height=100,
    #                           observation_distance=2,
    #                           initial_state_template="checkerboard",
    #                           use_speeds=True,
    #                           distinguishable_particles=True,
    #                           sigma=5,
    #                           social_reward=True,
    #                           allow_wait=True,
    #                           invert_speed_observation=True)
    #
    #     hyperparams = Hyperparams(BATCH_SIZE=32,
    #                               GAMMA=0.9,
    #                               EPS_START=0.9,
    #                               EPS_END=0.05,
    #                               EPS_DECAY=50_000,
    #                               TAU=tau,
    #                               LR=0.005,
    #                               MEMORY_SIZE=1_000_000)
    #     trainer = Trainer(envParams, hyperparams,
    #                       total_steps=5_000,
    #                       do_plot=False,
    #                       plot_interval=5000,
    #                       new_model=False)
    #     rewards = []
    #     for i in tqdm(range(100), leave=False):
    #         trainer.total_steps = 10000
    #         trainer.run(reset_stats=True)
    #         rewards.append(np.mean(trainer.all_rewards))
    #         trainer.total_steps = 2500
    #         trainer.train()
    #     dir = f"data/hyperparam_optim/tau/{tau}"
    #     os.makedirs(dir, exist_ok=True)
    #     with open(f"{dir}/rewards.npy", "wb") as f:
    #         np.save(f, np.array(rewards))
    #     plt.plot(rewards, label=f"tau={tau}")
    #
    # plt.legend()
    # plt.show()

    # memory_sizes = [33, 100, 500, 1000, 10000, 50000, 100000, 1000000]
    # for memory_size in tqdm(memory_sizes):
    #     envParams = EnvParams(render_mode=None,
    #                           length=100,
    #                           width=10,
    #                           moves_per_timestep=200,
    #                           window_height=100,
    #                           observation_distance=2,
    #                           initial_state_template="checkerboard",
    #                           use_speeds=True,
    #                           distinguishable_particles=True,
    #                           sigma=5,
    #                           social_reward=True,
    #                           allow_wait=True,
    #                           invert_speed_observation=True)
    #
    #     hyperparams = Hyperparams(BATCH_SIZE=32,
    #                               GAMMA=0.9,
    #                               EPS_START=0.9,
    #                               EPS_END=0.05,
    #                               EPS_DECAY=50_000,
    #                               TAU=0.005,
    #                               LR=0.005,
    #                               MEMORY_SIZE=memory_size)
    #     trainer = Trainer(envParams, hyperparams,
    #                       total_steps=5_000,
    #                       do_plot=False,
    #                       plot_interval=5000,
    #                       new_model=False)
    #     rewards = []
    #     for i in tqdm(range(100), leave=False):
    #         trainer.total_steps = 10000
    #         trainer.run(reset_stats=True)
    #         rewards.append(np.mean(trainer.all_rewards))
    #         trainer.total_steps = 2500
    #         trainer.train()
    #     dir = f"data/hyperparam_optim/memory_size/{memory_size}"
    #     os.makedirs(dir, exist_ok=True)
    #     with open(f"{dir}/rewards.npy", "wb") as f:
    #         np.save(f, np.array(rewards))
    #     plt.plot(rewards, label=f"memory_size={memory_size}")
    #
    # plt.legend()
    # plt.show()

    # hidden_layer_sizes = [[], [6], [24], [128], [512], [6, 6], [24, 24], [24, 12], [128, 128], [128, 24], [512, 512],
    #                       [6, 6, 6],
    #                       [24, 24, 24], [128, 128, 128]]
    #
    # with mp.Pool(mp.cpu_count()) as pool:
    #     #results = pool.map(train_for_hidden_layer_size, hidden_layer_sizes)
    #     results = list(tqdm(pool.imap(train_for_hidden_layer_size, hidden_layer_sizes), total=len(hidden_layer_sizes)))
    #
    # for rewards, hidden_layer_size in results:
    #     plt.plot(rewards, label=f"hidden_layer_size={hidden_layer_size}")
    #
    # plt.legend()
    # plt.show()

    # activation_functions = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU(), nn.Softsign()]
    # with mp.Pool(5) as pool:
    #     # results = pool.map(train_for_activation_function, activation_functions)
    #     results = list(tqdm(pool.imap(train_for_activation_function, activation_functions), total=len(activation_functions)))
    #
    # for rewards, activation_function in results:
    #     plt.plot(rewards, label=f"activation_function={activation_function.__class__.__name__}")
    #
    # plt.legend()
    # plt.show()

    train_for_activation_function(nn.Sigmoid())



