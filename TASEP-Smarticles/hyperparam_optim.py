# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
import os

import matplotlib.pyplot as plt

from Trainer import Trainer, Hyperparams, EnvParams
import numpy as np

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

    #taus = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    taus = [0]
    for tau in taus:
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
                                  TAU=tau,
                                  LR=0.005,
                                  MEMORY_SIZE=1_000_000)
        trainer = Trainer(envParams, hyperparams,
                          total_steps=5_000,
                          do_plot=False,
                          plot_interval=5000,
                          new_model=False)
        rewards = []
        for i in range(100):
            trainer.total_steps = 10000
            trainer.run(reset_stats=True)
            rewards.append(np.mean(trainer.all_rewards))
            trainer.total_steps = 2500
            trainer.train()
        dir = f"data/hyperparam_optim/tau/{tau}"
        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/rewards.npy", "wb") as f:
            np.save(f, np.array(rewards))
        plt.plot(rewards, label=f"tau={tau}")

    plt.legend()
    plt.show()
