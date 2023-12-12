# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode="human",
                          length=100,
                          width=10,
                          moves_per_timestep=200,
                          window_height=100,
                          observation_distance=4,
                          distinguishable_particles=True,
                          # initial_state_template="checkerboard",
                          density=0.5,
                          use_speeds=True,
                          sigma=5,
                          allow_wait=False,
                          # social_reward=0.8,
                          invert_speed_observation=True,
                          speed_observation_threshold=0.35,
                          punish_inhomogeneities=True,
                          inh_rew_idx=1,
                          # density=0.2,
                          speed_gradient_reward=False,
                          binary_speeds=True,
                          choices=4,
                          # speed_gradient_linearity=0.1,
                          )
    hyperparams = Hyperparams(BATCH_SIZE=64,
                              GAMMA=0.8,
                              EPS_START=0.9,
                              EPS_END=0.05,
                              EPS_DECAY=300_000,
                              TAU=0.005,
                              LR=0.005,
                              MEMORY_SIZE=1_000_000)

    trainer = Trainer(envParams, hyperparams, reset_interval=3000_000,
                      total_steps=3000_000, do_plot=True, plot_interval=4000, random_density=False, new_model=True,
                      different_models=True, num_models=4, prio_exp_replay=False)

    trainer.train_and_save()
