# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode="human",
                          length=128,
                          width=24,
                          moves_per_timestep=200,
                          window_height=200,
                          observation_distance=3,
                          distinguishable_particles=True,
                          initial_state_template="checkerboard",
                          social_reward=True,
                          # density=0.5,
                          use_speeds=True,
                          sigma=10,
                          allow_wait=True,
                          invert_speed_observation=True,
                          speed_observation_threshold=0.35,
                          #punish_inhomogeneities=True,
                          #inh_rew_idx=1,
                          #speed_gradient_reward=False,
                          #binary_speeds=True,
                          #choices=4,
                          # speed_gradient_linearity=0.1,
                          )
    hyperparams = Hyperparams(BATCH_SIZE=32,
                              GAMMA=0.85,
                              EPS_START=0.9,
                              EPS_END=0.05,
                              EPS_DECAY=100_000,
                              TAU=0.005,
                              LR=0.001,
                              MEMORY_SIZE=500_000)

    trainer = Trainer(envParams, hyperparams, reset_interval=100_000,
                      total_steps=500_000, do_plot=True, plot_interval=4000, new_model=True)

    trainer.train_and_save()
