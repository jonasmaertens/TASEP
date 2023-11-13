# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode="human",
                          length=128,
                          width=32,
                          moves_per_timestep=80,
                          window_height=300,
                          observation_distance=3,
                          distinguishable_particles=True,
                          #initial_state_template="checkerboard",
                          use_speeds=True,
                          sigma=5,
                          allow_wait=True,
                          # social_reward=0.8,
                          invert_speed_observation=True,
                          speed_observation_threshold=0.35,
                          punish_inhomogeneities=True,
                          density=0.2)
    hyperparams = Hyperparams(BATCH_SIZE=128,
                              GAMMA=0.99,
                              EPS_START=0.9,
                              EPS_END=0.05,
                              EPS_DECAY=50_000,
                              TAU=0.005,
                              LR=0.005,
                              MEMORY_SIZE=900_000)

    trainer = Trainer(envParams, hyperparams, reset_interval=80000,
                      total_steps=300_000, do_plot=True, plot_interval=2000, random_density=False)

    trainer.train_and_save()
