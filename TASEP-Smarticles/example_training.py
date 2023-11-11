# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode=None,
                          length=128,
                          width=32,
                          moves_per_timestep=400,
                          window_height=400,
                          observation_distance=3,
                          distinguishable_particles=True,
                          #initial_state_template="checkerboard",
                          use_speeds=True,
                          sigma=5,
                          average_window=5000,
                          allow_wait=True,
                          social_reward=0.6)
    hyperparams = Hyperparams(BATCH_SIZE=256,
                              GAMMA=0.99,
                              EPS_START=0.9,
                              EPS_END=0.01,
                              EPS_DECAY=240000,
                              TAU=0.005,
                              LR=0.005,
                              MEMORY_SIZE=900_000)

    trainer = Trainer(envParams, hyperparams, reset_interval=80000,
                      total_steps=1_500_000, do_plot=True, plot_interval=5000, random_density=True)

    trainer.train_and_safe()
