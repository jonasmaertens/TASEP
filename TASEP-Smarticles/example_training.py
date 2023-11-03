# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode="human",
                          length=128,
                          width=32,
                          moves_per_timestep=400,
                          window_height=400,
                          observation_distance=3,
                          distinguishable_particles=True,
                          use_speeds=False,
                          #sigma=1,
                          average_window=2500,
                          allow_wait=True,
                          social_reward=0.6)
    hyperparams = Hyperparams(BATCH_SIZE=256,
                              GAMMA=0.8,
                              EPS_START=0.9,
                              EPS_END=0.01,
                              EPS_DECAY=40000,
                              TAU=0.005,
                              LR=0.005,
                              MEMORY_SIZE=100000)

    trainer = Trainer(envParams, hyperparams, reset_interval=50000,
                      total_steps=1500000, do_plot=True, plot_interval=2500, random_density=True, )

    trainer.train()

    # trainer.save_plot()

    trainer.save("models/same_speeds/allow_wait/model_1500000_steps_social_0.6_different_rho.pt", append_timestamp=True)
