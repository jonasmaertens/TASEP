# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode=None,
                          length=128,
                          width=32,
                          moves_per_timestep=20,
                          window_height=256,
                          observation_distance=2,
                          initial_state_template="checkerboard",
                          distinguishable_particles=True,
                          use_speeds=True,
                          sigma=0.5,
                          average_window=2500)
    hyperparams = Hyperparams(BATCH_SIZE=512,
                              GAMMA=0.9,
                              EPS_START=0.9,
                              EPS_END=0.01,
                              EPS_DECAY=40000,
                              TAU=0.0001,
                              LR=0.005,
                              MEMORY_SIZE=100000)

    trainer = Trainer(envParams, hyperparams, reset_interval=40000,
                      total_steps=3000000, do_plot=True, plot_interval=2500)

    trainer.train()

    # trainer.save_plot()

    # trainer.save("example_model_distinguishable.pt")
