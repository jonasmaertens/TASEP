# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode="human",
                          length=128,
                          width=48,
                          moves_per_timestep=40,
                          window_height=512,
                          observation_distance=3,
                          initial_state_template="checkerboard",
                          distinguishable_particles=True,
                          use_speeds=False,
                          sigma=1,
                          average_window=2500,
                          allow_wait=True)
    hyperparams = Hyperparams(BATCH_SIZE=256,
                              GAMMA=0.99,
                              EPS_START=0.9,
                              EPS_END=0.01,
                              EPS_DECAY=40000,
                              TAU=0.005,
                              LR=0.005,
                              MEMORY_SIZE=100000)

    trainer = Trainer(envParams, hyperparams, reset_interval=60000,
                      total_steps=200000, do_plot=True, plot_interval=2500)

    trainer.train()

    #trainer.save_plot()

    trainer.save("models/different_speeds/allow_wait/model_500000_steps_sigma_1.pt")
