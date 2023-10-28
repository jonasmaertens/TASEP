import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer, Hyperparams, EnvParams
import numpy as np

if __name__ == '__main__':
    sigmas = np.logspace(-4, 1, 50, dtype=np.float32)
    print(sigmas)
    for sigma in sigmas:
        envParams = EnvParams(render_mode=None,
                              length=128,
                              width=32,
                              moves_per_timestep=20,
                              window_height=256,
                              observation_distance=2,
                              initial_state_template="checkerboard",
                              distinguishable_particles=True,
                              use_speeds=True,
                              sigma=sigma,
                              average_window=2500)
        hyperparams = Hyperparams(BATCH_SIZE=512,
                                  GAMMA=0.999,
                                  EPS_START=0.9,
                                  EPS_END=0.01,
                                  EPS_DECAY=40000,
                                  TAU=0.0001,
                                  LR=0.005,
                                  MEMORY_SIZE=100000)

        trainer = Trainer(envParams, hyperparams, reset_interval=40000,
                          total_steps=100000, do_plot=False, plot_interval=2500)

        trainer.train()

        trainer.save_plot(f"plots/different_speeds/individual_sigmas/sigma_{sigma:.2f}.png")

        trainer.save(f"models/different_speeds/individual_sigmas/model_100000_steps_sigma_{sigma:.2f}.pt")

