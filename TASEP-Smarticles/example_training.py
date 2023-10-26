from Trainer import Trainer, Hyperparams, EnvParams

if __name__ == '__main__':
    envParams = EnvParams(render_mode=None,
                          length=128,
                          width=32,
                          moves_per_timestep=20,
                          window_height=256,
                          observation_distance=2,
                          initial_state_template="checkerboard",
                          distinguishable_particles=True)
    hyperparams = Hyperparams(BATCH_SIZE=512,
                              GAMMA=0.999,
                              EPS_START=0.9,
                              EPS_END=0.01,
                              EPS_DECAY=40000,
                              TAU=0.0001,
                              LR=0.001,
                              MEMORY_SIZE=100000)

    trainer = Trainer(envParams, hyperparams, reset_interval=50000,
                      total_steps=200000, render_start=150000, do_plot=True, plot_interval=2500)

    trainer.train()

    trainer.save_plot()

    trainer.save("example_model_distinguishable.pt")
