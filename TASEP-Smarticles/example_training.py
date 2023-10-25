from Trainer import Trainer, Hyperparams, EnvParams

envParams = EnvParams(render_mode=None,
                      length=64,
                      width=16,
                      moves_per_timestep=20,
                      window_height=256,
                      observation_distance=3,
                      initial_state_template="checkerboard",
                      distinguishable_particles=False)
hyperparams = Hyperparams(BATCH_SIZE=256,
                          GAMMA=0,
                          EPS_START=0.9,
                          EPS_END=0.03,
                          EPS_DECAY=40_000,
                          TAU=0.001,
                          LR=1e-2,
                          MEMORY_SIZE=30_000)

trainer = Trainer(envParams, hyperparams, reset_interval=20000,
                  total_steps=100000, render_start=10000, do_plot=True, plot_interval=1000)

trainer.train()

trainer.save("example_model.pt")
