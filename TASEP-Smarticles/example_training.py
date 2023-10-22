from Trainer import Trainer, Hyperparams, EnvParams

envParams = EnvParams(render_mode=None,
                      length=64,
                      width=16,
                      moves_per_timestep=10,
                      window_height=256,
                      observation_distance=3,
                      initial_state_template="checkerboard")
hyperparams = Hyperparams(BATCH_SIZE=256,
                          GAMMA=0.6,
                          EPS_START=0.9,
                          EPS_END=0.03,
                          EPS_DECAY=40_000,
                          TAU=0.001,
                          LR=1e-2,
                          MEMORY_SIZE=30_000)

trainer = Trainer(envParams, hyperparams, reset_interval=2_000,
                  total_steps=5_000, render_start=2_500, do_plot=True, plot_interval=1000)

trainer.train()

trainer.save("example_model.pt")
