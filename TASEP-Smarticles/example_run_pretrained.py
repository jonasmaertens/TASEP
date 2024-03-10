from smarttasep import Trainer

if __name__ == '__main__':
    trainer = Trainer.load(model_id=7, do_plot=True, total_steps=250000 ,do_render=True)
                           # average_window=5000,
                           # moves_per_timestep=1000, do_render=True)
    trainer.env.unwrapped.sigma = 4
    trainer.env.unwrapped.prepare_lane_gradient = True
    trainer.env.unwrapped.inflate_speeds = True
    trainer.run(reset_stats=True)
    # trainer.save_run_data_for_plot(name="lanes_1")
