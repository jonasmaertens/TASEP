from smarttasep import Trainer

if __name__ == '__main__':
    trainer = Trainer.load(model_id=6, do_plot=True, total_steps=1_500_010, average_window=2000)
    # trainer.env.unwrapped.render_mode = None
    trainer.run(reset_stats=True)
    # trainer.save_run_data_for_plot(name="uniform_speeds")
