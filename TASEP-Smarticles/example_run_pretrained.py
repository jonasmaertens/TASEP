from smarttasep import Trainer

if __name__ == '__main__':
    trainer = Trainer.load(model_id=7, do_plot=True, total_steps=8_500_010, average_window=5000, wait_initial=5)
    # trainer.env.unwrapped.render_mode = None
    trainer.run(reset_stats=True)
    trainer.save_run_data_for_plot(name="speed_grad_4")
