# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    trainer = Trainer.load(model_id=41, do_plot=False, total_steps=1_500_010, average_window=2000)
    # trainer.env.unwrapped.render_mode = None
    trainer.run(reset_stats=True)
    trainer.save_run_data_for_plot(name="uniform_speeds")
