# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])

if __name__ == '__main__':
    trainer = Trainer.load(model_id=40, do_plot=False)
    trainer.total_steps = 1_500_010
    trainer.env.unwrapped.average_window = 2000
    trainer.plot_interval = 2000
    trainer.env.unwrapped.render_mode = None
    trainer.run(reset_stats=True)
    trainer.save_run_data_for_plot(name="equal_speeds")
