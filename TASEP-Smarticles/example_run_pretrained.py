# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer.load(total_steps=1000000,
                           render_start=0,
                           do_plot=True,
                           average_window=8000,
                           window_height=300,
                           moves_per_timestep=150,
                           wait_initial=0,

                           )
    trainer.run()
