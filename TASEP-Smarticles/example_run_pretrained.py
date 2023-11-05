# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer.load()
    trainer.total_steps = 1000000
    trainer.render_start = 0
    trainer.progress_bar = True
    trainer.do_plot = True
    trainer.run()
