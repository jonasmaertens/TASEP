# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer.load(model_id=36)
    trainer.run()
