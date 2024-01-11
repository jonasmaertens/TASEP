# import sys
# sys.path.append('src/SmartTasep/') # uncomment this line if PYTHNONPATH is not set in IDE
from Trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer.load(model_id=34)
    trainer.hyperparams["EPS_END"] = 0.15
    trainer.env.forward_reward = 2
    trainer.env.inh_rew_idx = 1
    # trainer.train_and_save()
    trainer.run()
