"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train.py --settings_file "config/settings_DDD17.yaml"
"""
import argparse
import wandb

from config.settings import Settings
from training.ess_trainer import ESSModel
from training.ess_supervised_trainer import ESSSupervisedModel

import numpy as np
import torch
import random
import os

# random seed
seed_value = 6
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    wandb.init(name=(settings.dataset_name_b.split("_")[0] + '_' + settings.timestr), project="zhaoning_sun_semester_thesis", entity="zhasun", sync_tensorboard=True)

    if settings.model_name == 'ess':
        trainer = ESSModel(settings)
    elif settings.model_name == 'ess_supervised':
        trainer = ESSSupervisedModel(settings)

    else:
        raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)

    wandb.config = {
        "random_seed": seed_value,
        "lr_front": settings.lr_front,
        "lr_back": settings.lr_back,
        "batch_size_a": settings.batch_size_a,
        "batch_size_b": settings.batch_size_b
    }

    trainer.train()


if __name__ == "__main__":
    main()
