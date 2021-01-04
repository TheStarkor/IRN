import argparse
import logging
import torch
from typing import Dict

import options.options as option
from utils import util
from data import create_dataset

PATH = "training"
NAME = "testing"
MANUAL_SEED = 10

def main():
    ### parser
    ### diff : cannot support distribution
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    ### mkdir and loggers
    # TODO : rm
    # util.mkdir_and_rename(PATH)

    util.setup_logger('base', PATH, 'train_' + NAME, level=logging.INFO, screen=True, tofile=False)
    util.setup_logger('val', PATH, 'val_' + NAME, level=logging.INFO, screen=True, tofile=False)

    logger = logging.getLogger('base')

    # TODO : tensorboard logger

    ### random seed
    seed = MANUAL_SEED
    logger.info(f'Random seed: {seed}')
    util.set_random_seed(seed)

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
    torch.backends.cudnn.benchmark = True

    ### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
        elif phase == 'val':
            print('VAL!')
        else:
            raise NotImplementedError(f'Phase [{phase:s}] is not recognized')

    # TODO : create model

    # TODO : training

if __name__ == "__main__":
    main()