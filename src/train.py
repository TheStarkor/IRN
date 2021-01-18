import argparse
import logging
import torch
import math
from typing import Dict, Union, Any

import options.options as option
from utils import util
from data import create_dataset, create_dataloader
from models import create_model

PATH = "training"
NAME = "testing"
MANUAL_SEED = 10


def main():
    ### parser
    ### diff : cannot support distribution
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to YAML file.")
    args = parser.parse_args()
    opt: Dict[str, Any] = option.parse(args.opt, is_train=True)

    ### mkdir and loggers
    # TODO : rm
    # util.mkdir_and_rename(PATH)

    util.setup_logger(
        "base", PATH, "train_" + NAME, level=logging.INFO, screen=True, tofile=False
    )
    util.setup_logger(
        "val", PATH, "val_" + NAME, level=logging.INFO, screen=True, tofile=False
    )

    logger: Logger = logging.getLogger("base")

    # TODO : tensorboard logger

    ### random seed
    seed: int = MANUAL_SEED
    logger.info(f"Random seed: {seed}")
    util.set_random_seed(seed)

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
    torch.backends.cudnn.benchmark = True

    ### create train and val dataloader
    phase: str
    dataset_opt: Dict[str, Any]
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set: Union[LQGTDataset] = create_dataset(dataset_opt)
            train_size: int = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters: int = int(opt["train"]["niter"])
            total_epochs: int = int(math.ceil(total_iters / train_size))

            train_loader = create_dataloader(train_set, dataset_opt, opt, None)

            logger.info(
                "Number of train images: {:,d}, iters: {:,d}".format(
                    len(train_set), train_size
                )
            )
            logger.info(
                "Total epochs needed: {:d} for iters {:,d}".format(
                    total_epochs, total_iters
                )
            )
        elif phase == "val":
            print("VAL!")
        else:
            raise NotImplementedError(f"Phase [{phase:s}] is not recognized")

    model = create_model(opt)

    current_step: int = 0
    start_epoch: int = 0

    # TODO : training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_step}")
    # for epoch in range(start_epoch, total_epochs + 1):
    for epoch in range(start_epoch, 3 + 1):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            ### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)


if __name__ == "__main__":
    main()
