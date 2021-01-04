import logging
import torch
from typing import Dict


def create_dataset(dataset_opt: Dict[str, str]):
    mode = dataset_opt["mode"]
    if mode == "LQ":
        # TODO
        pass
    elif mode == "LQGT":
        from data.LQGT_dataset import LQGTDataset as D
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

    dataset = D(dataset_opt)

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
