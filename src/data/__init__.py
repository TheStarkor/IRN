import logging
import torch
from typing import Dict, Union, Type
from data.LQGT_dataset import LQGTDataset


def create_dataset(dataset_opt: Dict[str, str]) -> Union[LQGTDataset]:
    mode = dataset_opt["mode"]
    if mode == "LQ":
        # TODO
        pass
    elif mode == "LQGT":
        from data.LQGT_dataset import LQGTDataset as D
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

    dataset: LQGTDataset = D(dataset_opt)

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset
