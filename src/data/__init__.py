import logging
import torch
from typing import Dict, Union, Type
from data.LQGT_dataset import LQGTDataset


def create_dataloader(
    dataset: Union[LQGTDataset],
    dataset_opt: Dict[str, str],
    opt: Dict[str, str] = None,
    sampler=None,
):
    phase: str = dataset_opt["phase"]

    if phase == "train":
        num_workers: int = int(dataset_opt["n_workers"]) * len(opt["gpu_ids"])
        batch_size: int = int(dataset_opt["batch_size"])

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            sampler=None,
            drop_last=True,
            pin_memory=False,
        )

    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )


def create_dataset(dataset_opt: Dict[str, str]) -> Union[LQGTDataset]:
    mode = dataset_opt["mode"]
    if mode == "LQ":
        # TODO : later
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
