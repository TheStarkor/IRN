import os
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Tuple


class BaseModel:
    def __init__(self, opt: dict):
        self.opt: dict = opt
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")
        self.is_train: bool = bool(opt["is_train"])
        self.schedulers: list = []
        self.optimizers: list = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        # TODO
        pass

    def _get_init_lr(self):
        # TODO
        pass

    def update_learning_rate(self, cur_iter, wramup_iter=1):
        # TODO
        pass

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network) -> Tuple[str, int]:
        if isinstance(network, nn.DataParallel):
            network = network.module
        s: str = str(network)
        n: int = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        # TODO
        pass

    def load_network(self, load_path, network, strict=True):
        # TODO
        pass

    def save_training_state(self, epoch, iter_step):
        # TODO
        pass

    def resume_training(self, resume_state):
        # TODO
        pass
