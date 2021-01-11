import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

class IRNModel(BaseModel):
    def __init__(self, opt: dict):
        super(IRNModel, self).__init__(opt)

        self.train_opt: str = opt['train']
        # self.test_opt: str = opt['test']

        # TODO : fix
        # self.netG = networks.define_G(opt).to(self.device)
        self.netG = networks.define_G(opt)
        self.netG = DataParallel(self.netG)

        self.print_network()
        # self.load() TODO

        # TODO

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str: str = f'{self.netG.__class__.__name__} - {self.netG.module.__class__.__name__}'
        else:
            net_struc_str: str = f'{self.netG.__class__.__name__}'
        logger.info(f'Network G structure: {net_struc_str}, with parameters: {n}')
        logger.info(s)

    def load(self):
        pass
        # load_path_G: str = self.opt['path']['pretrain_model_G']
        # if load_path_G is not None:
        #     logger.info(f'Loading model for G [{load_path_G}] ...')
            # self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])