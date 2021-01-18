import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

import models.networks as networks
from .base_model import BaseModel
from models.modules.Quantization import Quantization
from models.modules.loss import ReconstructionLoss

logger = logging.getLogger("base")


class IRNModel(BaseModel):
    def __init__(self, opt: dict):
        super(IRNModel, self).__init__(opt)

        self.train_opt: str = opt["train"]
        # self.test_opt: str = opt['test']

        # TODO : fix
        # self.netG = networks.define_G(opt).to(self.device)
        self.netG = networks.define_G(opt)
        self.netG = DataParallel(self.netG)

        self.print_network()
        # self.load() TODO

        # TODO
        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            self.Reconstruction_forw = ReconstructionLoss(
                losstype=self.train_opt["pixel_criterion_forw"]
            )
            self.Reconstruction_back = ReconstructionLoss(
                losstype=self.train_opt["pixel_criterion_back"]
            )

            wd_G = (
                self.train_opt["weight_decay_G"]
                if self.train_opt["weight_decay_G"]
                else 0
            )
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning("Params [{:s}] will not optimize.".format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params,
                lr=self.train_opt["lr_G"],
                weight_decay=wd_G,
                betas=(self.train_opt["beta1"], self.train_opt["beta2"]),
            )

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str: str = f"{self.netG.__class__.__name__} - {self.netG.module.__class__.__name__}"
        else:
            net_struc_str: str = f"{self.netG.__class__.__name__}"
        logger.info(f"Network G structure: {net_struc_str}, with parameters: {n}")
        logger.info(s)

    def load(self):
        pass
        # load_path_G: str = self.opt['path']['pretrain_model_G']
        # if load_path_G is not None:
        #     logger.info(f'Loading model for G [{load_path_G}] ...')
        # self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
