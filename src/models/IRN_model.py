import logging
from collections import OrderedDict
import torch
import torch.nn as nn

from .base_model import BaseModel

logger = logging.getLogger('base')

class IRNModel(BaseModel):
    def __init__(self, opt: dict):
        super(IRNModel, self).__init__(opt)

        self.train_opt: str = opt['train']
        self.test_opt: str = opt['test']

        # TODO
        self.netG = a