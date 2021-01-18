import torch
import torch.nn as nn
import numpy as np


class ReconstructionLoss(nn.Module):
    def __init__(self, losstype="l2", eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == "l2":
            return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))
        elif self.losstype == "l1":
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            print("err")
            return 0
