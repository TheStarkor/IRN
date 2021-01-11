import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Any, List


class HarrDownsampling(nn.Module):
    def __init__(self, channel_in: int):
        super(HarrDownsampling, self).__init__()
        self.channel_in: int = channel_in

        self.harr_weights: Any = torch.ones(4, 1, 2, 2)

        self.harr_weights[1, 0, 0, 1] = -1
        self.harr_weights[1, 0, 1, 1] = -1

        self.harr_weights[2, 0, 1, 0] = -1
        self.harr_weights[2, 0, 1, 1] = -1

        self.harr_weights[3, 0, 1, 0] = -1
        self.harr_weights[3, 0, 0, 1] = -1

        self.harr_weights = torch.cat([self.harr_weights] * self.channel_in, 0)
        self.harr_weights = nn.Parameter(self.harr_weights)
        self.harr_weights.requires_grad = False

    def forward(self, x, rev: bool=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.harr_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shpae[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            out = F.conv_transpose2d(out, self.harr_weights, bias=None, stride=2, groups=self.channel_in)
            return out


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor: Callable[[int, int], Any], channel_num: int, channel_split_num: int, clamp: float=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1: int = channel_split_num
        self.split_len2: int = channel_num - channel_split_num

        self.clamp: float = clamp

        # TODO : study
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev: bool=False):
        # TODO : study
        x1, x2 = x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s)) # TODO : study
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev: bool=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)
        
        return jac / x.shape[0]


class InvRescaleNet(nn.Module):
    def __init__(self, chaneel_in: int=3, channel_out: int=3, subnet_constructor: Callable[[int, int], Any]=None, block_num: List[int]=[], down_num: int=2):
        super(InvRescaleNet, self).__init__()

        operations: list = []

        current_channel: int = chaneel_in

        for i in range(down_num):
            b: HarrDownsampling = HarrDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4

            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev: bool=False, cal_jacobian: bool=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out