import torch
import logging
import math

from models.modules.Inv_arch import InvRescaleNet  # type: ignore
from models.modules.Subnet_constructor import subnet  # type: ignore

logger = logging.getLogger("base")


def define_G(opt):
    opt_net: dict = opt["network_G"]
    which_model: dict = opt_net["which_model_G"]
    subnet_type: str = which_model["subnet_type"]

    # ??? 어차피 전부 xavier 같은데
    if opt_net["init"]:
        init = opt_net["init"]
    else:
        init = "xavier"

    down_num: int = int(math.log(opt_net["scale"], 2))

    netG = InvRescaleNet(
        opt_net["in_nc"],
        opt_net["out_nc"],
        subnet(subnet_type, init),
        opt_net["block_num"],
        down_num,
    )

    return netG
