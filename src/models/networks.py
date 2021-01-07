import torch
import logging

logger = logging.getLogger('base')

def define_G(opt):
    opt_net: dict = opt['network_G']
    which_model: dict = opt_net['which_model_G']
    subnet_type: dict = which_model['subnet_type']

    # ??? 어차피 전부 xavier 같은데
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num: int = int(math.log(opt_net['scale'], 2))

    # TODO
    netG = 'invrescalenet'