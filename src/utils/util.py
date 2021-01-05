import os
import logging
import random
import numpy as np  # type: ignore
import torch
import time
from datetime import datetime
from collections import OrderedDict
from torchvision.utils import save_image  # type: ignore

import yaml
from yaml import CLoader as Loader, CDumper as Dumper


def OrderedYaml():
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represend_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)

    return Loader, Dumper


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def debugging_image_torch(data, filename=str(time.time())):
    save_image(data["LQ"], f"debugging/{filename}_lq.png")
    save_image(data["GT"], f"debugging/{filename}_gt.png")
