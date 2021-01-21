import os
import os.path as osp
import logging
import yaml

from utils.util import OrderedYaml

Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)

    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    opt["is_train"] = is_train
    if opt["distortion"] == "sr":
        scale = opt["scale"]

    # datasets
    for phase, dataset in opt["datasets"].items():
        dataset["phase"] = phase
        if opt["distortion"] == "sr":
            dataset["scale"] = scale
        if dataset.get("dataroot_GT", None) is not None:
            dataset["dataroot_GT"] = osp.expanduser(dataset["dataroot_GT"])
        if dataset.get("dataroot_LQ", None) is not None:
            dataset["dataroot_LQ"] = osp.expanduser(dataset["dataroot_LQ"])
        dataset["data_type"] = "img"

    opt["is_train"] = is_train
    if opt["distortion"] == "sr":
        scale = opt["scale"]

    # path
    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir)
    )
    if is_train:
        experiments_root = osp.join(opt["path"]["root"], "experiments", opt["name"])
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")
        opt["path"]["models"] = osp.join(experiments_root, "models")

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
