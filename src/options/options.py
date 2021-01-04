import os
import os.path as osp
import logging
import yaml

from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['data_type'] = 'img'

    # path
    opt['is_train'] = is_train
    if opt['distortion'] == 'sr':
        scale = opt['scale']

    return opt