import torch.utils.data as data

import data.util as util
from typing import Dict, List, Optional


class LQGTDataset(data.Dataset):
    def __init__(self, opt: Dict[str, str]):
        super(LQGTDataset, self).__init__()

        self.opt: Dict[str, str] = opt
        self.data_type: str = self.opt["data_type"]

        self.paths_LQ: Optional[List[str]] = None
        self.paths_GT: Optional[List[str]] = None
        self.sizes_LQ: Optional[int] = None
        self.sizes_GT: Optional[int] = None

        self.paths_GT, self.sizes_GT = util.get_image_paths(
            self.data_type, opt["dataroot_GT"]
        )
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(
            self.data_type, opt["dataroot_LQ"]
        )

        assert self.paths_GT, "Error: GT path is empty."

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), "GT and LQ datasets have different number of images - {}, {}.".format(
                len(self.paths_LQ), len(self.paths_GT)
            )

    def __getitem__(self, index: int):
        GT_path: Optional[str] = None
        LQ_path: Optional[str] = None

        scale: int = int(self.opt['scale'])
        GT_size: int = int(self.opt['GT_size'])

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_img(GT_path)