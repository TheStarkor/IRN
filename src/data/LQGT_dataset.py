import torch.utils.data as data
import random
import cv2  # type: ignore
import numpy as np  # type: ignore
import torch

import data.util as util
from typing import Dict, List, Optional, Any, Union


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

        # ???
        self.random_scale_list: List[int] = [1]

    def __getitem__(self, index: int) -> Dict[str, Union[Any, str]]:
        GT_path: Optional[str] = None
        LQ_path: Optional[str] = None

        scale: int = int(self.opt["scale"])
        GT_size: int = int(self.opt["GT_size"])

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT: Any = util.read_img(GT_path)

        H: int
        W: int
        C: int

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            img_LQ: Any = util.read_img(LQ_path)
        else:  # down-sampling on-the-fly
            # ??? 왜 하는거지
            if self.opt["phase"] == "train":
                random_scale: int = random.choice(self.random_scale_list)
                H_s: int
                W_s: int
                H_s, W_s, _ = img_GT.shape

                def _mod(n: int, random_scale: int, scale: int, thres: int) -> int:
                    rlt: int = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(
                    np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR
                )

                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # TODO: resize (img_GT, 1 / scale)

        # augmentation
        if self.opt["phase"] == "train":
            H, W, _ = img_GT.shape

            # ??? 이것도 왜 하는걸까
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(
                    np.copy(img_GT), (GT_size, GT_size), interpolation=cv2.INTER_LINEAR
                )
                # TODO: resize (img_GT, 1 / scale)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size: int = GT_size // scale

            # randomly crop
            rnd_h: int
            rnd_w: int
            rnd_h_GT: int
            rnd_w_GT: int

            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[
                rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
            ]

            img_LQ, img_GT = util.augment(
                [img_LQ, img_GT], bool(self.opt["use_flip"]), bool(self.opt["use_rot"])
            )

        # BGR to RGB
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))
        ).float()

        if LQ_path is None:
            LQ_path = GT_path

        return {"LQ": img_LQ, "GT": img_GT, "LQ_path": LQ_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.paths_GT)
