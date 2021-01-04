import cv2  # type: ignore
import numpy as np  # type: ignore
import random
from typing import Tuple, List, Any


def get_image_paths(data_type: str, dataroot: str) -> Tuple[List[str], int]:
    # TODO
    a: List[str] = ["aa", "a"]
    b: int = 10
    return a, b


def read_img(path: str) -> Any:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.0

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def modcrop(img_in: Any, scale: int) -> Any:
    img: Any = np.copy(img_in)

    H: int
    W: int
    C: int
    H_r: int
    W_r: int

    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r]
    else:
        raise ValueError("Wrong img ndim: [{:d}].".format(img.ndim))
    return img


def augment(img_list: List[Any], hflip: bool = True, rot: bool = True) -> List[Any]:
    hflip = hflip and random.random() < 0.5
    vflip: bool = rot and random.random() < 0.5
    rot90: bool = rot and random.random() < 0.5

    def _augment(img: Any) -> Any:
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]
