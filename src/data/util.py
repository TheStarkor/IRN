import cv2 # type: ignore
import numpy as np # type: ignore
from typing import Tuple, List, Any

def get_image_paths(data_type: str, dataroot: str) -> Tuple[List[str], int]:
    a: List[str] = ["aa", "a"]
    b: int = 10
    return a, b


def read_img(path: str) -> Any:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.


    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img