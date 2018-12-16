import numpy as np
from typing import *


def blank_image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), np.uint8)

# def blank_image(from_image: np.ndarray) -> np.ndarray:
#     return np.zeros((from_image[0], from_image[1], 3), np.uint8)
