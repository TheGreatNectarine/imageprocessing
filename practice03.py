"""
TASKS:
IMPLEMENT GAUSSIAN BLUR KERNEL ✔️
APPLY GAUSSIAN BLUR ✔️
"""

import cv2
import numpy as np
# import matplotlib.pyplot as plt
import math
from useful import *


def kernel(size: int, sigma: float) -> np.ndarray:
    return cv2.getGaussianKernel(size, sigma)


def apply(src, ksize, sigma_x) -> np.ndarray:
    return cv2.GaussianBlur(src, ksize, sigma_x)


def _kernel(size, sigma):
    size = int(size)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 / sigma + y ** 2 / sigma))
    return g / g.sum()


def _apply(img: np.ndarray, krnl: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    kh, kw = krnl.shape[:2]
    blank = img.copy()
    for i in range(kh // 2, height - (kh // 2)):
        for j in range(kw // 2, width - (kw // 2)):
            blank[i, j] = np.sum(img[i - kh // 2:i + kh // 2 + 1, j - kh // 2:j + kh // 2 + 1] * krnl)
    return blank


def main():
    path = "/Users/nickmarhal/PycharmProjects/ImageProcessing01/assets/chair.png"
    source = cv2.imread(path, 0)
    cv2.imshow("image", source)
    krnl = _kernel(3, 1.5)
    print(krnl)
    # plt.imshow(krnl, cmap=plt.get_cmap('inferno'), interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    cv2.imshow("blur", _apply(source, krnl))
    # three_times = apply(apply(apply(source, kernel), kernel), kernel)
    # cv2.imshow("3", three_times)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
