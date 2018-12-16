"""
TASKS:
MAKE THE EQUALIZATION OF THE IMAGE HISTOGRAM ✔️
DRAW A HISTOGRAM OF AN IMAGE ✔️
"""

import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from useful import *
from typing import *


def histogram(image: np.ndarray) -> [int]:
    height, width = image.shape[:2]
    hist = [0 for _ in range(256)]
    for i in range(height):
        for j in range(width):
            hist[image[i, j]] += 1
    return hist


def cumulative_freq(hist: List[int]) -> List[int]:
    return [sum(hist[:i + 1]) for i in range(len(hist))]


# cumulative_freq = lambda hist: [sum(hist[:i + 1]) for i in range(len(hist))]


def transform(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = width * height
    alpha = 255 / size
    transformed = blank_image(height, width)
    hist = histogram(image)
    cumsum = cumulative_freq(hist)
    for i in range(height):
        for j in range(width):
            transformed[i, j] = cumsum[image[i, j]] * alpha
    return transformed


def normalize(data: List, highest: int) -> List[int]:
    top = sorted(data)[len(data) - 1]
    scale = top // highest
    for i in range(len(data)):
        data[i] = data[i] // scale
    return data


# def draw_histogram(hist: List[int]) -> None:
#     blank = blank_image(256, 256)
#     hist = normalize(data=hist, highest=256)
#     for i in range(256):
#         blank[i, hist[i]] = hist[i]


def blue(image: np.ndarray) -> List[int]:
    return image[:, 0]


def green(image: np.ndarray) -> List[int]:
    return image[:, 1]


def red(image: np.ndarray) -> List[int]:
    return image[:, 2]


def main():
    path = "assets/cat.jpg"
    source = cv2.imread(path, 0)
    equalized = transform(source)
    cv2.imshow("image", source)
    cv2.imshow("equalized", equalized)
    # plt.hist(source.ravel(), 256, [0, 256], color="b")
    # plt.show()
    # plt.hist(equalized.ravel(), 256, [0, 256], color="r")
    # plt.show()
    image = cv2.imread(path)
    # draw_histogram(histogram(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
