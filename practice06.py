import numpy as np
import cv2
from useful import *


def erosion(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    blank = np.zeros(img.shape)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if img[i, j + 1] and img[i, j - 1] and img[i + 1, j] and img[i - 1, j]:
                blank[i, j] = 255
    return blank


if __name__ == '__main__':
    path = "/Users/nickmarhal/PycharmProjects/ImageProcessing01/assets/lena.png"
    source = cv2.imread(path, 0)
    _, source = cv2.threshold(source, 127, 255, cv2.THRESH_BINARY)
    eroded = erosion(source)
    cv2.imshow("original", source)
    cv2.imshow("eroded", eroded)
    cv2.imshow("contour", source - eroded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
