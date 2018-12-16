import math

import const
import cv2
from useful import *
import practice03

const.sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
const.sobel_w = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def gradient(z1, z2, z3,
             z4, z5, z6,
             z7, z8, z9):
    gx = (z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)
    gy = (z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)
    grad = (gx ** 2 + gy ** 2) ** 0.5
    theta = np.arctan2(gy, gx) * 180 / math.pi
    return grad, theta


def apply_sobel(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = img.shape[:2]
    blank = img.copy()
    phase = img.copy()
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            grad = gradient(img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                            img[i, j - 1], img[i, j], img[i, j + 1],
                            img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1])
            blank[i, j] = min(grad[0], 255)
            phase[i, j] = grad[1]
    return blank, phase


def main():
    path = "/Users/nickmarhal/PycharmProjects/ImageProcessing01/assets/lena.png"
    source = cv2.imread(path, 0)
    gauss = practice03._kernel(3, 2)
    source = practice03._apply(source, gauss)
    # ret, source = cv2.threshold(source, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("image", source)
    cv2.imshow("w", apply_sobel(source)[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
