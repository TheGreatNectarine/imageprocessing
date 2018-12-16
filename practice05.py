"""
TASKS: ✔️
IMPLEMENT CANNY ALGORITHM ✔️
"""
import cv2
import numpy as np
import practice04 as sobel


def non_maximum_suppression(grad: np.ndarray, phase: np.ndarray) -> np.ndarray:
    height, width = grad.shape[:2]
    blank = np.zeros(grad.shape, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if phase[i][j] < 0:
                phase[i][j] += 360

            if ((j + 1) < width) and ((j - 1) >= 0) and ((i + 1) < height) and ((i - 1) >= 0):
                # 0 degrees
                if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (157.5 <= phase[i][j] < 202.5):
                    if grad[i][j] >= grad[i][j + 1] and grad[i][j] >= grad[i][j - 1]:
                        blank[i][j] = grad[i][j]
                # 45 degrees
                elif (22.5 <= phase[i][j] < 67.5) or (202.5 <= phase[i][j] < 247.5):
                    if grad[i][j] >= grad[i - 1][j + 1] and grad[i][j] >= grad[i + 1][j - 1]:
                        blank[i][j] = grad[i][j]
                # 90 degrees
                elif (67.5 <= phase[i][j] < 112.5) or (247.5 <= phase[i][j] < 292.5):
                    if grad[i][j] >= grad[i - 1][j] and grad[i][j] >= grad[i + 1][j]:
                        blank[i][j] = grad[i][j]
                # 135 degrees
                elif (112.5 <= phase[i][j] < 157.5) or (292.5 <= phase[i][j] < 337.5):
                    if grad[i][j] >= grad[i - 1][j - 1] and grad[i][j] >= grad[i + 1][j + 1]:
                        blank[i][j] = grad[i][j]
    return blank


def thresholding(img: np.ndarray) -> np.ndarray:
    blank = np.zeros(img.shape)
    strong, weak = 1.0, 0.5
    max_ = np.max(img)
    lo, hi = 0.1 * max_, 0.2 * max_
    str_pixels = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            px = img[i][j]
            if px >= hi:
                blank[i][j] = strong
                str_pixels.append((i, j))
            elif px >= lo:
                blank[i][j] = weak

    res = np.zeros(img.shape, dtype=np.uint8)
    for st in str_pixels:
        res[st] = 255
        for i in range(max(0, st[0] - 1), min(st[0] + 1, img.shape[0])):
            for j in range(max(0, st[1] - 1), min(st[1] + 1, img.shape[1])):
                if blank[i][j] == weak:
                    res[i][j] = 255
    return res


def apply_canny(img: np.ndarray) -> np.ndarray:
    blank: np.ndarray = img.copy()
    blank = cv2.GaussianBlur(blank, (5, 5), 2)
    cv2.imshow("blur", blank)

    sobel_out, phase = sobel.apply_sobel(blank)
    cv2.imshow("sobel", sobel_out)

    non_max_out = non_maximum_suppression(sobel_out, phase)
    cv2.imshow("non-max", non_max_out)

    return thresholding(non_max_out)


def main():
    path = "/Users/nickmarhal/PycharmProjects/ImageProcessing01/assets/chair.png"
    source = cv2.imread(path, 0)
    filtered = apply_canny(source)
    cv2.imshow("original", source)
    cv2.imshow("post", filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
