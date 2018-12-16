import random
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def contours_(im: np.ndarray) -> np.ndarray:
    ctr = []
    width, height = im.shape[:2]
    x = 0
    y = 0
    dir = 0
    while im[y, x]:
        x += 1
        if x >= width:
            x = 0
            y += 1
        temp_dir = (dir + 3) % 4
        ctr.append((x, y))
    return ctr


def contours(im: np.ndarray) -> List[np.ndarray]:
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    largest_contour = contours[0]
    res = []
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    for c in contours:
        area = cv2.contourArea(c)
        if area > cv2.contourArea(largest_contour):
            largest_contour = c
            res.append(largest_contour)
    # epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    # approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    # print(approx)
    # return sorted(res, key=lambda x: cv2.contourArea(x))
    slc = len(contours) % 10 if len(contours) % 10 else 1
    return contours


def point_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def vec_angle(point: Tuple[float, float], vector: Tuple[float, float]) -> float:
    x_diff = np.abs(vector[0] - point[0])
    y_diff = np.abs(vector[1] - point[1])
    return np.arctan(x_diff / y_diff) if y_diff else 0


def curvature(contour: np.ndarray, k=1) -> List[float]:
    curvatures = [0.0 for _ in range(len(contour))]
    cnt_len = len(contour)
    for i in range(cnt_len):
        point = contour[i][0]
        fwd_vec_ind = i + k if i + k < cnt_len else cnt_len - 1
        bwd_vec_ind = i - k if i - k > 0 else 0
        fwd_vec = contour[fwd_vec_ind][0]
        bwd_vec = contour[bwd_vec_ind][0]
        fwd_dist = point_dist(fwd_vec, point)
        bwd_dist = point_dist(bwd_vec, point)
        fwd_angle = vec_angle(point, fwd_vec)
        bwd_angle = vec_angle(point, bwd_vec)
        angle_diff = fwd_angle - ((fwd_angle + bwd_angle) / 2)
        curvatures[i] = 0.5 * angle_diff * (fwd_dist + bwd_dist) / (fwd_dist * bwd_dist)
    return curvatures


def pltcrv(curvature: List[float]) -> None:
    plt.plot(curvature)
    plt.show()


if __name__ == '__main__':
    img_name = "assets/snow.png"
    img = cv2.imread(img_name)
    # crs = contours(img)[-2]
    # crv = curvature(crs, 20)
    # cv2.drawContours(img, crs, -1, (0, 0, 255), 2)
    ctrs = contours(img)[-10:]
    [cv2.drawContours(img, crs, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2) for i, crs in
     enumerate(ctrs, 1)]
    [pltcrv(curvature(crs, 10)) for crs in ctrs]
    cv2.imshow("original", img)
    # plt.plot(crv)
    # plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
