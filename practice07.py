import sys
import cv2
import numpy as np
from functools import *


def region_growing(img: np.ndarray, seed, threshold=1):
    frame_cnt = 0
    dims = img.shape[:2]
    reg = np.zeros(dims, dtype=img.dtype)

    # parameters
    mean_reg = float(img[seed[1], seed[0]])
    size = 1
    pix_area = dims[0] * dims[1]

    contour = []  # will be [ [[x1, y1], val1],..., [[xn, yn], valn] ]
    contour_val = []
    dist = 0
    # TODO: may be enhanced later with 8th connectivity
    orient = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 4 connectivity
    cur_pix = [seed[0], seed[1]]

    # Spreading
    while dist < threshold and size < pix_area:
        # try:
        # adding pixels
        for j in range(4):
            # select new candidate
            temp_pix = [cur_pix[0] + orient[j][0], cur_pix[1] + orient[j][1]]
            # check if it belongs to the image
            is_in_img = dims[0] > temp_pix[0] > 0 and dims[1] > temp_pix[1] > 0  # returns boolean
            # candidate is taken if not already selected before
            if is_in_img and (reg[temp_pix[1], temp_pix[0]] == 0):
                contour.append(temp_pix)
                contour_val.append(img[temp_pix[1], temp_pix[0]])
                reg[temp_pix[1], temp_pix[0]] = 150
        # add the nearest pixel of the contour in it

        dist = abs(int(np.mean(contour_val) if contour_val else 0) - mean_reg)

        dist_list = [abs(i - mean_reg) for i in contour_val]
        dist = 0 if not dist_list else min(dist_list)  # get min distance
        index = dist_list.index(0 if not dist_list else min(dist_list))  # mean distance index
        size += 1  # updating region size
        reg[cur_pix[1], cur_pix[0]] = 255

        # updating mean MUST BE FLOAT
        mean_reg = (mean_reg * size + float(contour_val[index])) / (size + 1)
        # updating seed
        cur_pix = contour[index]

        # removing pixel from neigborhood
        del contour[index]
        del contour_val[index]

        frame_cnt += 1
        if not frame_cnt % 1000:
            cv2.imshow("progress", reg)
            cv2.waitKey(1)
    # except:
    #     continue

    return reg


def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(
            'Seed: ' + str(x) + ', ' + str(y), img[y, x])
        clicks.append((x, y))


if __name__ == '__main__':
    clicks = []
    img = cv2.imread('assets/rickandmorty.png', 0)
    # ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('Input', img)
    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.waitKey()
    seed = clicks[-1]
    out = region_growing(img, seed, threshold=10)
    cv2.imshow('Region Growing', out)
    cv2.waitKey()
    cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
#
# def get8n(x, y, shape):
#     out = []
#     maxx = shape[1] - 1
#     maxy = shape[0] - 1
#
#     # top left
#     outx = min(max(x - 1, 0), maxx)
#     outy = min(max(y - 1, 0), maxy)
#     out.append((outx, outy))
#
#     # top center
#     outx = x
#     outy = min(max(y - 1, 0), maxy)
#     out.append((outx, outy))
#
#     # top right
#     outx = min(max(x + 1, 0), maxx)
#     outy = min(max(y - 1, 0), maxy)
#     out.append((outx, outy))
#
#     # left
#     outx = min(max(x - 1, 0), maxx)
#     outy = y
#     out.append((outx, outy))
#
#     # right
#     outx = min(max(x + 1, 0), maxx)
#     outy = y
#     out.append((outx, outy))
#
#     # bottom left
#     outx = min(max(x - 1, 0), maxx)
#     outy = min(max(y + 1, 0), maxy)
#     out.append((outx, outy))
#
#     # bottom center
#     outx = x
#     outy = min(max(y + 1, 0), maxy)
#     out.append((outx, outy))
#
#     # bottom right
#     outx = min(max(x + 1, 0), maxx)
#     outy = min(max(y + 1, 0), maxy)
#     out.append((outx, outy))
#
#     return out
#
#
# def region_growing(img, seed):
#     list = []
#     outimg = np.zeros_like(img)
#     list.append((seed[0], seed[1]))
#     processed = []
#     while (len(list) > 0):
#         pix = list[0]
#         outimg[pix[0], pix[1]] = 255
#         for coord in get8n(pix[0], pix[1], img.shape):
#             if img[coord[0], coord[1]] != 0:
#                 outimg[coord[0], coord[1]] = 255
#                 if coord not in processed:
#                     list.append(coord)
#                 processed.append(coord)
#             elif img[coord[0], coord[1]] == 0:
#                 outimg[coord[0], coord[1]] = 0
#                 if coord not in processed:
#                     list.append(coord)
#                 processed.append(coord)
#         list.pop(0)
#         cv2.imshow("progress",outimg)
#         cv2.waitKey(1)
#
#
# def on_mouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(
#             'Seed: ' + str(x) + ', ' + str(y), img[y, x])
#         clicks.append((y, x))
#
#
# clicks = []
# img = cv2.imread('assets/chair.png', 0)
# ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
# cv2.imshow('Input', img)
# cv2.namedWindow('Input')
# cv2.setMouseCallback('Input', on_mouse, 0, )
# cv2.waitKey()
# seed = clicks[-1]
# out = region_growing(img, seed)
# cv2.imshow('Region Growing', out)
# cv2.waitKey()
# cv2.destroyAllWindows()
