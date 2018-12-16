"""
TASKS:
RESIZING IMAGES ✔️
GETTING GRAY-SCALE IMAGE AND CHANGING PIXEL VALUES ✔️
"""

import cv2
import numpy as np


def larger(old: np.ndarray, scl: int) -> np.ndarray:
    height, width = old.shape[:2]
    new_width, new_height = int(height * scl), int(width * scl)
    new = np.zeros((new_width, new_height, 3), np.uint8)
    for j in range(new_height):
        for i in range(new_width):
            new[i, j] = old[i // scl][j // scl]
    return new


def avg(block: np.ndarray):
    res = 0
    cnt = 0
    for row in block:
        for val in row:
            res += val
            cnt += 1
    return res / cnt


def smaller(old: np.ndarray, scl: int) -> np.ndarray:
    height, width = old.shape[:2]
    new_width, new_height = int(height * scl), int(width * scl)
    new = np.zeros((new_width, new_height), np.uint8)
    block_size = int(1 / scl)
    print(block_size)
    for j in range(new_height):
        for i in range(new_width):
            block = old[i * block_size:(i * block_size + block_size), j * block_size:(j * block_size + block_size)]
            pixel = avg(block)
            new[i, j] = pixel
    return new


def scale(image: np.ndarray, scl):
    if scl > 1:
        return larger(image, scl)
    elif scl < 1:
        return smaller(image, scl)
    else:
        return image


def greyscale(image: np.ndarray, upper, lower, change_upper, change_lower):
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            if image[x, y] > upper:
                image[x, y] = change_upper
            elif image[x, y] < lower:
                image[x, y] = change_lower
    return image


def main():
    path = "assets/bike.jpeg"

    img = cv2.imread(path, 0)
    cv2.imshow('image', img)
    img_resize = scale(img, 2)  # TODO scaling function
    cv2.imshow("image_res", img_resize)
    # img = greyscale(img, 150, 150, 0,10)
    # cv2.imshow('image2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
