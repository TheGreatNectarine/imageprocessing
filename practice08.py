import cv2
import numpy as np


def apply_thrs(img: np.ndarray, threshold: float) -> np.ndarray:
    blank = img.copy()
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if img[i, j] <= threshold:
                blank[i, j] = 0
            else:
                blank[i, j] = 255
    return blank


def optimus(img: np.ndarray):
    avg = np.mean(img)
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    while True:
        avg0 = avg
        avg = (mu(histogram, 1, avg0) + mu(histogram, avg0, len(histogram))) / 2
        if avg == avg0:
            return avg


def mu(hist, lower, upper):
    lower = int(lower)
    upper = int(upper)
    sum1 = sum([i * hist[i] for i in range(lower, upper)])
    sum2 = sum([hist[i] for i in range(lower, upper)])
    return sum1 / sum2


if __name__ == '__main__':
    img_name = "assets/bike.jpeg"
    img = cv2.imread(img_name, 0)
    cv2.imshow("original", img)
    threshold = optimus(img)
    cv2.imshow("thrs", apply_thrs(img, threshold))
    cv2.waitKey()
    cv2.destroyAllWindows()
