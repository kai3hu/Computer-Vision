import cv2
import numpy as np


def gauss2dfilter(fsize, sigma):
    if fsize % 2 == 0:
        fsize += 1
    print('Filter size:', fsize)
    x, y = np.mgrid[-fsize//2+1:fsize//2+1, -fsize//2+1:fsize//2+1]
    g = 1.0 / (2.0 * np.pi * sigma**2) * np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


if __name__ == '__main__':
    img = cv2.imread('./image00.png', 0)  # Read out as a gray scale image
    img = img.astype(float)
    g = gauss2dfilter(51, 5)
    filtered = cv2.filter2D(img, -1, g)
    cv2.imshow('Input', img.astype(np.uint8))
    cv2.imshow('Filtered', filtered.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
