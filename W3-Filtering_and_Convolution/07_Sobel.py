import cv2
import numpy as np

img = cv2.imread('./image00.png', 0)    # Read out as a gray scale image
img = img.astype(float)

gx = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]], float)

gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]], float)

hx = cv2.filter2D(img, -1, gx)
hy = cv2.filter2D(img, -1, gy)

h = np.sqrt(np.square(hx) + np.square(hy))
h = h / np.max(h) * 255.0  # re-scale

cv2.imshow('Input', img.astype(np.uint8))
cv2.imshow('Filtered', h.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
