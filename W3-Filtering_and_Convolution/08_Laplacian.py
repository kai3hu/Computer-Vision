import cv2
import numpy as np

img = cv2.imread('./image00.png', 0)    # Read out as a gray scale image
img = img.astype(float)
g = np.array([[0, -1, 0],
              [-1, 4, -1],
              [0, -1, 0]], float)

h = cv2.filter2D(img, -1, g)
h = (h - np.min(h)) / (np.max(h) - np.min(h)) * 255.0
cv2.imshow('Input', img.astype(np.uint8))
cv2.imshow('Filtered', h.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
