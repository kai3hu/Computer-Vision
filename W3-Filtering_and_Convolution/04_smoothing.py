import cv2
import numpy as np

img = cv2.imread('./image00.png', 0)  # Read out as a gray scale image
img = img.astype(float)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9.0
# or, kernel = np.ones((3, 3)) / 9.0
filtered = cv2.filter2D(img, -1, kernel)

cv2.imshow('Input', img.astype(np.uint8))
cv2.imshow('Filtered', filtered.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
