import cv2
import numpy as np

filename = 'shapes00.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

cornerness = cv2.cornerHarris(gray, 2, 3, 0.04)
threshold = 0.01 * cornerness.max()
x, y = np.where(cornerness > threshold)

for (i, j) in zip(x, y):
    cv2.circle(img, (j, i), 4, [0, 0, 255])
cv2.imshow('Harris corner detector', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
