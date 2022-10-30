import cv2
import numpy as np

img = cv2.imread('./lena_saltandpepper.jpg',0)  # Read out as a gray scale image
img = img.astype(float)
NewImg = np.copy(img)

# or, kernel = np.ones((3, 3)) / 9.0
# filtered = cv2.filter2D(img, -1, kernel)``

W,H = img.shape[1],img.shape[0]
size = 3
for i in range(0,W-size,size) :
    for j in range(0,H-size,size):
        NewImg[j:j+size,i:i+size] = np.median(img[j:j+size,i:i+size]) # median filter
        


cv2.imshow('Input', img.astype(np.uint8))
cv2.imshow('New', NewImg.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
