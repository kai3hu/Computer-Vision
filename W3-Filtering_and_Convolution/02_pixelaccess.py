import cv2

img = cv2.imread('./image00.png')

x, y = 100, 200    # Pixel location
print(img[y, x])
