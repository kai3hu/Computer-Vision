import cv2

img = cv2.imread('./image00.png')    # BGR-image
tmp = img[:, :, 0]     # Temporary buffer for storing B channel
img[:, :, 0] = img[:, :, 2]    # Copy R channel to B channel
img[:, :, 2] = tmp    # Copy back stored B channel to R channel

cv2.imshow('Window', img)
cv2.waitKey()
cv2.destroyAllWindows()
