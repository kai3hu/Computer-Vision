import cv2

img = cv2.imread('./image00.png')
cv2.imshow('Window', img)
cv2.waitKey()
cv2.destroyAllWindows()
