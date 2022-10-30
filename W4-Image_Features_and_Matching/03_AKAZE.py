import cv2

img = cv2.imread('image00.png', 0)
akaze = cv2.AKAZE_create()
kp = akaze.detect(img, None)

img = cv2.drawKeypoints(img, kp, None,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('AKAZE features', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

