import numpy as np
import cv2


img1 = cv2.imread('templeSR0001.png', 0)  # Image 1
img2 = cv2.imread('templeSR0002.png', 0)  # Image 2
# Initiate AKAZE detector
akaze = cv2.AKAZE_create()
# find the keypoints and descriptors with AKAZE
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)
# Compute matches
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)  # sort by good matches
# draw top 50 matches
out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matched Features', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
