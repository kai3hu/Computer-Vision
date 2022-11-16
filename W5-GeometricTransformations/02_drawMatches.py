import numpy as np
import cv2


def warpTwoImages(img1, img2, H):
    '''
    This function will be used for Project 2.
    Warps img2 to img1 coordinates with a 3x3 homograph matrix H.
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.array([[[0, 0], [0, h1], [w1, h1], [w1, 0]]], dtype=np.float32)
    pts2 = np.array([[[0, 0], [0, h2], [w2, h2], [w2, 0]]], dtype=np.float32)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1[0, :, :], pts2_[0, :, :]), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5, dtype=np.float32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5, dtype=np.float32)
    t = np.array([-xmin, -ymin], dtype=np.int)
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32) 
    result = cv2.warpPerspective(img1,  Ht@H, (int(xmax - xmin), int(ymax - ymin)))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img2
    return result
  


if __name__=='__main__':
    img1 = cv2.imread('halfdome-00.png', 0)  # Image 1
    img2 = cv2.imread('halfdome-01.png', 0)  # Image 2
    # Initiate AKAZE detector
    akaze = cv2.AKAZE_create()
    # find the keypoints and descriptors with AKAZE
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    # Compute matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # sort by good matches
    # draw top 100 matches
    out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # compute homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # warp images
    result = warpTwoImages(img1, img2, H)

    cv2.imshow('Matched Features', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

