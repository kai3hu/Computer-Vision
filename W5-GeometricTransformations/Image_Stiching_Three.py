import numpy as np
import cv2


def warpleft(img1, img2, H):
    # read image size
    h1, w1 = img1.shape[:2] 
    h2, w2 = img2.shape[:2] 
    # define the four corners of the image
    pts1 = np.array([[[0, 0], [0, h1], [w1, h1], [w1, 0]]], dtype=np.float32) 
    pts2 = np.array([[[0, 0], [0, h2], [w2, h2], [w2, 0]]], dtype=np.float32)
    pts2_ = cv2.perspectiveTransform(pts2, H) 
    #concatenate the four corners
    pts = np.concatenate((pts1[0, :, :], pts2_[0, :, :]), axis=0) #
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5, dtype=np.float32) 
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5, dtype=np.float32)
    # translation matrix
    t = np.array([-xmin, -ymin], dtype=np.int) 
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32) 
    result = cv2.warpPerspective(img1,  Ht@H, (int(xmax-xmin), int(ymax-ymin)))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img2 # copy img2 to the warped img1
    return result
    
  
def warpright(img1, img2, H):
    # read image size
    h1, w1 = img1.shape[:2] 
    h2, w2 = img2.shape[:2] 
    pts1 = np.array([[[0, 0], [0, h1], [w1, h1], [w1, 0]]], dtype=np.float32) 
    pts2 = np.array([[[0, 0], [0, h2], [w2, h2], [w2, 0]]], dtype=np.float32)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1[0, :, :], pts2_[0, :, :]), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5, dtype=np.float32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5, dtype=np.float32)
    result= cv2.warpPerspective(img2, H, (int(xmax-xmin), int(ymax-ymin)))
    result[0:h1, 0:w1] = img1
    # crop the black border
    rows, cols = np.where(result[:, :,0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    return result[min_row:max_row, min_col:max_col] 


if __name__=='__main__':
    img1 = cv2.imread('kyoto01.jpg')  # Image 1
    img2 = cv2.imread('kyoto02.jpg')  # Image 2
    img3 = cv2.imread('kyoto03.jpg')  # Image 3
    
    # Initiate AKAZE detector
    akaze = cv2.AKAZE_create()
    # find the keypoints and descriptors with AKAZE
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    kp3, des3 = akaze.detectAndCompute(img3, None)
    # Compute matches_left
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_left = matcher.match(des1, des2)
    matches_left = sorted(matches_left, key=lambda x: x.distance)  # sort by good matches_left

    # compute homography
    left_pts = np.float32([kp1[m.queryIdx].pt for m in matches_left]).reshape(-1, 1, 2)
    mid_pts = np.float32([kp2[m.trainIdx].pt for m in matches_left]).reshape(-1, 1, 2)
    H_left, _ = cv2.findHomography(left_pts, mid_pts, cv2.RANSAC, 5.0)

    # warp left images
    left = warpleft(img1, img2, H_left)

    # find the keypoints and descriptors with AKAZE
    kpleft, desleft = akaze.detectAndCompute(left, None)

    # Compute matches_right
    matches_right = matcher.match(desleft,des3)
    matches_right = sorted(matches_right, key=lambda x: x.distance)
    right_pts = np.float32([kpleft[m.queryIdx].pt for m in matches_right[:100]]).reshape(-1, 1, 2)
    left_pts = np.float32([kp3[m.trainIdx].pt for m in matches_right[:100]]).reshape(-1, 1, 2)
    out = cv2.drawMatches(left, kpleft,img3, kp3, matches_right[:100], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # compute homography
    H_right, _ = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, 5.0)
    # warp right images
    result = warpright(left, img3, H_right)

    # show result
    cv2.imshow('out', out)
    cv2.imshow('Right', left)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

