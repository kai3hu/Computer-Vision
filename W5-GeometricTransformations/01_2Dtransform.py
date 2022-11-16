import numpy as np
import cv2


def translation_matrix(tx, ty):
    return np.array([[1, 0, tx],
                    [0, 1, ty],
                    [0, 0, 1]], np.float32)


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0, 0, 1]], np.float32)


def affine_matrix(a, b, c, d, e, f):
    return np.array([[a, b, c],
                    [d, e, f],
                    [0, 0, 1]], np.float32)


def homography_matrix(a, b, c, d, e, f, g, h):
    return np.array([[a, b, c],
                    [d, e, f],
                    [g, h, 1]], np.float32)


def warpImage(img, H):
    '''
    Warps img with a 3x3 matrix H
    '''
    h1, w1 = img.shape[:2]
    pts1 = np.array([[[0, 0], [0, h1], [w1, h1], [w1, 0]]], dtype=np.float32)
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = pts1_[0, :, :]
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5, dtype=np.float32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5, dtype=np.float32)
    t = np.array([-xmin, -ymin], dtype=int)
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)  # translate
    result = cv2.warpPerspective(img, Ht@H, (int(xmax - xmin), int(ymax - ymin)))

    return result




if __name__ == '__main__':
    img = cv2.imread('vegas.png')
    rows, cols, ch = img.shape

    # Perspective transformation example
    pts1 = np.float32([[0, 0], [rows , 0], [0, cols], [rows, cols - 1]])
    pts2 = np.float32([[20, 20], [rows, 0], [0, cols ], [rows, cols - 1]])
    H = cv2.getPerspectiveTransform(pts1, pts2)

    # Rotation example
#    theta = 30.0 / 180.0 * np.pi
#   H = rotation_matrix(theta)

    # Warp image
    dst = warpImage(img, H)

    # Visualize
    cv2.imshow('Original', img)
    cv2.imshow('Warped', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
