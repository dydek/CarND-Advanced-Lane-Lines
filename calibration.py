import pickle

import cv2
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from settings import BASE_DIR, CALIBRATION_IMAGES_DIR

X_CORNERS = 9
Y_CORNERS = 6


def run_calibration():
    objp = np.zeros((X_CORNERS * Y_CORNERS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:X_CORNERS, 0:Y_CORNERS].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    for image_dir in os.listdir(
            os.path.join(BASE_DIR, CALIBRATION_IMAGES_DIR)
    ):
        img = cv2.imread(
            os.path.join(BASE_DIR, CALIBRATION_IMAGES_DIR, image_dir)
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        found, corners = cv2.findChessboardCorners(gray, (X_CORNERS, Y_CORNERS))

        if found:
            obj_points.append(objp)
            img_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    pickle.dump(
        {'mtx': mtx, 'dist': dist},
        open(os.path.join(CALIBRATION_IMAGES_DIR, 'mtx_dist_pickle.p'), 'wb')
    )


if __name__ == '__main__':
    run_calibration()
