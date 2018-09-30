import os
import pickle

import cv2
import glob

SAVE_DIR = 'test_images_corrected'

MTX_DICT = pickle.load(open('camera_cal/mtx_dist_pickle.p', 'rb'))


def undistort(img):
    return cv2.undistort(img, MTX_DICT['mtx'], MTX_DICT['dist'], None, MTX_DICT['mtx'])


def run_undistort():
    for image_url in glob.glob('test_images/*.jpg'):
        img = cv2.imread(image_url)
        cv2.imwrite(
            image_url.replace(
                os.path.split(image_url)[0], SAVE_DIR
            ),
            undistort(img)
        )


if __name__ == '__main__':
    run_undistort()
