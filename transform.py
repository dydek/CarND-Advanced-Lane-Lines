import os

import cv2
import glob
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def image_bird_view(image, debug=False):
    height, width = image.shape[:2]
    vertices = np.float32(
        [
            [width * 0.4, height * 0.65],
            [width * 0.6, height * 0.65],
            [width * 0.9, height * 0.95],
            [width * 0.1, height * 0.95]
        ]
    )
    if debug:
        cv2.polylines(image, np.int32([vertices]), True, (0, 255, 255), 3)

        plt.imshow(image)
        plt.waitforbuttonpress()

    pts2 = np.float32([
        [width * 0.1, 0],
        [width * 0.9, 0],
        [width * 0.9, height],
        [width * 0.1, height]
    ])
    M = cv2.getPerspectiveTransform(vertices, pts2)
    dst = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return dst


def transform_images():
    for img_path in glob.glob('test_images_with_threshold/*.jpg'):
        img = cv2.imread(img_path)
        img_processed = image_bird_view(img)
        cv2.imwrite(
            img_path.replace(
                os.path.split(img_path)[0], 'test_images_bird_view'
            ),
            img_processed
        )


if __name__ == '__main__':
    transform_images()
