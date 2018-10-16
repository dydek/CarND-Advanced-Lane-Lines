import os

import numpy as np
import cv2
import glob


def image_histogram(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0).sum(axis=1)
    indexes = np.arange(len(histogram))

    # draw histogram point on the image
    points = np.int32([indexes, histogram / 300])
    points[1] = np.abs(points[1] - img.shape[0])
    points = [np.abs(points.T)]

    cv2.polylines(img, points, isClosed=False, color=[0, 255, 0], thickness=3)
    return img


def create_historgrams():
    for img_path in glob.glob('test_images_bird_view/*.jpg'):
        img = cv2.imread(img_path)
        img_processed = image_histogram(img)
        cv2.imwrite(
            img_path.replace(
                os.path.split(img_path)[0], 'test_images_histogram'
            ),
            img_processed
        )


if __name__ == '__main__':
    create_historgrams()
