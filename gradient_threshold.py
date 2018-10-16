import glob
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def process_image_v1(img, threshold=(0, 255)):
    # convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    thresh_min = 40
    thresh_max = 120
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    #
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


def process_images():
    for image_path in glob.glob('test_images_corrected/*.jpg'):
        img = cv2.imread(image_path)

        img_processed = process_image_v1(img)

        cv2.imwrite(
            image_path.replace(
                os.path.split(image_path)[0], 'test_images_with_threshold'
            ),
            np.dstack((img_processed, img_processed, img_processed))*255
        )


if __name__ == '__main__':
    process_images()
