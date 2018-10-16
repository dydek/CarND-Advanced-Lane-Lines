import cv2
import glob

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def image_polynominal(img):
    out_img = out_img = np.dstack((img, img, img))

    height, width = img.shape
    # this is the part of the implementation
    # https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/096009a1-3d76-4290-92f3-055961019d5e/concepts/4dd9f2c2-1722-412f-9a02-eec3de0c2207
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    midpoint = np.int(width // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 30

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(height // nwindows)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        # this is for left frame
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        # this is for right frame
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin



        # Draw the windows on the visualization image

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 100), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###

        good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # getting avarage x-pixel for non-zero values

        if len(good_left_inds) > minpix:
            leftx_current = np.int(
                np.mean(nonzerox[good_left_inds])
            )

        if len(good_right_inds) > minpix:
            rightx_current = np.int(
                np.mean(nonzerox[good_right_inds])
            )
    #
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    #
    # # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    #

    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='blue')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.waitforbuttonpress()
    plt.clf()

    return ploty, left_fitx, right_fitx, left_fit, right_fit


def run_polynominals():
    for img_path in glob.glob('test_images_bird_view/*.jpg'):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_polynominal(img)


if __name__ == '__main__':
    run_polynominals()
