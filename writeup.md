## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration]: ./writeup_images/calibration.png "Undistorted"
[distortion-corrected]: ./writeup_images/test1.jpg "Road Transformed"
[threshold-gif]: ./writeup_images/project_video_threshold.gif "Binary Example"
[image_foi]: ./writeup_images/foi.png "Warp Example"
[image_bird_view]: ./writeup_images/bird_view.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./writeup_images/lines_back.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./lane_line.ipynb"  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][calibration]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][distortion-corrected]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I've used these steps ( all process is available to see in the file `lane_line.ipdb` )

- `abs_sobel_thresh` function with kernel 7, using `x` direction with `(30, 250)`
- `abs_sobel_thresh` function with kernel 7, using `y` direction with `(15, 250)`
- change image to HLS, get only the saturation channel and select only values from `170` to `250`

Here's an example of my output for this step. Full movie is added to this repo too. ( the full video available here -> `project_video_threshold.mp4` )

![alt text][threshold-gif]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I'm transforming each image inside the `ProcessWrapper` class. This is done in the first few lines of the method `process`:

```python
undistort_img = undistort(img)
combined = image_pipeline_final(undistort_img)
# perspective transform
combined_warp = cv2.warpPerspective(
            combined, transform, (width, height), flags=cv2.INTER_LINEAR
        )
```

```python

source_vertices = np.float32(
        [
            [width * 0.44, height * 0.64],
            [width * 0.59, height * 0.64],
            [width * 0.9, height * 0.95],
            [width * 0.17, height * 0.95]
        ]
    )
destination_vertices = np.float32([
        [width * 0.1, 0],
        [width * 0.9, 0],
        [width * 0.9, height],
        [width * 0.1, height]
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 563  460     | 128    0        | 
| 755  460      | 1152    0     |
| 1152  684    | 1152  720      |
| 217  684     | 128  720        |

I verified that my perspective transform was working as expected by drawing the `source_vertices ` and `destination_vertices ` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image_foi]

![alt text][image_bird_view]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


I did it using functions `find_lane_pixels` and `search_around_poly`. The first one is trying to find the lines from the scratch - so I'm calling it only once at the beginnign of the processing ( but the fact is that it should be called also when we lost the correct data for our lines ). After that the next calls goes to the second functions. In the both functions I calculate `left_fit` and `right_fit` parameter, along with `left_fit_m` and `left_fit_m` which are scaled.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I'm doing this in the last few cells of my jupyter notebook, using functions named `calculate_radius` and `calculate_deviation`:

```python

def calculate_radius(y_eval, left_fit_m, right_fit_m):    
    left_radius = ((1 + (2*left_fit_m[0]*y_eval + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
    right_radius = ((1 + (2*right_fit_m[0]*y_eval + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])
    
    return left_radius, right_radius


def calculate_deviation(y_eval_pix, width_pix, left_fit_m, right_fit_m):
    x1 = left_fit_m[0]*(y_eval_pix * ym_per_pix)**2 + left_fit_m[1]*30 + left_fit_m[2]
    x2 = right_fit_m[0]*(y_eval_pix * ym_per_pix)**2 + right_fit_m[1]*30 + right_fit_m[2]
    return (x2 + x1) * 0.5 - (width_pix * xm_per_pix) * 0.5

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

```python

def draw_line(img, M, left_fit, right_fit):
    img_width = img.shape[1]
    img_height = img.shape[0]
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    ploty = np.linspace(0, img_height, num=10)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, M, (img_width, img_height))
    result = cv2.addWeighted(img, 1, newwarp, 0.4, 0)
    
    return result

```

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


- guessing what kind of the image transforming would be the best for most cases ( and it's still not perfect )
- writing the code for radius and center 
