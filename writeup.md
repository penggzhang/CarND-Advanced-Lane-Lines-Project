##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

[//]: # (Image References)

[image1]: ./output_images/corners_found_11.jpg "corners found"
[image2]: ./output_images/undistorted_cal_11.jpg "undistorted calibration"
[image3]: ./test_images/test1.jpg "test"
[image4]: ./output_images/undistorted_0.jpg "undistorted"
[image5]: ./output_images/thresholded_0.jpg "thresholded"
[image6]: ./output_images/warped_0.jpg "warped"
[image7]: ./output_images/window_masked_0.jpg "window masked"
[image8]: ./output_images/lanes_fit_0.jpg "fit"
[image9]: ./output_images/with_info_0.jpg "lanes on"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here are the rubric points for this project and I accordingly describe how each point was addressed in my implementation.  

---
###Camera Calibration

#####Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file of `camera_calibration.py`.  

Start with preparing object points, which will be the (x, y, z) coordinates of the chessboard corners in the world. Assume the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time chessboard corners in a test image were successfully detect.  `imgpoints` will be appended with the (x, y) pixel position of each of the `corners` in the image plane with each successful chessboard detection. `cv2.findChessboardCorners` function was used for corners detection. Images were converted into grayscale prior to the detection. And after detection, corners were drawn using `cv2.drawChessboardCorners` function as followed:

![alt text][image1]

Then `objpoints` and `imgpoints` were then used to compute the camera calibration `mtx` and distortion coefficients `dist` using `cv2.calibrateCamera()` function. `mtx` and `dist` were then saved into picke file `calibration_pickle.p`. 

The above example of calibration image was then undistorted using `cv2.undistort()` function to verify the correction was successfully done.

![alt text][image2]

###Pipeline (test images)

#####1. Provide an example of a distortion-corrected image.

Here is one test image.

![alt text][image3]

With the result of `mtx` and `dist`, distortion correction to test images was applied using `cv2.undistort()` function and the following results were obtained:

![alt text][image4]

The code were in lines `255~258`, `161` of `lane_finding.py`.

#####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

A combination of gradient and color thresholding was applied to generate a binary image in lines `10~61`, `164` and `261~264` in `lane_finding.py`.  Here's an example of my output for this step. 

![alt text][image5]

`cv2.Sobel` function was used to compute directional gradient. Then gradients were taken absolute values and scaled to 0~255 range. Thresholding was applied on the scaled absolute gradients. And then output a binary image. Lines `10~26` in `lane_finding.py`.

Color thresholding was performed on S and V channel from HLS and HSV color space respectively. Corresponding color space conversion was done, and thresholding was applied. Then a combination of these two channel thresholded pixels output a binary image. Lines `29~49` in `lane_finding.py`.

The x and y gradient thresholds were set as (12, 255), (25, 255) respectively. The s and v color thresholds were set as (100, 255), (50, 255) respectively. Lines `261~264` in `lane_finding.py`.

This step outputs a binary image `thresholded`. Lines `52~61` in `lane_finding.py`.

#####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform includes a function called `get_perspective_warp_matrice`, lines `64~86` in the file `lane_finding.py`.  This function takes as input an image size `img_size`.  

Parameters of trapezoid for perspective transformation were set as followed:

	top_width = 0.08 # percent of image width
	bot_width = 0.70 # percent of image width
	top_to_imgtop = 0.62 # percent of image height
	bot_to_imgbot = 0.065 # percent of image height


Source and destination points, clockwise starting from the top-left, were accordingly hardcoded in the following manner:

	src = np.float32([[img_size[0]*(0.5-top_width/2), img_size[1]*top_to_imgtop],
					  [img_size[0]*(0.5+top_width/2), img_size[1]*top_to_imgtop],
					  [img_size[0]*(0.5+bot_width/2), img_size[1]*(1-bot_to_imgbot)],
					  [img_size[0]*(0.5-bot_width/2), img_size[1]*(1-bot_to_imgbot)]])
	offset = img_size[0] * 0.25
	dst = np.float32([[offset, 0],
					  [img_size[0]-offset, 0],
					  [img_size[0]-offset, img_size[1]],
					  [offset, img_size[1]]])


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 588, 446      | 320, 0        | 
| 691, 446      | 960, 0        |
| 1088, 673     | 960, 720      |
| 192, 673      | 320, 720      |

Transformation matrix `M` and inverse matrix `M_inv` were then obtained through the function `cv2.getPerspectiveTransform`. Lines `84~85` in `lane_finding.py`. 

In order to avoid repeating computation, `flag_warp` was set to True once these matrices had been computed. Lines `267` and `167~169` in `lane_finding.py`. 

Then the undistorted image from last step was warped perspective as the output of this step, `warped`. Line `164` in `lane_finding.py`.

The perspective transform was working as expected as followed to verify that the lines appear parallel in the warped image.

![alt text][image6]

#####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

Then lane line pixels detection was implemented by defining a class `centroid_finder` in the file `centroid_finder.py`. Through this class's methods, sliding window search was applied to locate the centroids of lane line segments. Smoothed centroids found in images were output to pipeline for following procedures.

By the `find_window_centroids` function of this class, each warped image frame was segmented into several (in this case, 9) layers from top to bottom according to the class attribute of sliding window `win_height`. Then, `numpy.sum`, `numpy.convolve` and `numpy.argmax` functions were used to find the x coordinate where most '1' pixels appear at the 3/4 bottom of the warped image. The width of searching window is set by class attribute `win_width`. Left and right centroids are respectively searched in the left and right half of the image. The x coordinates of found centroids, as a tuple `(l_center, r_center)`, was appended to a list `window_centroids`.

Then, bottom up, layer by layer, centroids were found by the sum-convolve-argmax method. In this process, centroid positions in lower layer were used as a reference to find the best centroid in current layer. The range for searching is set accordingly by another class attribute, `margin`.

Once all centroids were found for all layers in one image frame, the list of `window_centroids` was appended to class variable `past_centroids`. `past_centroids` helps store all found centroids for a few past image frames, whose number is set by class attribute `smooth_factor`. This will eventually smooth the estimated position of centroids given a sequence of images and prevent line markers from jumping around too much.

The function `get_smoothed_centroids` of this class is to average centroid positions over the past few images and return the smoothed centroids.

Equipped with the `centroid_finder` class, the pipeline program instantiated a finder in lines `271~276` of `lane_finding.py`, called the `get_smoothed_centroids` method, line `174` in `lane_finding.py`, and thus obtained the smoothed `window_centroids` of lane line segments for current image frame.

To verify that line pixels have been correctly identified, `window_mask` and `draw_windows` functions in `lane_finding.py` were used to overlay a series of green windows on the warped image. An example is as followed:

![alt text][image7]

With these `window_centroids`, 2nd order polynomial fits were made for both left and right lane lines by `fit_line` helper function and the underlying `numpy.polyfit` function. Lines `181~191` and `125~133` in `lane_finding.py`. 

Then, lane line and inner area boundary points were generated by the functions of `get_line_boundary_points` and `get_area_boundary_points`. Lines `193~196` and `136~151` in `lane_finding.py`. 

The `left_lane` and `right_lane` were drawn as followed to verify the two lane lines are basically parallel and their fits were correct.

![alt text][image8]

#####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature of the lane was calculated in lines `235~239` in `lane_finding.py`. First, a sequence of lane center points were averaged out from the found left and right centroids. Then a new polynomial to these lane center points were fit by the `numpy.polyfit` function in world space, where meters per pix, `m_per_pix_y` and `m_per_pix_x`, were set as lines `220~221` of `lane_finding.py`. Finally the radius of curvature at the near end of vehicle was calculated as `curve_rad` in line `239` of `lane_finding.py`.

The position of the vehicle with respect to lane center was calculated in lines `221~230` of `lane_finding.py`. It is the offset from the lane center to the vehicle center, which is at the midpoint of image bottom. The relative position, left or right, to the lane center was also given once the offset was calculated.

#####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step was implemented in lines `201~215` and `243~246` of `lane_finding.py`.  Here is an example result on the test image:

![alt text][image9]

Lane lines and inner area were warped back by the function `cv2.warpPerspective` to original perspective, and then overlaid on the test image. In this step, the boundary points were used to draw the polygons by the function `cv2.fillPoly`, and the `cv2.addWeighted` function was used to overlay.

Eventually, main function `find_lane` of the pipeline outputs this `lanes_on` image.

---

###Pipeline (video)

#####Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to project video result](./project_video_output.mp4)

---

###Discussion

#####Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

As for the standard project video, the far ends of the detected lane were a bit wobbly when going over hard-recognized pavements. Overall performance on that video was still reasonably fine. 

However, for the challenge and harder challenge videos, current pipeline failed in various circumstances. And the fact told that more sophisticated algorithm and upgraded pipeline shall be worked out. 

Here are some points to make it more robust:

* Hard recognized pavement and shadowed area -- Shall improve the detection of correct lane line pixels at such circumstances. More gradient and color thresholding shall be tried out.
* Strong line of other type, but not lane line -- Such case happens in the challenge video. The median of highway was detected as a strong line, and incorrectly distracted the lane line estimate. More sophisticated color thresholding might be applied to solve this problem.
* Non-flat road -- If the road has slope, downhill or uphill, the perspective transformation changes accordingly. But at present, the trapezoid for transforamtion is hardcoded, and cannot dynamically change. A method to detect road slope shall be implemented and used to correct perspective transformation.
* Sharp turns -- In cases like harder challenge video, one lane line will disappear because of sharp turns. This is against the setup of current algorithm, which shall be updated to cater for such case.
