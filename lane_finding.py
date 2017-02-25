import numpy as np
import cv2
import glob
import pickle
from centroid_finder import centroid_finder
from moviepy.editor import VideoFileClip


# Helper function for absolute gradient thresholding
def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	# Convert image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Calculate absolute values of directional gradient
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

	# Scale the gradient
	scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

	# Apply threshold
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	return binary_output

# Helper function for color thresholding
def color_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):
	# Convert image to HLS color space
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	# Separate S channel
	s_channel = hls[:,:,2]
	# Apply threshold
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

	# Convert image to HSV color space
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# Separate V channel
	v_channel = hsv[:,:,2]
	# Apply threshold
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

	# Combine the thresholding of two channels
	output = np.zeros_like(s_channel)
	output[(s_binary == 1) & (v_binary == 1)] = 1
	return output

# Helper function for performing gradient and color thresholding
def threshold(img, x_thresh, y_thresh, s_thresh, v_thresh):
	# Gradient thresholding 
	grad_x = abs_sobel_threshold(img, orient='x', thresh=x_thresh)
	grad_y = abs_sobel_threshold(img, orient='y', thresh=y_thresh)
	# Color thresholding
	color_binary = color_threshold(img, sthresh=s_thresh, vthresh=v_thresh)
	# Generate thresholded image 
	thresholded = np.zeros_like(img[:,:,0])
	thresholded[((grad_x == 1) & (grad_y == 1) | (color_binary == 1))] = 255
	return thresholded

# Helper function for calculate perspective transformation matrices
def get_perspective_warp_matrices(img_size):	
	# Define source and destination points for perspective transformation
	# Trapezoid parameters
	top_width = 0.08 # percent of image width
	bot_width = 0.70 # percent of image width
	top_to_imgtop = 0.62 # percent of image height
	bot_to_imgbot = 0.065 # percent of image height
	# 4 source points as [x, y] clockwise starting from top-left
	src = np.float32([[img_size[0]*(0.5-top_width/2), img_size[1]*top_to_imgtop],
					  [img_size[0]*(0.5+top_width/2), img_size[1]*top_to_imgtop],
					  [img_size[0]*(0.5+bot_width/2), img_size[1]*(1-bot_to_imgbot)],
					  [img_size[0]*(0.5-bot_width/2), img_size[1]*(1-bot_to_imgbot)]])
	# Offset after transformed
	offset = img_size[0] * 0.25
	# 4 destination points
	dst = np.float32([[offset, 0],
					  [img_size[0]-offset, 0],
					  [img_size[0]-offset, img_size[1]],
					  [offset, img_size[1]]])
	# Warp prespective
	M = cv2.getPerspectiveTransform(src, dst)
	M_inv = cv2.getPerspectiveTransform(dst, src)
	return M, M_inv

# Helper function for drawing one window area based on the center
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height) : int(img_ref.shape[0]-level*height),
           max(0, int(center-width/2)) : min(int(center+width/2), img_ref.shape[1])] = 1
    return output

# Helper function for drawing diagnostic windows 
# given the found centroids of lane line segments
def draw_windows(window_centroids, warped):
	# Points used to draw all the left and right windows
	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)
	# Go through each level and draw the windows
	for level in range(0, len(window_centroids)):
		# Window_mask is a function to draw window areas
		l_mask = window_mask(win_width, win_height, warped, window_centroids[level][0], level)
		r_mask = window_mask(win_width, win_height, warped, window_centroids[level][1], level)
		# Add graphic points from window mask here to total pixels found 
		l_points[(l_points == 255) | ((l_mask == 1))] = 255
		r_points[(r_points == 255) | ((r_mask == 1))] = 255

	# Draw the results
	# Add both left and right window pixels together
	template = np.array(r_points + l_points, np.uint8) 
	# Create a zero color channle 
	zero_channel = np.zeros_like(template) 
	# Make window pixels green
	template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) 
	# Make the original road pixels 3 color channels
	warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) 
	# Overlay the orignal road image with window results
	output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)
	return output

# Helper function for fitting line and generating x values for the fit line
# given y and x values of input points
def fit_line(yvals, xvals, res_yvals):
	# Fit the line
	fit = np.polyfit(yvals, xvals, 2)
	# Recast y values to avoid data type warning
	res_yvals = np.array(res_yvals, np.float32) 
	# Generate x values from the fit
	fit_x = fit[0] * np.square(res_yvals) + fit[1] * res_yvals + fit[2]
	fit_x = np.array(fit_x, np.int32)
	return fit, fit_x

# Helper function for generating boundary points for a line
def get_line_boundary_points(xvals, yvals, win_width):
	# Concatenate left and right boundary points
	boundary_x = np.concatenate((xvals-win_width/2, xvals[::-1]+win_width/2), axis=0)
	boundary_y = np.concatenate((yvals, yvals[::-1]), axis=0)
	# Return the boundary points as a list
	boundary_pts = list(zip(boundary_x, boundary_y))
	return np.array(boundary_pts, np.int32)

# Helper function for generating boundary points for an area
def get_area_boundary_points(left_xvals, right_xvals, yvals, win_width):
	# Concatenate left and right boundary points
	boundary_x = np.concatenate((left_xvals+win_width/2, right_xvals[::-1]-win_width/2), axis=0)
	boundary_y = np.concatenate((yvals, yvals[::-1]), axis=0)
	# Return the boundary points as a list
	boundary_pts = list(zip(boundary_x, boundary_y))
	return np.array(boundary_pts, np.int32)

# Main function for processing each frame
def find_lane(img):
	global mtx, dist, x_thresh, y_thresh, s_thresh, v_thresh
	global flag_warp, M, M_inv, win_width, win_height, finder

	img_size = (img.shape[1], img.shape[0])

	### Undistort image ###	
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)

	### Gradient and color thresholding ### 
	thresholded = threshold(undistorted, x_thresh, y_thresh, s_thresh, v_thresh)

	### Warp perspective ###
	if flag_warp == False:
		M, M_inv = get_perspective_warp_matrices(img_size)
		flag_warp = True
	warped = cv2.warpPerspective(thresholded, M, img_size, flags=cv2.INTER_LINEAR)

	### Detect centroids of lane line segments ###
	# Use the finder instance to find centroids
	window_centroids = finder.get_smoothed_centroids(warped)
	# Draw diagnostic windows given the centroids
	#window_masked = draw_windows(window_centroids, warped)


	### Fit the centroids into lane lines ###
	# Points that are used to fit the left and right lanes
	left_x, right_x = [], []
	# Go through each level to add the centroids
	for level in range(0, len(window_centroids)):
		left_x.append(window_centroids[level][0])
		right_x.append(window_centroids[level][1])
	# Prepare y values
	yvals = range(0, warped.shape[0])
	res_yvals = np.arange(warped.shape[0]-(win_height/2), 0, -win_height)
	# Fit the lane line
	left_fit, left_fit_x = fit_line(res_yvals, left_x, yvals)
	rigght_fit, right_fit_x = fit_line(res_yvals, right_x, yvals)
	# Generate lane line boundary points
	left_lane = get_line_boundary_points(left_fit_x, yvals, win_width)
	right_lane = get_line_boundary_points(right_fit_x, yvals, win_width)
	# Generate inner lane boundary points
	inner_lane = get_area_boundary_points(left_fit_x, right_fit_x, yvals, win_width)


	### Warp lane line back to original perspective ###
	# Draw lane lines
	lanes = np.zeros_like(img)
	cv2.fillPoly(lanes, [left_lane], color=[255, 0, 0])
	cv2.fillPoly(lanes, [right_lane], color=[0, 0, 255])
	# Draw inner lane area
	cv2.fillPoly(lanes, [inner_lane], color=[0, 255, 0])
	# Make a lane background to accentuate the lanes
	lanes_bkg = np.zeros_like(img)
	cv2.fillPoly(lanes_bkg, [left_lane], color=[255, 255, 255])
	cv2.fillPoly(lanes_bkg, [right_lane], color=[255, 255, 255])
	# Warp back perspective
	lanes_warped_back = cv2.warpPerspective(lanes, M_inv, img_size, flags=cv2.INTER_LINEAR)
	lanes_bkg_warped_back = cv2.warpPerspective(lanes_bkg, M_inv, img_size, flags=cv2.INTER_LINEAR)
	# Draw the warped-back lanes on original image
	base = cv2.addWeighted(img, 1.0, lanes_bkg_warped_back, -1.0, 0.0)
	lanes_on = cv2.addWeighted(base, 1.0, lanes_warped_back, 0.7, 0.0)


	### Calculate the offset of the car from the center of lanes ###
	# Meters per pixel
	m_per_pix_y = 30./720
	m_per_pix_x = 3.7/570
	# Calculate lane center, i.e. the midpoint between left and right lanes
	lane_midpoint = (left_fit_x[-1] + right_fit_x[-1]) / 2
	# Calculate the offset between the lane center and car center
	# which is at the midpoint of image bottom
	car_offset = (lane_midpoint - warped.shape[1]/2) * m_per_pix_x
	# Get the car position relative to the lane center 
	side_pos = "left"
	if car_offset <= 0:
		side_pos = "right"


	### Calculate the curvature of the lane ###
	# Fit new polynomial to lane center in world space
	lane_center = (np.array(left_x) + np.array(right_x))/2
	fit_cr = np.polyfit(res_yvals*m_per_pix_y, lane_center*m_per_pix_x, 2)
	# Calculate radius of curvature
	y_eval = np.max(res_yvals)
	curve_rad = ((1 + (2*fit_cr[0]*y_eval*m_per_pix_y + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])


	# Draw text information of lane curvature and car offset
	cv2.putText(lanes_on, "Radius of Curvature = " + str(round(curve_rad, 3)) + "(m)",
				(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	cv2.putText(lanes_on, "Vehicle is " + str(abs(round(car_offset, 3))) + "(m) " + side_pos + " of lane center",
				(50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	
	# Write images
	#write_fname = "output_images/with_info_" + str(i) + '.jpg'
	#cv2.imwrite(write_fname, lanes_on)

	return lanes_on

# Read in the saved calibration result
with open("calibration_pickle.p", "rb") as f:
	dist_pickle = pickle.load(f)
mtx  = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Set up thresholding parameters
x_thresh = (12, 255)
y_thresh = (25, 255)
s_thresh = (100, 255)
v_thresh = (50, 255)

# Set up flag and variables for perspective transformation
flag_warp = False
M, M_inv = None, None

# Set up sliding window parameters for lane line detection
win_width  = 25
win_height = 80 
margin = 50
smooth_factor = 15
# Make a finder instance from centroid_finder class
finder = centroid_finder(win_width, win_height, margin, smooth_factor=smooth_factor)

# Set up input and output video
input_video = "project_video.mp4"
output_video = "project_video_output.mp4"

# Make the lane-on video
clip = VideoFileClip(input_video)
video_clip = clip.fl_image(find_lane)
video_clip.write_videofile(output_video, audio=False)


































