import numpy as np
import cv2
import glob
import pickle

# Arrays to store object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Make a list of calibration images
images = glob.glob("camera_cal/calibration*.jpg")

# Step through calibration images and search for chessboard corners
for i, fname in  enumerate(images):
	# Read in each image
    img = cv2.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If corners are found, add object points, image points
    if ret == True:
    	print("Working on {}".format(fname.split('/')[-1]))

    	imgpoints.append(corners)
    	objpoints.append(objp)

    	# Draw the corners
    	cv2.drawChessboardCorners(img, (9, 6), corners, ret)

    	# Write output image
    	write_fname = "output_images/corners_found_" + str(i) + ".jpg"
    	cv2.imwrite(write_fname, img)

# Find image size
img_size = (img.shape[1], img.shape[0])

# Calibrate camera given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save calibration result into pickle file
dist_pickle = {'mtx': mtx, 'dist': dist}
with open('calibration_pickle.p', 'wb') as f:
	pickle.dump(dist_pickle, f)

print("\nCalibration result saved.")

# Undistort calibration images
images_with_corners = glob.glob("output_images/corners_found_*.jpg")
for i, fname in  enumerate(images_with_corners):
    file_id = fname.split('/')[-1][:-4].split('_')[-1]
    print("Undistorting calibration {}".format(file_id))
    img = cv2.imread(fname)

    # Undistort image with mtx and dist result
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    write_fname = "output_images/undistorted_cal_" + str(file_id) + ".jpg"
    cv2.imwrite(write_fname, undistorted)

print("\nCalibration test done.")














