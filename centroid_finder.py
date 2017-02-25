import numpy as np
import cv2

class centroid_finder():
	def __init__(self, win_width, win_height, margin, smooth_factor=15):
		# List that stores centroid values for smoothing output
		self.past_centroids = []

		# Window parameters, number in pixels
		self.win_width = win_width
		self.win_height = win_height

		# Pixel distance to slide window for searching
		self.margin = margin

		# Smooth centroid values over how many frames
		self.smooth_factor = smooth_factor

	# Function for finding and storing lane segment positions
	def find_window_centroids(self, warped):
		# Store the (left,right) window centroid positions per level
		window_centroids = [] 
		# Create our window template that we will use for convolutions    
		window = np.ones(self.win_width) 

		# First find the two starting positions for the left and right lane
		# by using np.sum to get the vertical image slice,
		# and then np.convolve the vertical image slice with the window template 

		# Sum quarter bottom of image to get slice, could use a different ratio
		l_sum = np.sum(warped[int(3*warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
		l_center = np.argmax(np.convolve(window, l_sum)) - self.win_width/2
		r_sum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
		r_center = np.argmax(np.convolve(window, r_sum)) - self.win_width/2 + int(warped.shape[1]/2)

		# Add what we found for the first layer
		window_centroids.append((l_center, r_center))

		# Go through each layer looking for max pixel locations
		for level in range(1, (int)(warped.shape[0]/self.win_height)):
        
			# Convolve the window into the vertical slice of the image
			row_high = int(warped.shape[0] - (level+1) * self.win_height)
			row_low  = int(warped.shape[0] - level * self.win_height)
			image_layer = np.sum(warped[row_high : row_low, :], axis=0)
			conv_signal = np.convolve(window, image_layer)

			# Find the best left centroid by using past left center as a reference.
			# Use self.win_width/2 as offset because convolution signal reference is
			# at the right side of window, not center of window.
			offset = self.win_width/2
			l_min_index = int(max(l_center + offset - self.margin, 0))
			l_max_index = int(min(l_center + offset + self.margin, warped.shape[1]))
			l_center = np.argmax(conv_signal[l_min_index : l_max_index]) + l_min_index - offset

			# Find the best right centroid by using past right center as a reference
			r_min_index = int(max(r_center + offset - self.margin, 0))
			r_max_index = int(min(r_center + offset + self.margin, warped.shape[1]))
			r_center = np.argmax(conv_signal[r_min_index : r_max_index]) + r_min_index - offset

			# Add what we found for that layer
			window_centroids.append((l_center, r_center))

		# Add all centroids found in this frame to past_centroids list
		# Store limit is set by smooth factor
		if len(self.past_centroids) == self.smooth_factor:  
			self.past_centroids.pop(0)
		self.past_centroids.append(window_centroids)

	# Function for smoothing centroids over past frames
	def get_smoothed_centroids(self, warped):
		# Find centroids, i.e. lane segment for one frame
		self.find_window_centroids(warped)
		# Return averaged centroid values over frames
		# that will prevent line markers from jumping around too much
		return np.average(self.past_centroids[-self.smooth_factor:], axis=0)























