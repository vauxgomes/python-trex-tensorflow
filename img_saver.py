#!/usr/bin/env python

"""
	Use this code to build your training set

	Note: Create a folder called "tmp" in this directory
"""

# Imports
import cv2 as cv
import numpy as np
import datetime

# Docs
__author__ = 'Vaux Gomes'
__version__ = '1.0.0'

#			X	Y	W	H	X+W	Y+H
#  ---------------------------------
#	LEFT:	50	125	185	185	235	330
#	CENTER:	235	125	185	205	420	330
#	RIGHT:	420	125	185	205	605	330
#  ----------2---0-----------3---1--

# Camera
camera = cv.VideoCapture(0)

#
dims = [185, 185] # W,H
position = [235, 125] # X,Y

# Mouse
def click(event, x, y, flags, param):
	global position, dims

	# Set crop position
	if event == cv.EVENT_LBUTTONDOWN:
		position = [int(x - dims[0]/2), int(y - dims[1]/2)]

	# Set crop size
	elif event == cv.EVENT_MOUSEWHEEL:
		if flags < 0:
			position = [position[0] + 5, position[1] + 5]
			dims = [dims[0] - 10, dims[1] - 10]
		else:
			position = [position[0] - 5, position[1] - 5]
			dims = [dims[0] + 10, dims[1] + 10]

# Main loop
while(1):
	# Take each frame
	_, frame = camera.read()

	# Crop
	cropped = frame[position[1]:position[1] + dims[1], position[0]:position[0] + dims[0]]

	# Keys
	key = cv.waitKey(1)
	if key == ord('s'):
		st = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
		sample = "tmp/img_{}.jpg".format(st)
		cv.imwrite(sample, cropped)
		
		print("Saved:", sample)
	elif key == ord('q'):
		break

	# Rectangle
	cv.rectangle(frame, tuple(position), (position[0] + dims[1], position[1] + dims[0]), (220,220,220), 1)

	# Show images
	cv.imshow('frame',frame)
	#cv.imshow('cropped',cropped)

	# Mouse
	cv.setMouseCallback("frame", click)
	
# Ending interaction
camera.release()
cv.destroyAllWindows()