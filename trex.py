#!/usr/bin/env python

"""
	Use this code to build your training set

	Note: Create a folder called "tmp" in this directory
"""

# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import numpy as np
import argparse
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # This is specific for my PC

from pynput.keyboard import Key, Controller

# Docs
__author__ = 'Vaux Gomes'
__version__ = '1.0.0'

#
dims = [185, 185] # W,H
position = [235, 125] # X,Y

#
keyboard = Controller()

# Mouse event
def click(event, x, y, flags, param):
	global position, dims

	if event == cv.EVENT_LBUTTONDOWN:
		position = [int(x - dims[0]/2), int(y - dims[1]/2)]
	elif event == cv.EVENT_MOUSEWHEEL:
		if flags < 0:
			position = [position[0] + 5, position[1] + 5]
			dims = [dims[0] - 10, dims[1] - 10]
		else:
			position = [position[0] - 5, position[1] - 5]
			dims = [dims[0] + 10, dims[1] + 10]

# 
def load_graph(model_file):
	graph = tf.Graph()

	with tf.gfile.FastGFile(model_file, 'rb') as handler:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(handler.read())
		
		with graph.as_default():
			tf.import_graph_def(graph_def)

	return graph

#
def load_labels(label_file):
	labels = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()		

	for l in proto_as_ascii_lines:
		labels.append(l.rstrip())

	return labels

# CV.PutText
def put_text(img, text, x=10):
	cv.putText(img, text, (10, x), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv.LINE_AA) 

#
def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument("--graph", help="graph/model to be executed", required=True)
	parser.add_argument("--labels", help="name of file containing labels", required=True)
	
	return parser.parse_args()

# Main
if __name__ == "__main__":
	# Parsing arguments
	args = parse_arguments()
	
	# Building mode
	input_height = 299
	input_width = 299

	input_layer = "Placeholder"
	output_layer = "final_result"

	graph = load_graph(args.graph)
	labels = load_labels(args.labels)

	input_operation = graph.get_operation_by_name("import/" + input_layer)
	output_operation = graph.get_operation_by_name("import/" + output_layer)
	
	# Camera
	video_capture = cv.VideoCapture(0)

	#
	c = 0
	with tf.Session(graph=graph) as sess:
		while(1):

			# Trying to save processing
			c += 1
			if c < 5: continue
			else: c = 0

			# Take each frame
			_, frame = video_capture.read() # 480x640

			# Crop
			cropped = frame[position[1]:position[1] + dims[1], position[0]:position[0] + dims[0]]

			# Feed image
			feed = cv.resize(cropped, (299, 299), interpolation=cv.INTER_CUBIC)
			feed = np.asarray(feed)
			feed = cv.normalize(feed.astype('float'), None, -.5, .5, cv.NORM_MINMAX)
			feed = np.expand_dims(feed, axis=0)
			
			# Prediction
			results = sess.run(output_operation.outputs[0], { input_operation.outputs[0]: feed })
			y_hat = np.argmax(results[0])
			
			# Marks
			cv.rectangle(frame, tuple(position), (position[0] + dims[1], position[1] + dims[0]), (0,0,255), 2)
			put_text(frame, labels[np.argmax(results[0])] + ': ' + str(results[0]), 80)

			# Show images
			cv.imshow('frame',frame)

			# Mouse
			cv.setMouseCallback("frame", click)

			# Close the video_capture if 'q' is pressed
			if cv.waitKey(1) == ord('q'):
				sess.close()
				break

			# Action
			if labels[y_hat] == 'open' and results[0][y_hat] >= 0.77:
				keyboard.press(Key.up)
				keyboard.release(Key.up)

				print ("JUMP", results[0][y_hat])

	# Ending interaction
	video_capture.release()
	cv.destroyAllWindows()