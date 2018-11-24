import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import transform as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from getBoundingBox import getBoundingBox
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation

from applyGeometricTransformation import applyGeometricTransformation
from parameters import *

import scipy.ndimage as ndimage

import sys

def video_write(video_path, images, width, height):

	video = cv2.VideoWriter(video_path,
		cv2.VideoWriter_fourcc(*'MJPG'),20,(width,height))

	for img in images:
		video.write(np.array(img[:,:,::-1], dtype=np.uint8))

	video.release()

def plotBboxAndPoints(img, bbox, xstarts, ystarts):
	F, _, _ = bbox.shape

	for f in range(F):
		b = bbox[f]
		x1,y1,x2, y2 = min(b[:,0]), min(b[:,1]), max(b[:,0]), max(b[:,1])
		cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

		x_new, y_new = xstarts[:, f], ystarts[:, f]
		x_new, y_new = x_new[x_new > -1], y_new[y_new > -1]
		n = len(x_new)
		for i in range(n):
			cv2.circle(img, (int(np.round(x_new[i])), int(np.round(y_new[i]))), 3, (0, 255, 0), 2)
	
	return img

def objectTracking(video_path, bbox):

	cap = cv2.VideoCapture(video_path)
	frames = []
	images = []

	F,_, _ = bbox.shape

	frame_idx = 0
	while True:
		ret, frame = cap.read()
	
		bbox = bbox[:F]
		
		if frame_idx < 1:
			if not ret:
				print("Error! Video has < than 2 frame")
				sys.exit(1)

			frame = frame[:,:,::-1]

			if("Medium" in video_path.split("/")[1]):
				frame = ndimage.rotate(frame, -90)

			frames.append(frame)
			img1 = frame
			xstarts, ystarts = getFeatures(img1, bbox)

			img1c = img1.copy()
			img1c = plotBboxAndPoints(img1c, bbox, xstarts, ystarts)

			images.append(img1c)

			frame_idx += 1
			continue

		if not ret:
			break

		frame = frame[:,:,::-1]

		if("Medium" in video_path.split("/")[1]):
			frame = ndimage.rotate(frame, -90)

		frames.append(frame)

		img1 = frames[frame_idx-1]
		img2 = frames[frame_idx]

		H, W, _ = img1.shape

		newXs, newYs = estimateAllTranslation(xstarts, ystarts, img1, img2)
		newXs_th, newYs_th, newbbox = applyGeometricTransformation(xstarts, ystarts, newXs, newYs, bbox, frame_idx)

		if(newXs_th is None or newYs_th is None):
			print("Early exit")
			break

		img2c = img2.copy()
		img2c = plotBboxAndPoints(img2c, newbbox, newXs_th, newYs_th)

		images.append(img2c)

		bbox = newbbox.copy()

		xstarts = newXs_th.copy()
		ystarts = newYs_th.copy()

		if((frame_idx % UPDATE_FREQ == 0)):
			xstarts, ystarts = getFeatures(img1, bbox)
			if(xstarts is None or ystarts is None):
				print("Early exit. Bye")
				break

		frame_idx += 1

		if "Easy" in video_path.split("/")[1] and frame_idx > 400:
		 	break

	output_path = "videos_track/" + video_path.split('.')[0].split('/')[1] + "_tracking.avi"
	print("Saving video as", output_path)
	video_write(output_path, images, W, H)

if __name__ == "__main__":

	if(len(sys.argv) < 2):
		print("No video name given (should be under video_src). Bad")
		sys.exit(1)

	video_path = sys.argv[1]
	video_path = "videos_src/" + video_path

	if(len(sys.argv) == 3):
		bbox_path = "bbox_preload/" + sys.argv[2] + ".npy"
		bbox = np.load(bbox_path)

		F, _, _ = bbox.shape
		assert F == NUM_OBJECTS_TO_TRACK, "Number of bounding boxes differ from config"

	else:
		F = NUM_OBJECTS_TO_TRACK
		print("Draw thy bounding boxes please", F, " objects")

		cap = cv2.VideoCapture(video_path)
		ret, frame = cap.read()
		if not ret:
			print("No frame! Bye.")
			sys.exit(1)

		frame = frame[:,:,::-1]
		bbox = getBoundingBox(frame, F)

		cap.release()
		cv2.destroyAllWindows()

	objectTracking(video_path, bbox)
