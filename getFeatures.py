import numpy as np
from utils import rgb2gray
from skimage.feature import corner_shi_tomasi, corner_peaks
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from getBoundingBox import getBoundingBox
from parameters import *

def getFeatures(img, bbox):

	"""
		INPUT:

		img: H x W x 3, RGB Image
		bbox: F x 4 x 2, bounding box coordinates of F objects


		OUTPUT:

		x_feats: N x F, x-coordinates of N feature points of F objects
		y_feats: N x F, y-coordinates of N feature points of F objects
	"""

	F, _, _ = bbox.shape

	N_max = NUM_FEATURES
	x_feats, y_feats = np.zeros((N_max, F)), np.zeros((N_max, F))

	for f in range(F):
		b = bbox[f]

		x,y,x1, y1 = min(b[:,0]), min(b[:,1]), max(b[:,0]), max(b[:,1])

		b_img = img[y:y1, x:x1, :].copy()
		img_g = cv2.cvtColor(b_img, cv2.COLOR_RGB2GRAY)
		points = cv2.goodFeaturesToTrack(img_g, N_max, FEATURES_QUALITY, FEATURES_MIN_DISTANCE)

		if points is None:
			return None, None
		elif len(points) <= 2:
			return None, None

		points = points[:,0,:]
		points = np.round(points).astype(np.int)

		n = len(points)
		print("Number of corner points: ", n)

		x_feats[:n, f] = x + points[:, 0]
		y_feats[:n, f] = y + points[:, 1]

		x_feats[n:, f] = -1
		y_feats[n:, f] = -1

	return x_feats, y_feats

if __name__ == "__main__":
	pass



	