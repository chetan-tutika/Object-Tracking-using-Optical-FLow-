import numpy as np
from utils import rgb2gray, GaussianPDF_2D
from skimage.feature import corner_shi_tomasi, corner_peaks
import cv2
from scipy.signal import convolve2d
from skimage import transform as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from parameters import *

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox, frame=None):
	N, F = startXs.shape

	newXs_th, newYs_th = np.zeros((N, F)), np.zeros((N, F))

	newbbox = np.zeros((F, 4, 2), dtype=np.int)

	for f in range(F):
		startX, startY = startXs[:, f], startYs[:, f]

		newX, newY = newXs[:, f], newYs[:, f]
		valid_idx = np.logical_and(newX > -1, newY > -1)

		startX, startY = startX[valid_idx], startY[valid_idx]
		newX, newY = newX[valid_idx], newY[valid_idx]

		n = len(startX)

		src = np.hstack((startX.reshape((n,1)), startY.reshape((n,1))))
		dst = np.hstack((newX.reshape((n,1)), newY.reshape((n,1))))
		
		if COMBINE:
			if frame is not None and frame > 170 and frame < 208:
				sform = tf.estimate_transform(TRANSFORM_COMBINE, src, dst)
			else:
				sform = tf.estimate_transform(TRANSFORM, src, dst)
		else:
			sform = tf.estimate_transform(TRANSFORM, src, dst)

		newXY_pred = sform(src)
		error = np.sqrt(np.sum((newXY_pred - dst)**2, axis=1))
		idx = error < ERROR_TH
		num_inliers = np.sum(idx)

		if frame is not None:
			print("Num of inliers: ", num_inliers, "Frame", frame, "min error: ", np.min(error), "max error", np.max(error))
		else:
			print("Num of inliers:", num_inliers)

		if(num_inliers <= 2):
			return None, None, None
		
		newXY_af = dst.copy()
		newXY_af[np.logical_not(idx)] = [-1, -1]

		newX_th, newY_th = newXY_af[:, 0], newXY_af[:, 1]
		newX_th, newY_th = np.round(newX_th).astype(np.int), np.round(newY_th).astype(np.int)

		n_th = len(newX_th)
		newXs_th[:n_th, f], newYs_th[:n_th, f] = newX_th, newY_th
		newXs_th[n_th:, f], newYs_th[n_th:, f] = -1, -1

		b = bbox[f]
		new_b = sform(b)
		newbbox[f] = np.round(new_b).astype(np.int)

	return newXs_th, newYs_th, newbbox

if __name__ == "__main__":
	pass
