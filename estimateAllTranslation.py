import numpy as np
from utils import rgb2gray, GaussianPDF_2D
from skimage.feature import corner_shi_tomasi
import cv2
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from estimateFeatureTranslation import estimateFeatureTranslation

from parameters import *

def getDerivatives(img):
	w = GAUSSIAN_KERNEL_WIDTH
	G = GaussianPDF_2D(0, 0.5, w,w)
	Gx = convolve2d(G, [[0,0,0],[1, 0, -1],[0,0,0]], mode='same')
	Ix = convolve2d(img, Gx, mode='same')

	Gy = convolve2d(G, [[0,1,0], [0,0,0], [0,-1,0]], mode='same')
	Iy = convolve2d(img, Gy, mode='same')

	return Ix, Iy


def estimateAllTranslation(startXs, startYs, img1, img2):

	N, F = startXs.shape

	img1_g = rgb2gray(img1)

	H, W = img1_g.shape

	Ix, Iy = getDerivatives(img1_g)

	newXs, newYs = np.zeros((N, F)), np.zeros((N, F))

	for f in range(F):
		startX, startY = startXs[:, f], startYs[:, f] # N x 2
		valid_idx = np.logical_and(startX > -1, startY > -1)

		newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)

		x_new, y_new = newX, newY
		valid_idx_n = np.logical_and(np.logical_and(x_new >= 0, y_new >= 0), 
			np.logical_and(x_new < W, y_new < H))

		x_new[np.logical_not(valid_idx_n)] = -1 
		y_new[np.logical_not(valid_idx_n)] = -1
		
		n = len(x_new)
		assert n == len(y_new), "X and Y len differing"

		newXs[valid_idx, f], newYs[valid_idx, f] = x_new, y_new
		newXs[np.logical_not(valid_idx), f], newYs[np.logical_not(valid_idx), f] = [-1,-1]
		n = len(startX)
		newXs[n:, f], newYs[n:, f] = -1, -1

	return  newXs, newYs

if __name__ == "__main__":
	pass

