import numpy as np
from utils import rgb2gray, GaussianPDF_2D
from skimage.feature import corner_shi_tomasi
import cv2
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import map_coordinates

from parameters import *

def getXYNeighbours(x, y, W, H):
	n = len(x)
	Wxx, Wyy = np.meshgrid(np.arange(-5, 6, 1), np.arange(-5, 6, 1), indexing='xy')
	Wxy = np.array(list(zip(Wxx.flatten(), Wyy.flatten()))) # 100 x 2
	Wxy = np.repeat(Wxy[:,:,np.newaxis], n, axis=2) # 100 x 2 x N
	Wxy = Wxy.transpose((2,1,0)) # N x 2 x 100

	xy = np.hstack((x.reshape((n,1)), y.reshape((n,1))))
	xy = xy[:,:,np.newaxis] # N x 2 x 1
	xy_neigh = xy + Wxy # N x 2 x 100

	xy_neigh[:, 0, :] = np.clip(xy_neigh[:, 0, :], 0, W-1)
	xy_neigh[:, 1, :] = np.clip(xy_neigh[:, 1, :], 0, H-1)

	xy_neigh = xy_neigh.transpose((0,2,1)) # N x 100 x 2

	return xy_neigh

def getApinv(Ix_w, Iy_w):
	A = np.dstack((Ix_w, Iy_w)) # N x 100 x 2
	Apinv = np.linalg.pinv(A) # N x 2 x 100

	return Apinv

def getDisplacement(Apinv, It_w):
	It_w = It_w[:,:,np.newaxis] # N x 100 x 1
	uv = np.matmul(Apinv, It_w).squeeze()

	return uv

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
	img1_g = rgb2gray(img1)
	img2_g = rgb2gray(img2)

	T = NUM_ITERS_OPTICAL_FLOW # num of optical flow iterations

	H, W, _ = img1.shape

	x, y = startX.copy(), startY.copy()
	valid_idx = np.logical_and(x > -1, y > -1)
	x, y = x[valid_idx], y[valid_idx]
	xy_neigh = getXYNeighbours(x, y, W, H)

	N , win, _ = xy_neigh.shape

	Ix_w = map_coordinates(Ix, [xy_neigh[:,:,1], xy_neigh[:,:,0]], order=1, mode='constant').reshape((N,win))
	Iy_w = map_coordinates(Iy, [xy_neigh[:,:,1],xy_neigh[:,:,0]], order=1, mode='constant').reshape((N,win))


	Apinv = getApinv(Ix_w, Iy_w)

	img1_w = map_coordinates(img1_g, [xy_neigh[:,:,1],xy_neigh[:,:,0]], order=1, 
		mode='constant').reshape((N,win))
	
	sx, sy = startX.copy(), startY.copy()
	valid_idx = np.logical_and(sx > -1, sy > -1)
	sx, sy = sx[valid_idx], sy[valid_idx]

	for t in range(T):
		xy_neigh2 = getXYNeighbours(sx, sy, W, H)

		img2_w = map_coordinates(img2_g, [xy_neigh2[:,:,1], xy_neigh2[:,:,0]], order=1, mode='constant')	.reshape((N, win))
		
		It_w = img1_w - img2_w # N x 100

		uv = getDisplacement(Apinv, It_w) # N x 2

		assert uv.shape[0] == sx.shape[0]

		x_new = sx + uv[:, 0]
		y_new = sy + uv[:, 1]

		sx, sy = x_new.copy(), y_new.copy()


		### Stopping Criterion
		# norm = np.linalg.norm(uv)


	return x_new, y_new

if __name__ == "__main__":
	pass

