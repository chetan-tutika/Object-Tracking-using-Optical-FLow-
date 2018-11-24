import numpy as np
import cv2
from maskImage import maskImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from parameters import NUM_OBJECTS_TO_TRACK

def drawRectangle(ax, b):
	x, y = min(b[:,0]), min(b[:,1])
	x1, y1 = max(b[:,0]), max(b[:,1])

	w , h= x1 - x, y1 - y
	rect = patches.Rectangle((x,y), w, h, linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)

def getBoundingBox(img, F):
	bbox = np.zeros((F, 4, 2)).astype(np.int)

	for f in range(F):
		xy = maskImage(img)
		x, y, w, h = cv2.boundingRect(np.array(xy, dtype=np.int))
		print(x,y,w,h)

		bbox[f] = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])


	plt.figure(1)
	ax = plt.gca()
	ax.imshow(img)
	for f in range(F):
		drawRectangle(ax, bbox[f])
	
	plt.show()

	return bbox


if __name__ == "__main__":
	import sys
	import scipy.ndimage as ndimage

	video_path = "videos_src/" + sys.argv[1]
	print("Loading: ", video_path)

	F = NUM_OBJECTS_TO_TRACK
	if(len(sys.argv) > 2):
		F = int(sys.argv[2])
	
	cap = cv2.VideoCapture(video_path)
	ret, frame = cap.read()
	if not ret:
		print("No frame! Bye.")
		sys.exit(1)

	frame = frame[:,:,::-1]
	frame = ndimage.rotate(frame, -90)
	bbox = getBoundingBox(frame, F)

	cap.release()
	cv2.destroyAllWindows()
	
	print("Number of Objects to be tracked: ", F)

	bbox_path = "bbox_preload/" + sys.argv[1].split(".")[0] + "_bbox"

	np.save(bbox_path, bbox)
	print("Saving bounding box as: ", bbox_path)
		
