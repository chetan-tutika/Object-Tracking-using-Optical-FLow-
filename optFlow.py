import numpy as np
import argparse
import cv2
import dlib
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from skimage import transform as tf
from getFacialLandmarks import dlibFaceDetect
from getFacialLandmarks import dlibFaceLandmarks
from warpImage import warpImage
from facemorph import morph_tri
from getFacialLandmarks import faceAlignment
import pdb

def video_write(video_path, images, width, height):

	video = cv2.VideoWriter(video_path, 
		cv2.VideoWriter_fourcc(*'MJPG'),20,(width,height))

	for img in images:
		video.write(np.array(img, dtype=np.uint8))

	video.release()

def resizePts(fts, face_rect, size_x, size_y):
	xr, yr, wr, hr = getXYWH(face_rect)
	ratio_width = float(size_x/wr)
	ratio_height = float(size_y/hr)

	#print("source: current h,w",h,w)
	#Im = cv2.resize(Im, (0,0), fx= 1.0/ratio_width, fy= 1.0/ratio_height)
	print('fts before', fts)
	fts = fts*np.array([1.0/ratio_width, 1.0/ratio_height])
	print('fts after', fts)
	#print("source: resized h,w", face.shape[:2])
	#w, h = face.shape[1],face.shape[0]


	return fts

def getXYWH(rect):
	x1 = rect.left()
	y1 = rect.top()
	w1 = rect.right() - x1
	h1 = rect.bottom() - y1

	return x1,y1,w1,h1




def optFlow(featPoints, imPrev, imCur):

	
	featPoints = featPoints.astype(np.float32)
	featPoints = featPoints.reshape(featPoints.shape[0], 1, 2)
	#print('---------------------------------------------------------------------------------')
	#print('feat as array', featPoints[:5])
	# boxList = []
	#for face in bbox:
	#print('bbox type', bbox)
	lk_params = dict( winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	# if type(bbox) == 'dlib.rectangle':
	# 	print('iteration')
	# 	x = bbox.left()
	# 	y = bbox.top()
	# 	w = bbox.right() - x
	# 	h = bbox.bottom() - y
	# 	#boxList.append([x,y,w,h])
	# 	boxList = np.array([x,y,w,h]).reshape(2,2)
	# 	#print('boxList', boxList)
	# else:
	# 	print('iteration which shouldnt')
	# boxList = bbox.reshape(2,2)



	#featPoints = featPoints + [left, top]
	imPrev_gray = cv2.cvtColor(imPrev, cv2.COLOR_BGR2GRAY)
	imCur_gray = cv2.cvtColor(imCur, cv2.COLOR_BGR2GRAY)


	p1, st, err = cv2.calcOpticalFlowPyrLK(imPrev_gray, imCur_gray, featPoints, None, **lk_params)
	#p1B, st, err = cv2.calcOpticalFlowPyrLK(imPrev_gray, imCur_gray, boxList, None, **lk_params)
	# print('p1', p1[:5])
	# print('st', len(st[st == 1]))
	good_new = p1[st == 1]
	# good_old = featPoints[st == 1]

	# print('new features', good_new[:5])
	# print('good_old', good_old[:5])

	# sform = tf.estimate_transform('affine', good_old, good_new)
	# bboxNew = sform(boxList)

	# #print('bboxNew', bboxNew)
	# bboxNew  = np.round(bboxNew).astype(int)
	# bboxNew = bboxNew.flatten()
	# #print('bboxNew reshaped', bboxNew)
	return good_new

def getOriginalIndex(coor, leftInd, topInd):

	coor = coor + np.array([leftInd, topInd])

	return coor




if __name__ == "__main__":
	# stream1 = cv2.VideoCapture('videos/easy/MrRobot.mp4')
	# stream1 = cv2.VideoCapture('videos/Trump.mp4')
	stream1 = cv2.VideoCapture('videos/easy/FrankUnderwood.mp4')

	# stream2 = cv2.VideoCapture('videos/easy/FrankUnderwood.mp4')
	stream2 = cv2.VideoCapture('videos/easy/JonSnow.mp4')


	images = []
	images1 = []

	#while True:
	for i in range(100):
		(grabbed1, frame1) = stream1.read()
		(grabbed2, frame2) = stream2.read()
		if (not (grabbed1 and grabbed2)):
			break

		#frame2 = cv2.resize(frame2, (0,0), fx=0.7, fy=0.7)
		print('iteration', i)
		frameC = frame2.copy()
		frameSC = frame1.copy()


		if i == 0:
 
			
			#cv2.imshow('frame',frame1)
			# face = cv2FaceDetect(frame) # 50 ms per face, poor performace sometims
			# face = dlibCNNFaceDetect(frame) # 3 seconds per frame toooo slow
			srcIm, faces_hog1 = dlibFaceDetect(frame1) # best as of now 150ms per face
			dstIm, faces_hog2 = dlibFaceDetect(frame2)
			print('faces =', len(faces_hog2))

			f = 0.25
			# faces_hog1 = dlib.rectangle(int(faces_hog1[0].left() - faces_hog1[0].left()*f), int(faces_hog1[0].top() - faces_hog1[0].top()*f ), int(faces_hog1[0].right() + f*faces_hog1[0].right()), int(faces_hog1[0].bottom() + f*faces_hog1[0].bottom()))
			# faces_hog2 = dlib.rectangle(int(faces_hog2[0].left() - faces_hog2[0].left()*f) , int(faces_hog2[0].top() - faces_hog2[0].top()*f ), int(faces_hog2[0].right() + f*faces_hog2[0].right()), int(faces_hog2[0].bottom() + f*faces_hog2[0].bottom())
			faces_hog1 = dlib.rectangle(int(faces_hog1[0].left() - 20), int(faces_hog1[0].top() - 20 ), int(faces_hog1[0].right() + 20), int(faces_hog1[0].bottom() + 20))
			faces_hog2 = dlib.rectangle(int(faces_hog2[0].left() - 20) , int(faces_hog2[0].top() - 20 ), int(faces_hog2[0].right() + 20), int(faces_hog2[0].bottom() + 20))

			#faces_hog2R = getXYWH(faces_hog2)
			#faces_hog1R = getXYWH(faces_hog1)
			#cv2.rectangle(frameSC,(faces_hog1R[0]-10, faces_hog1R[1] - 10),(faces_hog1R[0] + faces_hog1R[3] + 10, faces_hog1R[1] + faces_hog1R[3] + 10),(255,0,0),2)

			#cv2.imshow('0', frameSC)
			dstFace, dstPts = faceAlignment(dstIm.copy(), faces_hog2)
			sizey,sizex = dstFace.shape[0],dstFace.shape[1]

			#srcIm = resizeIm(frame1.copy(), faces_hog1, sizex, sizey)
			#print('srcIm', srcIm)
			
			# Just take one face per frame for now #change function name to this --dlibFaceLandmarks  ,faceAlignment
			
			
			srcFace, srcPts = faceAlignment(srcIm.copy(), faces_hog1, sizex, sizey)
			
			srcPts = resizePts(srcPts, faces_hog1, sizex, sizey)
			#print('srcPts', srcPts1)
			left_index_dst = faces_hog2.left()
			top_index_dst = faces_hog2.top()

			left_index_src = faces_hog1.left()
			top_index_src = faces_hog1.top()


			# dstPts_Plus = dstPts + np.array([left_index, top_index])
			# srcPts_Plus = srcPts + np.array([left_index, top_index])

			dstPts_Plus = getOriginalIndex(dstPts, left_index_dst, top_index_dst)
			srcPts_Plus = getOriginalIndex(srcPts, left_index_src, top_index_src)
			#print('srcPts_Plus', srcPts_Plus)

			xbs, ybs, wbs, hbs = cv2.boundingRect(srcPts_Plus.astype(int))
			cv2.rectangle(frameSC,(xbs-10, ybs - 10),(xbs + wbs + 10, ybs + hbs + 10),(255,0,0),2)
			#cv2.imshow('1', frameSC)




			prevFrameSrc = frame1
			prevFrameDst = frame2
			dstImC = dstIm.copy()
			for (x, y) in dstPts:
			 	cv2.circle(dstImC, (x + faces_hog2.left(), y + faces_hog2.top()), 1, (0, 0, 255), -1)

			images.append(dstImC)
		else:
			curFrameSrc = frame1
			curFrameDst = frame2
			
			newDstPoints = optFlow(dstPts_Plus, prevFrameDst, curFrameDst)
			newSrcPoints = optFlow(srcPts_Plus, prevFrameSrc, curFrameSrc)

			prevFrameSrc = curFrameSrc
			prevFrameDst = curFrameDst
			#faces_hog2 = new_bbox
			dstPts_Plus = newDstPoints
			srcPts_Plus = newSrcPoints
			xb, yb, wb, hb = cv2.boundingRect(dstPts_Plus)
			xbs, ybs, wbs, hbs = cv2.boundingRect(srcPts_Plus)

			for (x, y) in dstPts_Plus:
			 	cv2.circle(frameC, (x, y), 1, (0, 0, 255), -1)
			cv2.rectangle(frameC,(xb-10, yb - 10),(xb + wb + 10, yb + hb + 10),(255,0,0),2)

			for (x, y) in srcPts_Plus:
			 	cv2.circle(frameSC, (x, y), 1, (0, 0, 255), -1)
			cv2.rectangle(frameSC,(xbs-10, ybs - 10),(xbs + wbs + 10, ybs + hbs + 10),(255,0,0),2)

			images.append(frameC)
			images1.append(frameSC)


		# cv2.imshow('dstFace', dstIm)
		# cv2.waitKey(0)
	H, W, _ = images[0].shape
	H1, W1, _ = images1[0].shape

	video_write('face_swap_opti.avi', images, W, H)
	video_write('face_swap_opti_Src.avi', images1, W1, H1)



cv2.waitKey(0)	#break

