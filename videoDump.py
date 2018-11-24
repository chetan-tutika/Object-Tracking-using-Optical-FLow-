import cv2
import numpy as np

def videoRead(videopath):
	cap = cv2.VideoCapture(videopath)
	frames = []

	i = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
			
		frame = frame[:, :, ::-1]
		frames.append(frame)

		i += 1
				
	cap.release()
	cv2.destroyAllWindows()

	return frames

if __name__ == "__main__":
	import sys
	video_path = "videos_src/" + sys.argv[1] + ".mp4"

	frames = videoRead(video_path)

	frames_path = sys.argv[1] + "_frames"
	
	print("Saving as: ", frames_path)

	np.save(frames_path, frames)