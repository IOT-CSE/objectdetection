import numpy as np
import time
import cv2
import imutils
from modules import yoloenv
from modules import yoloimage
from imutils.video import FPS
from imutils.video import VideoStream


"""
Conclusions after testing shopping center video
-Confidence threshold not that good 
-Processing each frame takes too much time. Instead of each frame, we can implement dynamic
function that decides frame increment. So that, we can skip up frames where there are no person detected(eg. not working hours or no customer appeared yet)
-Apparently even the guy on advertisement board in shopping center detected as person. 
-Check resources about how to determine distance of object from camera.

Distance = real object height * focal length / object height on sensor
But we dont know the persons height ? ? 

https://www.scantips.com/lights/subjectdistance.html 
"""

INPUT_FILE='retailstore.mp4'
OUTPUT_FILE='output.mp4'
OUTPUT_CODEC='mp4v'

DEBUG_MODE = False

H_OUT=800
W_OUT=600

FOCAL_LENGTH = 3.5
OBJECT_HEIGHT = 170

def detect():
	fps = FPS().start()
	vs = cv2.VideoCapture(INPUT_FILE)

	# Get the frames per second
	fps2 = vs.get(cv2.CAP_PROP_FPS) 

	# Get the total numer of frames in the video.
	frame_count = vs.get(cv2.CAP_PROP_FRAME_COUNT)
	print("Total frame count", frame_count)

	frame_number = 0
	frame_speedup = 1
	
	#Output video properties
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
	(H_OUT, W_OUT), True)
	while frame_number <= frame_count:
		frame_number += frame_speedup 
		vs.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
		print ("Frame number", frame_number)
		grabbed, image = vs.read()
		if not grabbed:
			print("No frame grabbed, exiting..")
			break
		img_result, n_idx = yoloimage.detect(image,CONFIDENCE_THRESHOLD,LABELS,COLORS,net,ln,H_OUT,W_OUT,DEBUG_MODE)
		writer.write(cv2.resize(img_result,(H_OUT, W_OUT)))
		if n_idx < 4:
  			frame_speedup = 5 - n_idx
		else:
  			frame_speedup = 1
		fps.update()
		key = cv2.waitKey(1) & 0xFF
		if DEBUG_MODE and key == ord("q"):
			break
	fps.stop()
	cv2.destroyAllWindows()
	writer.release()
	vs.release()
	
if __name__ == '__main__':
    CONFIDENCE_THRESHOLD,LABELS,COLORS,net,ln = yoloenv.setup()
    detect()
