import numpy as np
import time
import cv2
import imutils
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
LABELS_FILE='coco.names'
CONFIG_FILE='yolov3.cfg'

#Pretrained weights of yolo.

WEIGHTS_FILE='yolov3.weights' 
#WEIGHTS_FILE='yolov3-tiny.weights'
CONFIDENCE_THRESHOLD=0.3

#Just for debugging 
DEBUG_MODE = False

H=None
W=None

#Distance to object = (Real object height*Focal Length)/Object height sensor mm
#Real object heigh = Distance to object* Object height on sensor / Focal length
## Calculate height of each person 

FOCAL_LENGTH = 3.5
OBJECT_HEIGHT = 170


def yoloenv_setup():
	LABELS = open(LABELS_FILE).read().strip().split("\n")

	#Deterministic random variable generation
	np.random.seed(4)

	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	cnt =0;
	
	return LABELS,COLORS,net,ln

def detect_from_image(image):

	LABELS,COLORS,net,ln = yoloenv_setup()
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	H, W = image.shape[:2]
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
		CONFIDENCE_THRESHOLD)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]

			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)
	# show the output image
	if DEBUG_MODE:
		cv2.imshow("output", cv2.resize(image,(800, 600)))
	return image,len(idxs)

def videostream():
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
	(800, 600), True)
	while frame_number <= frame_count:
		frame_number += frame_speedup 
		vs.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
		print ("Frame number", frame_number)
		grabbed, image = vs.read()
		if not grabbed:
			print("No frame grabbed, exiting..")
			break
		img_result, n_idx = detect_from_image(image)
		writer.write(cv2.resize(img_result,(800, 600)))
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
    videostream()
