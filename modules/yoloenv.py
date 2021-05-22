import numpy as np
import cv2

LABELS_FILE='coco.names'
CONFIG_FILE='yolov3.cfg'
WEIGHTS_FILE='yolov3.weights' 
#WEIGHTS_FILE='yolov3-tiny.weights'

CONFIDENCE_THRESHOLD=0.3

def setup():
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
	
	return CONFIDENCE_THRESHOLD,LABELS,COLORS,net,ln

if __name__ == '__main__':
    print("Standalone not implemented yet..")
    setup()
