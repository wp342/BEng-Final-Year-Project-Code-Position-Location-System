# The initial parts of this code was taken from armaanpriyadarshans repository
# Reference: https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-PiCamera-od.py
import tflite_runtime.interpreter as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import triangulation as tc
import matplotlib.pyplot as plt
import csv

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='model/detect.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='model/labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
                    
args = parser.parse_args()

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
import time
print('Loading model...', end='')
start_time = time.time()

# LOAD TFLITE MODEL
interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)
# LOAD LABELS
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

input_mean = 127.5
input_std = 127.5

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()

# initialise the reference coordinates in the room
realCoords = {'1': [51.5,36], '2': [53,154.5], '3': [107.5,96.5], '4' : [161.5,43], '5' : [164.5,160.5], '6' : [235.5,100]}

# drawing the map of the test environment
testMap = np.full((520,780),255, np.uint8)
testMap = cv2.rectangle(testMap, (258,0),(520,75),(0),-1) # blocked part of the room 
testMap = cv2.rectangle(testMap, (0,0),(258,132),(100),1) # wardrobe
testMap = cv2.rectangle(testMap, (360,224),(780,520),(100),1) # bed
testMap = cv2.rectangle(testMap, (520,0),(780,115),(100),1) # desk
for i in range(1,7):
    x = 2*realCoords[str(i)][0]
    y = 520 - 2*realCoords[str(i)][1]
    testMap = cv2.circle(testMap,(int(x),int(y)),(5),(0),1)
sx=0
sy=0
sz=0

while True:
    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame,5)
    frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    frame = np.repeat(frame[..., np.newaxis], 3, -1)
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values as using a floating model (i.e. if model is non-quantized)
    input_data = (np.float32(input_data) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    
    # begin the dection data processing 
    index = tc.find_class_box_index(classes, labels, scores)
    if len(index)<3:
        print("not enough reference points in view")
        continue
    centreCoords = tc.find_bbox_centre(index, boxes, imW, imH)
    angles = tc.calculate_angles(centreCoords, [imW/2, imH/2])
    refCoords = tc.sort_data(index, angles, realCoords, centreCoords)
    
    # feed the processed data into the triangualtion algorithms
    if len(refCoords)>3 :
        x,y = tc.ToTal_Algorithm_Special_Cases(refCoords)   
    else:
        x,y = tc.ToTal_Algorithm(refCoords)
    # calculate the depth or z distance of the device from the ceiling     
    z = 231.5-tc.z_depth_calculator(refCoords)
    print(" ")
    print( "x cordinate is " + str(x))
    print( "y coordinate is " + str(y))
    print( "z coordinate is " + str(z))
    X=2*x
    Y=520-2*y
    testMap = cv2.circle(testMap,(sx,sy),(3),(255),-1)
    testMap = cv2.circle(testMap,(X,Y),(3),(0),-1)
    cv2.putText(testMap,'Z : {0:.2f}'.format(sz),(680,500),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255),1,cv2.LINE_AA)
    cv2.putText(testMap,'Z : {0:.2f}'.format(z),(680,500),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0),1,cv2.LINE_AA)
    sx=X
    sy=Y
    sz=z
    cv2.imshow('Test Environment Map', testMap)
    
     # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
print("Done")