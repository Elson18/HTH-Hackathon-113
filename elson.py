import cv2
import numpy as np
import os

whT = 224  
confThreshold = 0.3
nmsThreshold = 0.2


classfile = 'coco.names'  
with open(classfile, 'r') as f:
    classname = f.read().strip().split("\n")


modelConfiguration = 'yolov3-tiny.cfg'  
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def objectFind(outputs, frame_width):
    detected_objects = set()
    for output in outputs:
        for d in output:
            scores = d[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(d[0] * frame_width)

                # Determine position (Left, Center, Right)
                if center_x < frame_width // 3:
                    position = "left"
                elif center_x > 2 * frame_width // 3:
                    position = "right"
                else:
                    position = "center"

                object_name = classname[classId].upper()
                
                if object_name not in detected_objects:  
                    detected_objects.add(object_name)
                    print(f"Detected: {object_name} at {position}")  

cap = cv2.VideoCapture(0) 
frame_skip = 10  
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:  
        continue  

    frame_height, frame_width = frame.shape[:2]

    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    objectFind(outputs, frame_width)  

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
