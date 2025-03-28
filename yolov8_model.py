import cv2
import numpy as np
import os
from ultralytics import YOLO


model = YOLO("yolov8n.pt")  

def ann(object_name):
    os.system(f'espeak-ng "{object_name}"')


classfile = '/home/pi/hackathon/coco.names'  
with open(classfile, 'r') as f:
    classname = f.read().strip().split("\n")


cap = cv2.VideoCapture(0)  
frame_skip = 15  
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  

    
    results = model.predict(frame, conf=0.3)  

    detected_objects = set()
    for result in results:
        for box in result.boxes:
            classId = int(box.cls[0])
            object_name = classname[classId].upper()
            
            if object_name not in detected_objects:
                detected_objects.add(object_name)
                ann(object_name) 

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
