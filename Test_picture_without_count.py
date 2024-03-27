import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import time

# Initialize YOLO detector
model = YOLO('best-bin2.pt')

# Read the image
img = cv2.imread('testy1.png')

# Resize the image to the desired input size (e.g., 416x416)
resized_frame = cv2.resize(img, (416, 416))

# YOLO Prediction
results = model(resized_frame)
result = results[0]
bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
classes = np.array(result.boxes.cls.cpu(), dtype='int')
confidences = np.array(result.boxes.conf.cpu())
class_name = result.names

for cls, bbox, conf in zip(classes, bboxes, confidences):
    x1, y1, x2, y2 = bbox
    labels = result.names[cls]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, str(labels) + str(np.round(conf, 2)), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 0, 255), 2)

# Show the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
