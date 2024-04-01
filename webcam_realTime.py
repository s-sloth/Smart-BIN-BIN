##Input delay time for calculation 
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import time

# Initialize YOLO detector
model = YOLO('best.pt')

# create a VideoCapture object to read the video file
cap = cv2.VideoCapture(1) #'video\IMG_1926.MOV'

# get the dimensions of the screen
screen_width = 1280#int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height =600 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create a window named 'video' and set its dimensions to fit the screen
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', screen_width, screen_height)

# define the codec and create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 30, (screen_width, screen_height))

# Initialize variables
frame_count = 0
interval =0.01  # in seconds
start_time = time.time()

# loop through each frame in the video
while cap.isOpened():
    # read the current frame
    ret, frame = cap.read()

    # check if reading the frame was successful
    if not ret:
        break
    frame_count += 1

    # Check if 0.5 seconds have passed since the last image was saved
    elapsed_time = time.time() - start_time
    if elapsed_time >= interval:
        
        #YOLO Prediction  
        # results = model.track(frame,tracker="bytetrack.yaml",show = True)
        results = model(frame)
        result = results[0]
        # print(result)
        bboxes = np.array(result.boxes.xyxy.cpu(),dtype='int')
        classes = np.array(result.boxes.cls.cpu(),dtype='int')
        confidences = np.array(result.boxes.conf.cpu())
        class_name = result.names
        class_name_list = [class_name[i] for i in classes]
        
        # Count the number of predictions for each class in the current frame
        conf_threshold = 0.2 
        df = pd.DataFrame({'class': class_name_list, 'confidence': confidences})
        df = df[df['confidence'] > conf_threshold]
        count_df = df.groupby(['class']).size().reset_index(name='counts')
        count_str = ', '.join([f"{row['class']}: {row['counts']}" for i, row in count_df.iterrows()])
        
        # Add the count to the frame
        cv2.putText(frame, count_str, (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        for cls, bbox , conf in zip(classes, bboxes, confidences ):
            # conf_threshold = 0.8
            if conf > conf_threshold:
                x1, y1, x2, y2 = bbox
                labels = result.names[cls]
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0, 255, 0), 2)
                cv2.putText(frame, str(labels)+ str(np.round(conf,2)), (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2 )

        # out.write(frame)
        # show the current frame in the 'video' window
        cv2.imshow('video', frame)

        # wait for a key press and check if the key pressed was 'q' (to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        start_time = time.time()
    
    

# release the VideoCapture object and destroy the window
cap.release()
cv2.destroyAllWindows()


