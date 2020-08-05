#pip3 install opencv-python
#pip3 install opencv-contrib-python

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

cv2.namedWindow("face detection activated")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
filter = cv2.imread("images/sunglasses_4.png")
# Load facial keypoints detector model
model = load_model('my_model.h5')

def resize(img, width):
    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

while(True):
    # Wait for Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #cv2.rectangle(frame,(400,150),(560,300),(0,255,0), 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25,6)
    for (x, y, w, h) in faces:
        current_filter = cv2.resize(filter, (w, h), interpolation = cv2.INTER_AREA)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
        gray_face_slice = gray[y:y+h, x:x+w]
        color_face_slice = frame[y:y+h, x:x+w]
        
        # Normalize the gray image for input into the model, so that the values are between [0, 1]
        gray_scale = gray_face_slice / 255
        
        # Resize the image into 96x96
        original_shape = gray_scale.shape # Keep track of the original size before resizing.
        resized_face = cv2.resize(gray_scale, (96, 96), interpolation = cv2.INTER_AREA)
        resized_face_copy = resized_face.copy()
        
        resized_face = resized_face.reshape(1, 96, 96, 1) # Resize it further to match expected input into the model
        
        # Predict
        landmarks = model.predict(resized_face)
        landmarks = landmarks * 48 + 48 # undo the standardization
        resized_face_color = cv2.resize(color_face_slice, (96, 96), interpolation = cv2.INTER_AREA)
    
        # Resize the 96x96 image to its original size
        # Paste it into the original image
    
        points = []
        for i, co in enumerate(landmarks[0][0::2]):
            points.append((co, landmarks[0][1::2][i]))
        for landmark_centre in points:
            cv2.circle(resized_face_color, landmark_centre, 1, (0,255,0), 1)
            
        resized_face_color = cv2.resize(resized_face_color, original_shape, interpolation = cv2.INTER_CUBIC)
        
        frame[y:y+h, x:x+w] = resized_face_color
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    # Display the resulting frame
    cv2.imshow('webcam',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()