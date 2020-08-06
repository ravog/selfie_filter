#pip3 install opencv-python
#pip3 install opencv-contrib-python

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Constantes
X = 0   # indice de la coordenada X del keypoint
Y = 1   # indice de la coordenada X del keypoint
L_EYE_CENTER = 0
R_EYE_CENTER = 1
L_EYE_INNER = 2
L_EYE_OUTER = 3
R_EYE_INNER = 4
R_EYE_OUTER = 5
L_EYE_BROW_INNER = 6
L_EYE_BROW_OUTER = 7
R_EYE_BROW_INNER = 8
R_EYE_BROW_OUTER = 9
NOSE = 10
MOUTH_LEFT = 11
MOUTH_RIGHT = 12
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Variables
cv2.namedWindow("face detection activated")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
filter = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)
frame = cv2.imread("images/obamas4.jpg", cv2.IMREAD_UNCHANGED)
model = load_model('my_model.h5')

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
        # for landmark_centre in points:
        #     cv2.circle(resized_face_color, landmark_centre, 1, (0,255,0), 1)
        right_eye_brow_outer_x = int(points[R_EYE_BROW_OUTER][X])
        right_eye_brow_outer_y = int(points[R_EYE_BROW_OUTER][Y])

        filter_height = int((points[NOSE][Y] - right_eye_brow_outer_y) / 1.5)
        filter_width = int((points[L_EYE_BROW_OUTER][X] - right_eye_brow_outer_x) * 1.1)

        filter_resized = cv2.resize(filter, (filter_width, filter_height))
        alpha_region = filter_resized[:, :, 3] != 0
        resized_face_color[
            right_eye_brow_outer_y:right_eye_brow_outer_y + filter_height,
            right_eye_brow_outer_x:right_eye_brow_outer_x + filter_width, :][alpha_region] = \
            filter_resized[:, :, :3][alpha_region]

        resized_face_color = cv2.resize(resized_face_color, original_shape, interpolation = cv2.INTER_CUBIC)

        frame[y:y+h, x:x+w] = resized_face_color
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    # Display the resulting frame
    cv2.imshow('webcam',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()