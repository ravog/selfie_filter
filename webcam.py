# pip3 install opencv-python
# pip3 install opencv-contrib-python

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Constantes
X = 0  # indice de la coordenada X del keypoint
Y = 1  # indice de la coordenada X del keypoint
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
sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)
model = load_model('my_model.h5')
action_key = 255


# Fucniones
def draw_marks(x, y, w, h, keypoints, image):
    for keypoint_centre in keypoints:
        cv2.circle(image, keypoint_centre, 1, (0,255,0), 1)

    return image


def put_sunglasses(x, y, w, h, keypoints, image):
    right_eye_brow_outer_x = int(keypoints[R_EYE_BROW_OUTER][X])
    right_eye_brow_outer_y = int(keypoints[R_EYE_BROW_OUTER][Y])

    sunglasses_height = int((keypoints[NOSE][Y] - right_eye_brow_outer_y) / 1.1)
    sunglasses_width = int((keypoints[L_EYE_BROW_OUTER][X] - right_eye_brow_outer_x))

    sunglasses_resized = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))
    alpha_region = sunglasses_resized[:, :, 3] != 0
    image[
    right_eye_brow_outer_y:right_eye_brow_outer_y + sunglasses_height,
    right_eye_brow_outer_x:right_eye_brow_outer_x + sunglasses_width, :][alpha_region] = \
        sunglasses_resized[:, :, :3][alpha_region]

    return image

def hide_face(x, y, w, h,keypoints, image):
    kernel = np.ones((100, 100), np.float32) / 10000
    image = cv2.filter2D(image, -1, kernel)

    return image


def reset(x, y, w, h,keypoints, image):
    action_key = 255

    return image


ACTIONS = {
    49: draw_marks,  # Tecla 1
    50: put_sunglasses,  # Tecla 2
    51: hide_face,  # Tecla 3
    114: reset,  # Tecla R
}


while (True):
    k = cv2.waitKey(10) & 0xff
    ret, frame = cap.read()  # Se captura cuadro por cuadro el video
    frame = cv2.flip(frame, 1)
    if k != 255 or action_key != 255:
        if k != 255:
            action_key = k
        if action_key == ord('q'):
            break
        else:
            gray = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)  # comvierte a grayscale la imagen para la deteccion de caras
            faces = face_cascade.detectMultiScale(gray, 1.25, 6)  # deteccion de caras
            for (x, y, w, h) in faces:
                gray_face_slice = gray[y:y + h, x:x + w]
                color_face_slice = frame[y:y + h, x:x + w]
                gray_scale = gray_face_slice / 255  # Normalize the gray image for input into the model,
                # so that the values are between [0, 1]
                original_shape = gray_scale.shape  # Keep track of the original size before resizing.
                resized_face = cv2.resize(gray_scale, (96, 96), interpolation=cv2.INTER_AREA)  # Resize into 96x96
                resized_face = resized_face.reshape(1, 96, 96,
                                                    1)  # Resize it further to match expected input into the model
                landmarks = model.predict(resized_face)  # Se realiza la prediccion de keypoints
                landmarks = landmarks * 48 + 48  # Se desnormaliza la salida de la prediccion
                resized_face_color = cv2.resize(color_face_slice, (96, 96), interpolation=cv2.INTER_AREA)
                keypoints = []
                for i, co in enumerate(landmarks[0][0::2]):
                    keypoints.append((co, landmarks[0][1::2][i]))
                resized_face_color = ACTIONS[action_key](x, y, w, h, keypoints, resized_face_color)
                resized_face_color = cv2.resize(resized_face_color, original_shape, interpolation=cv2.INTER_CUBIC)
                frame[y:y + h, x:x + w] = resized_face_color
    cv2.imshow('webcam', frame)  # Display the resulting frame