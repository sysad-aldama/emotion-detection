# Load video/image/webcam for processing
# Written by: JP Aldama
# Date: 3/13/2020 10:06 pm
# Github: https://github.com/sysad-aldama
# Web: https://www.jeanaldama.info [WIP]
# Email me: quaxiscorp@gmail.com
# Copyright Quaxis Corporation (c) 2020
# Special thanks to: Neha Yadav

import os, cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

model = model_from_json(open('fer.json', 'r').read())
model.load_weights('fer.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while capture.isOpened():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

    



