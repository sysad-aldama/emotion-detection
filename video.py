# Load video/image/webcam for processing
# Written by: JP Aldama
# Date: 3/13/2020 10:06 pm
# Github: https://github.com/sysad-aldama
# Web: https://www.jeanaldama.info [WIP]
# Email me: quaxiscorp@gmail.com
# Copyright Quaxis Corporation (c) 2020
# Special thanks to: Neha Yadav

import os 
import cv2
import tensorflow
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        continue
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.5,8)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 8)
        roi_grayscale = grayscale[y:y+h, x:x+w]
        roi_grayscale = cv2.resize(roi_grayscale,(48,48))
        image_px = image.img_to_array(roi_grayscale)
        image_px = np.expand_dims(image_px,axis=0)
        image_px /= 255
        predictions = model.predict(image_px)
        index_max = np.argmax(predictions[0])
        emotions = ('angry','disgust','fear','happy','sad','surprise','neutral')
        predicted_emotion = emotions[index_max]
        cv2.putText(img, predicted_emotion, (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.putText(img, ('Emotion Detected: {}'.format(predicted_emotion)), (10, 50),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
    
    resized_image = cv2.resize(img, (1000,700))
    
    cv2.imshow('Emotion analysis ', resized_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    else: 
        continue 

cap.release()
cv2.destroyAllWindows()

