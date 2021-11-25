import cv2
import numpy as np
import os

haar_cascade = cv2.CascadeClassifier('..\haarcascade_frontalface_default.xml')  # ADD A PATH OF HAARCASCADE FILE 
people = []

for p in os.listdir(r''): #   ADD A PATH OF FOLDER THAT YOU WANT TO TRAIN
    people.append(p)
features = np.load('trained_features.npy', allow_pickle=True)
labels = np.load('trained_labels.npy')

face_recogniser = cv2.face.LBPHFaceRecognizer_create()
face_recogniser.read('face_trained.yml')

img = cv2.imread(r'ADD A PATH OF PICTURE FOR TESTING') #   ADD A PATH OF FOLDER THAT YOU WANT TO TEST
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h, x:x+h]
    label,confidence = face_recogniser.predict(face_roi)
    cv2.putText(img, str(people[label]),(20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(img, str(confidence),(20,190), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), 2)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 4)
print(confidence)
cv2.imshow('detectedface', img)
cv2.waitKey()