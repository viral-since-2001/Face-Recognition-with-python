import os
import cv2
import numpy as np


haar_cascade = cv2.CascadeClassifier('..\haarcascade_frontalface_default.xml')  # ADD A PATH OF HAARCASCADE FILE 

people = []
for p in os.listdir(r''): #   ADD A PATH OF FOLDER THAT YOU WANT TO TRAIN
    people.append(p)
print(people)

DIR = r'PATH OF FOLDER'  #   ADD A PATH OF FOLDER THAT YOU WANT TO TRAIN

features = []
labels = []

def create_train():
    for person in people:
        # print(person)
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            # print(img)
            img_path = os.path.join(path, img)

            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rectangle = haar_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces_rectangle:
                faces_reg = gray[y:y+h, x:x+w]
                features.append(faces_reg)
                labels.append(label)
                # print(faces_reg)
                # print( labels)


create_train()

## CONVERT THE FEATURE AND LABELS TO THE NUMPY ARRAY

features = np.array(features, dtype= 'object')
labels = np.array(labels)

## STARTING OUR FACE RECOGNISER

face_recigniser = cv2.face.LBPHFaceRecognizer_create()

face_recigniser.train(features,labels)
face_recigniser.save('face_trained.yml')
np.save('trained_features', features)
np.save('trained_labels', labels)
np.save('trained_labels', labels)
