import os
from PIL import Image
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

current_id=0
label_ids={}
x_train = []
y_label = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            #print(label,path)

            if label not in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            # y_labels.append(label) # we got to have some number for the labels insted of text
            # x_train.append(path) # verify this image, turn into numpy array, GRAY
            pil_image = Image.open(path).convert('L') # convert('L') will convert itto grayscale.
            size = (550,550)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8") # converting the image to a numpy array
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array)

            for (x,y,w,h) in faces:
                region_of_intrest = image_array[y:y+h , x:x+w]
                x_train.append(region_of_intrest)
                y_label.append(id_)
#print(x_train)
#print(y_label) 

# storing the labels with their IDs in a file 
with open("labels.pickel", 'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")