import os
from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths


face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
BS = 32

recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

current_id=0
label_ids={}
x_train = []
y_label = []

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

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
                print(aug.flow(x_train,y_label))
print(x_train)
print(y_label) 

#print(aug.flow(x_train,y_label))
#recognizer.train(aug.flow(x_train,y_label))
# storing the labels with their IDs in a file 
with open("labels.pickel", 'wb') as f:
    pickle.dump(label_ids,f)
# construct the training image generator for data augmentation
