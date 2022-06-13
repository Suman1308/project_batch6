import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickel", "rb") as f:
    labels = pickle.load(f)
    # inverting the labels to "id": labels
    labels={v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.9,minNeighbors=5)
    eye_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')

    for (x,y,w,h) in faces:
        #print(x,y,w,h) #(x_coordinat_start,y_coordinat_start,width,height)
        region_of_intrest= gray[y:y+h,x:x+h]

        id_, conf = recognizer.predict(region_of_intrest) # conf -> confidence
        print("confidence = ",conf," and label = ",labels[id_])

        font = cv2.FONT_HERSHEY_SIMPLEX
        color=(255,255,255)
        name=labels[id_]
        thickness=2
        cv2.putText(frame,name,(x,y+h+30),font,1,color,thickness,cv2.LINE_AA)

        img_item="my-image.png"
        cv2.imwrite(img_item, region_of_intrest)


        #Drawing a rectangale around the face
        color = (255,0,0) #BGR 0-255
        thickness = 2
        end_coordinate_x = x + w
        end_coordinate_y = y + h

        cv2.rectangle(frame,(x,y),(end_coordinate_x,end_coordinate_y), color, thickness)
        eyes = eye_cascade.detectMultiScale(region_of_intrest)
    
    # Displaying the frames
    cv2.imshow('frame',frame )
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break