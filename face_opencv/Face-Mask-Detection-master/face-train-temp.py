# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import pickle
import mysql.connector
from datetime import date
import smtplib
import face_recognition

connection = mysql.connector.connect(host='localhost',
                                         database='face mask',
                                         user='root',
                                         password='')

path = r"C:\xampp\htdocs\face_opencv\Face-Mask-Detection-master\images"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
	curImg = cv2.imread(f'{path}/{cl}')
	images.append(curImg)
	classNames.append(os.path.splitext(cl)[0])
	print(classNames)
 
def findEncodings(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')

visited=list()
def converted_image():
	with open(r'C:\xampp\htdocs\face_opencv\Face-Mask-Detection-master\cap_image.png' , 'rb') as file:
		contents = file.read()
	return contents

image_serial = 0
image_count=0


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"C:\xampp\htdocs\face_opencv\Face-Mask-Detection-master\face_detector\deploy.prototxt"
weightsPath = r"C:\xampp\htdocs\face_opencv\Face-Mask-Detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame1 = vs.read()
	frame = imutils.resize(frame1, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		
		if (label=='No Mask' and image_count==10):
			img_name= "cap_image.png"
			img =frame1
			imgS = cv2.resize(img,(0,0),None,0.25,0.25)
			imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
			facesCurFrame=face_recognition.face_locations(imgS)
			encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
			for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
				matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
				faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
				matchIndex = np.argmin(faceDis)

				if matches[matchIndex]:
					rec_name = classNames[matchIndex].upper()

					cv2.imwrite(img_name, frame1)
					date = date.today()
					image = converted_image() 
					if not ((date,rec_name) in visited):
						visited.append((date,rec_name))

						Table_Query = "SELECT name,roll_number,email,phone_number FROM student_details WHERE LOWER( name ) LIKE  '%%{}%%'".format(rec_name)
						cursor = connection.cursor(buffered=True)
						result = cursor.execute(Table_Query)
						record = cursor.fetchall()
						connection.commit()
						(name,roll_no,email,phone)=record[0]
						print(record)
						


						Table_Query = "INSERT INTO fine_details (sno, date, roll_no, name, fine,captured_image) VALUES (%s,%s,%s,%s,%s,%s) "
						val=(image_serial,date,roll_no,rec_name,"50",image)
						cursor = connection.cursor()
						result = cursor.execute(Table_Query,val)
						connection.commit()

						
						print(record)
						sub= "Please wear a mask"
						text= " This is to inform that, You,{} (Roll no {}) were spotted not wearing a mask. It is not advised to roam in the college premises without wearing a mask. Please follow safety protocols to avoid further inconvenience.\n Please report to academic cell, and kindly pay the fine amount of Rs.50/- at the earliest.".format(name, roll_no)
						message = 'Subject: {}\n\n{}'.format(sub, text)
						mail= smtplib.SMTP("smtp.gmail.com", 587)
						mail.ehlo()
						mail.starttls()
						mail.login("saurabhjohn17@gmail.com", "fwjoyyruccnfbsgb")
						mail.sendmail("saurabhjohn17@gmail.com",email, message)
						print("Mail sent")
						mail.close
						exit()
						
			
			

			image_count=0
			image_serial+=1
		elif label=='Mask':
			image_count=0
		else:
			image_count+=1
			print("Img_count:", image_count)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
