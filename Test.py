import cv2
import os
import numpy as np

subjects = ["", "Ross", "Joey", "Monica", "Phoebe", "Rachel", "Chandler"];

def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier('/home/kamila/opencv_build/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7);
	if(len(faces))==0:
		return None, None
	i = 0
	for face_det in faces:
		(x, y, w, h) = faces[i]
		i = i+1
	return gray, faces

def draw_rectangle(img, rect):
	i = 0
	for any in rect:
		(x, y, w, h) = rect[i]
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		i = i+1	

def prepare_training_data(data_folder_path):
	dirs=os.listdir(data_folder_path)
	faces = []
	labels = []
	for dir_name in dirs:
		if not dir_name.startswith("s"):
			continue;
		label = int(dir_name.replace("s", ""))
		subject_dir_path = data_folder_path + "/" + dir_name
		subject_images_names = os.listdir(subject_dir_path)
		for image_name in subject_images_names:
			if image_name.startswith("."):
				continue;
			image_path = subject_dir_path + "/" + image_name
			image = cv2.imread(image_path)
			face, rect = detect_face(image)
			if face is not None: 
				faces.append(face)
				labels.append(label)
				draw_rectangle(image, rect)
				cv2.imshow("Training on image...", cv2.resize(image, (640, 480)))
				cv2.waitKey(100) #0 для проверки
			if face is None:
				print("ERROR   " + image_path)
		cv2.destroyAllWindows()
		cv2.waitKey(1)
		cv2.destroyAllWindows()
	return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
print("\n")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
print("\n")

face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(np.array(faces), np.array(labels))
	
def predict(test_img):
	img = test_img.copy()
	face, rect = detect_face(img)
	for (x, y, w, h) in rect:
		confidence = 0.0
		label, confidence = face_recognizer.predict(cv2.resize(face[y:y+h, x:x+w], (960, 540)))
		cv2.putText(img, subjects[label], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
		# cv2.putText(img, subjects[label] + str(confidence), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	draw_rectangle(img, rect)
	return img

print("Predicting images...")
test_img1 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test1.jpg")
predicted_img1 = predict(test_img1)
cv2.imshow("Predicted image", cv2.resize(predicted_img1, (640, 480)))
cv2.waitKey(0)

test_img2 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test2.jpg")
predicted_img2 = predict(test_img2)
cv2.imshow("Predicted image", cv2.resize(predicted_img2, (640, 480)))
cv2.waitKey(0)

test_img3 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test3.jpg")
predicted_img3 = predict(test_img3)
cv2.imshow("Predicted image", cv2.resize(predicted_img3, (640, 480)))
cv2.waitKey(0)

test_img4 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test4.jpg")
predicted_img4 = predict(test_img4)
cv2.imshow("Predicted image", cv2.resize(predicted_img4, (640, 480)))
cv2.waitKey(0)

test_img5 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test5.jpg")
predicted_img5 = predict(test_img5)
cv2.imshow("Predicted image", cv2.resize(predicted_img5, (640, 480)))
cv2.waitKey(0)

test_img6 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test6.jpg")
predicted_img6 = predict(test_img6)
cv2.imshow("Predicted image", cv2.resize(predicted_img6, (640, 480)))
cv2.waitKey(0)

test_img7 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test7.jpg")
predicted_img7 = predict(test_img7)
cv2.imshow("Predicted image", cv2.resize(predicted_img7, (640, 480)))
cv2.waitKey(0)

test_img8 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test8.jpg")
predicted_img8 = predict(test_img8)
cv2.imshow("Predicted image", cv2.resize(predicted_img8, (640, 480)))
cv2.waitKey(0)

test_img9 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test9.jpg")
predicted_img9 = predict(test_img9)
cv2.imshow("Predicted image", cv2.resize(predicted_img9, (640, 480)))
cv2.waitKey(0)

test_img10 = cv2.imread("/home/kamila/OpenCV_LIRS/FriendsTest/test10.jpg")
predicted_img10 = predict(test_img10)
print("Prediction complete")
cv2.imshow("Predicted image", cv2.resize(predicted_img10, (640, 480)))
cv2.waitKey(0)
cv2.destroyAllWindows()
