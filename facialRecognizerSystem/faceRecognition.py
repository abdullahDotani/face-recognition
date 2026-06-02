import cv2
import face_recognition

print("Loading image...")
img = cv2.imread("faces/face1.jpg")

print("Converting colors...")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Scanning for face...")
face_loc = face_recognition.face_locations(img_rgb)[0]

print("Drawing box...")
cv2.rectangle(img, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 0), 2)

cv2.imshow('Face Detection Target', img)
cv2.waitKey(0)