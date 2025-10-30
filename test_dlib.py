import cv2
import dlib
import numpy as np

print("Testing dlib with camera...")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Camera not working!")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.convertScaleAbs(gray)

print(f"gray.dtype = {gray.dtype}")  # MUST be uint8

detector = dlib.get_frontal_face_detector()
rects = detector(gray, 0)

print(f"Found {len(rects)} face(s)!")
for rect in rects:
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()