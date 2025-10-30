import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import threading
import time

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye indices & thresholds
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
EYE_AR_THRESH = 0.25  # Eyes closed threshold
EYE_AR_CONSEC_FRAMES = 20  # Frames before alarm

# Alarm setup
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Download any alarm sound!

counter = 0
alarm_playing = False

def sound_alarm():
    global alarm_playing
    pygame.mixer.music.play(-1)  # Loop alarm
    alarm_playing = True
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera failed!")
        break

    # Resize frame for speed (optional but helps)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FORCE 8-bit conversion — THIS IS THE KEY
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    # DEBUG: Print dtype to confirm
    print(f"[DEBUG] gray.dtype = {gray.dtype}")  # Should say: uint8

    # Now dlib works
    rects = detector(gray, 0)

    # ... rest of your code (landmarks, EAR, etc.)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Left & right eye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
        
        # Check drowsiness
        if ear < EYE_AR_THRESH:
            counter += 1
            if counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "WAKE UP!!!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                if not alarm_playing:
                    t = threading.Thread(target=sound_alarm)
                    t.deamon = True
                    t.start()
        else:
            counter = 0
            alarm_playing = False
            pygame.mixer.music.stop()
            cv2.putText(frame, "Eyes Open - Good!", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Anti-Doze Lecture Buddy", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()