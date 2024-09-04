from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
##import winsound
import serial

ser = serial.Serial('COM3',baudrate=9600,timeout=1)
##import matplotlib.pyplot as plt
frequency = 2500
duration = 3000

def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouthAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[11])
    B = dist.euclidean(eye[2], eye[10])
    C = dist.euclidean(eye[3], eye[9])
    D = dist.euclidean(eye[4], eye[8])
    E = dist.euclidean(eye[5], eye[7])
    F = dist.euclidean(eye[0], eye[6])
    ear = ((A+B+C+D+E+F) / 6)
    return ear
count = 0
earThresh = 0.3 #distance between vertical eye coordinate Threshold
earFrames = 5 #consecutive frames for eye closure
shapePredictor = "shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

#get the coord of left & right eye & mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#get the coord of mouth
(Lmouth, Rmouth) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        Mouth = shape[Lmouth:Rmouth]
        Mouth_1 = mouthAspectRatio(Mouth)
        mar = Mouth_1
        print(mar)

        mouthHull = cv2.convexHull(Mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        

        if ear < earThresh:
               count += 1
               print(count)
               if count >= earFrames:
                    cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    ser.write('1'.encode())
                    
                    cv2.imwrite("capture.jpg",frame)
               else:
                    ser.write('2'.encode())
##                    winsound.Beep(frequency, duration)
##                email()
        else:
            count = 0
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()

