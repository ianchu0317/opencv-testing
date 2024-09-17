from __future__ import print_function
import cv2 as cv

def detectAndDisplay(frame):
    #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame)

    # If detected face, it will return (x, y) and width and height
    for (x, y, w, h) in faces:
        # Operación int // int = int
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 255, 255), 4)

        print(frame)
        faceROI = frame[y:y+h,x:x+w]

        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
 
    cv.imshow('Capture - Face detection', frame)
 

# Set file source to match
face_cascade_name = "haarcascade_frontalcatface.xml"
eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml"
 
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
 
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

#-- 2. Read the video stream
camera_device = 0
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

# Main loop
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
 
    detectAndDisplay(frame)
 
    if cv.waitKey(1) == ord("q"):
        break

# Exit loop and camera usage
cap.release()
cv.destroyAllWindows()