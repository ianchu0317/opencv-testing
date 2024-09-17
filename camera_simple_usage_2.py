import numpy as np
import cv2 as cv

# select camera device
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot use camera !")
    exit()

# set width and height of capture
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# get width and height
print("Width is: ", cap.get(cv.CAP_PROP_FRAME_WIDTH))
print("Height is: ", cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    # capture frame
    ret, frame = cap.read()

    # check frame capture status
    if not ret:
        print("Cant receive frame")
        break

    # operation for frame
    frame = cv.cvtColor(frame, cv.COLOR_Luv2RGB)

    # display final frame
    cv.imshow('my_frame', frame)

    # wait key press to exit
    if cv.waitKey(1) == ord('q'):
        break

# release capture
cap.release()
cv.destroyAllWindows()
