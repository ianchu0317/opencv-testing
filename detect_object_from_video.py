import cv2 as cv

# Open video
cam = cv.VideoCapture('videos/WIN_20240917_22_11_05_Pro.mp4')

if not cam.isOpened():
    print("Can't open video file")
    exit()

# Object to detect
obj_path = ""
obj = cv.CascadeClassifier()
if not obj.load(cv.samples.findFile(obj_path)):
    print("Can't open object cascade xml")

# Play video
while True:
    ret, frame = cam.read()

    if not ret:
        print("Frame problem!")
        break

    # detect obj
    obj_frame = obj.detectMultiScale(frame)

    # draw obj rectangle
    for (x, y, w, h) in obj_frame:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Object detection
    cv.imshow("object detection", frame)

    # quit program
    if cv.waitKey(0) == ord('q'):
        print("Exit program")
        break

cam.release()
cv.destroyAllWindows()

