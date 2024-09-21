import cv2 as cv
import imutils
from os import path


# select current samples folder
positive_folder = "p"
negative_folder = "n"
current_folder = input("- Input folder to put samples [p/n]: ")
image_count = 0

# start video capture
cam = cv.VideoCapture(0)  # default camera
if not cam.isOpened():
    print("Can't use camera")
    exit()

# set width and height of windows
width = 854
height = 480
cam.set(cv.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, height)

# set ROI: Region Of Interest (rectangle)
x0, y0 = round(width*0.25), round(height*0.2)
x1, y1 = round(width*0.5), round(height*0.8)
print("Start point: ", x0, y0)
print("end point: ", x1, y1)

# main loop
while True:
    # capture frame
    ret, frame = cam.read()
    if not ret:
        print("Can't capture frame")
        break

    # get ROI
    frameROI = frame.copy()[y0:y1, x0:x1]
    frameROI_resized = imutils.resize(frameROI, width=24)
    cv.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # show frames
    cv.imshow("frameROI", frameROI_resized)
    cv.imshow("Collect training samples", frame)

    # save files pressing Key 's'
    if cv.waitKey(1) == ord("s"):
        img_path = path.join(current_folder, f"{image_count}.jpg")
        if current_folder == positive_folder:
            cv.imwrite(img_path, frameROI_resized)
        else:
            cv.imwrite(img_path, frameROI)
        print(f"image saved at {img_path}")
        image_count += 1

    # exit loop pressing 'q'
    if cv.waitKey(1) == ord("q"):
        print("Exit loop !")
        break

# cleanup program
cam.release()
cv.destroyAllWindows()
