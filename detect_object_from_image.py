import cv2 as cv

# open file
filename = 'images/WIN_20240917_22_53_20_Pro.jpg'
img = cv.imread(filename)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# load object to detect
obj_path = 'data/haarcascade_frontalcatface.xml'
obj = cv.CascadeClassifier(obj_path)
#obj.load(cv.samples.findFile(obj_path))

# detect object on image
obj_img = obj.detectMultiScale(img_gray, 1.01, 7)
for (x, y, w, h) in obj_img:
    cv.rectangle(img, (x, y), (x+w-10, y+h), (250, 0, 250), 10)

#cv.rectangle(img, (0, 100), (150, 200), (250, 0, 250), 10)

# display image
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
