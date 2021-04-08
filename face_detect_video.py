import cv2
import numpy as numpy

face_recog = cv2.CascadeClassifier("HAAR_cascades/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_recog.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow("live feed", img)
    k = cv2.waitKey(30) & 0xff
    #press esc to quit
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()