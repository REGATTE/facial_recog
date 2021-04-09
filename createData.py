import cv2
import dlib
import os

def cehckPath(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
cam = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

face_id = 1
count = 0

cehckPath("/Users/regatte/Desktop/currentProjects/facial_recog/images/data")

while(True):
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(frame, 1)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        count += 1
        cv2.imwrite("/Users/regatte/Desktop/currentProjects/facial_recog/images/data/user." + str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
        cv2.imshow('creating datasets', frame)
    if cv2.waitKey(100) & 0xff ==27:
        break
    elif count > 100:
        break
cam.release()
cv2.destroyAllWindows()