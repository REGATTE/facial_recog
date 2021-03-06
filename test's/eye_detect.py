import cv2
import sys
#change path in regards to your directory
faceCascade = cv2.CascadeClassifier("/Users/regatte/Desktop/currentProjects/facial_recog/HAAR_cascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("/Users/regatte/Desktop/currentProjects/facial_recog/HAAR_cascades/haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

while 1:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces & eyes
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    k = cv2.waitKey(30) & 0xff
    if k ==27:
        break
    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
