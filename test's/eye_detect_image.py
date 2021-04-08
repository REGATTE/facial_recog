import cv2
faceCascade = cv2.CascadeClassifier("/Users/regatte/Desktop/currentProjects/facial_recog/HAAR_cascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("/Users/regatte/Desktop/currentProjects/facial_recog/HAAR_cascades/haarcascade_eye.xml")
img = cv2.imread("/content/Trump.jpg")
gray = cv2.imread("/content/Trump.jpg", 0)
faces = face_recog.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow("result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()