import cv2

face_recog = cv2.CascadeClassifier("HAAR_cascades/haarcascade_frontalface_default.xml")

img = cv2.imread("images/Trump.jpg")
gray = cv2.imread("images/Trump.jpg", 0)
faces = face_recog.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 3)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
cv2.imshow("live feed", img)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()