import cv2, dlib

image = cv2.imread("images/Trump.jpg")
hog_face_detector = dlib.get_frontal_face_detector()
faces_hog = hog_face_detector(image, 1)
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imshow("face detection with dlib", image)
cv2.waitKey()
cv2.destroyAllWindows()