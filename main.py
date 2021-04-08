import cv2
import numpy as np
import tensorflow as tf
from tf.keras.models import load_model

face_cascader = cv2.CascadeClassifier("HAAR_cascades/haarcascade_frontalface_default.xml")
eye_cascader = cv2.CascadeClassifier("HAAR_cascades/haarcascade_eye.xml")

gender_model = load_model("")

#webcam
cap = cv2.VideoCapture(0)
