import cv2
import numpy as np
import tensorflow as tf
import glob, os, random
from tf.keras.preprocessing.image import img_to_array
#general param
epochs = 100
batch_size = 94
image_dim = (100, 100, 3)

df = []
labels = []

data = [f for f in glob.glob(r'images/datasets/gender_dataset_face'+'/**/*', recursive=True) if not os.path.isdir(f)]
random.shuffle(data)


for img in data:
    image = cv2.imread(img)
    sized_image = cv2.resize(image, image_dim[0], image_dim[1])
    array_image = img_to_array(sized_image)
    df.append(array_image)
    
    label = img.split(os.path.sep)[-2]
    if label == woman:
        label = 1
    else:
        label = 0
    labels.append(label)