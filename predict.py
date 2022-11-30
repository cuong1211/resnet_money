# import the necessary packages
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from os import listdir
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import load_model
import sys
from keras.applications.resnet import ResNet50
from keras.layers.pooling import AveragePooling2D

cap = cv2.VideoCapture(0)

# Dinh nghia class
class_name = ['0000','10000','100000','20000','200000','50000','500000']
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True)
# args = vars(ap.parse_args())

# Load model da train
model = load_model('resnetmodel.h5')

while (True):
    # Capture frame-by-frame
    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None, fx=0.5, fy=0.5)
    # Resize
    image = image_org.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)).astype("float32")
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0], axis=0))
    if (np.max(predict) >= 0.5) and (np.argmax(predict[0]) != 0):

        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
