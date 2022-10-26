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
class_name = ['00000', '10000', '20000', '50000']
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True)
# args = vars(ap.parse_args())

# Load model da train


# Khai bao queue nhan dien
# mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
# Q = deque(maxlen=128)

# # Khoi tao cac bien
# i=0
# label = "Predicting..."

# # Doc video
# vs = cv2.VideoCapture(args["video"])

# while True:
#         # Lay anh tu video
#         ret, frame = vs.read()
#         if not ret:
#             break

#         i+=1
#         display = frame.copy()

#         # Xu ly moi 10 frame
#         if i%10==0:
#                 # Resize dua vao mang
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (224, 224)).astype("float32")
#                 frame -= mean

#                 # Du doan va dua ra ket qua
#                 preds = model.predict(np.expand_dims(frame, axis=0))[0]
#                 Q.append(preds)

#                 # Tinh trung binh cong
#                 results = np.array(Q).mean(axis=0)

#                 # Lay lop lon nhat va gan label
#                 i = np.argmax(results)
#                 label = lb.classes_[i]

#         # Hien thi len video
#         text = "I'm watching: {}".format(label)
#         cv2.putText(display, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#         # show the output image
#         cv2.imshow("Output", display)
#         if cv2.waitKey(1) & 0xFF ==ord("q"):
#                 break

# vs.release()
def get_model():
    model_resnet_conv = ResNet50(
        weights="imagenet", include_top=False,	input_tensor=Input(shape=(224, 224, 3)))


    output_resnet_conv = model_resnet_conv.output
    
# Them cac layer FC va Dropout
    output_resnet_conv = AveragePooling2D(pool_size=(7, 7))(output_resnet_conv)
    output_resnet_conv = Flatten(name="flatten")(output_resnet_conv)
    output_resnet_conv = Dense(512, activation="relu")(output_resnet_conv)
    output_resnet_conv = Dropout(0.5)(output_resnet_conv)
    output_resnet_conv = Dense(7, activation="softmax")(output_resnet_conv)

# Dong bang cac layer
    for layer in model_resnet_conv.layers:
        layer.trainable = False
# Compile
    my_model = Model(inputs=model_resnet_conv.input,
                     outputs=output_resnet_conv)
    my_model.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])
    return my_model
model = get_model()
model.load_weights("weights-01-0.99.hdf5")
while (True):
    # Capture frame-by-frame
    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None, fx=0.5, fy=0.5)
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0], axis=0))
    if (np.max(predict) >= 0.8) and (np.argmax(predict[0]) != 0):

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
