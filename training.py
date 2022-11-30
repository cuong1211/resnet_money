from inspect import ArgSpec
from token import LBRACE
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications.resnet import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import gradient_descent_v2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import ImageDataGenerator


raw_folder = "data/"


def save_data(raw_folder=raw_folder):

    dest_size = (128, 128)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    # Lặp qua các folder con trong thư mục raw
    for folder in os.listdir(raw_folder):
        if folder != '.DS_Store':
            print("Folder=", folder)
            # Lặp qua các file trong từng thư mục chứa các em
            for file in os.listdir(raw_folder + folder):
                if file != '.DS_Store':
                    print("File=", file)
                    pixels.append(cv2.resize(cv2.imread(
                        raw_folder + folder + "/" + file), dsize=(224, 224)))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)  # .reshape(-1,1)

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    print(labels)

    file = open('pix.data', 'wb')
    # dump information to that file
    pickle.dump((pixels, labels), file)
    # close the file
    file.close()

    return


def load_data():
    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels


# save_data()
X, y = load_data()
# random.shuffle(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)

print(X_train.shape)
print(y_train.shape)

# Chia train/val theo ty le 80/20
# (trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.2, stratify=labels, random_state=42)

# Augment anh input
# trainAug = ImageDataGenerator(	rotation_range=30,	zoom_range=0.15,	width_shift_range=0.2,	height_shift_range=0.2,	shear_range=0.15,	horizontal_flip=True,
# 	fill_mode="nearest")

# valAug = ImageDataGenerator()

# mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# trainAug.mean = mean
# valAug.mean = mean

# # Su dung mang ResNet50 voi weight imagenet
# baseModel = ResNet50(weights="imagenet", include_top=False,	input_tensor=Input(shape=(224, 224, 3)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(512, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
# model = Model(inputs=baseModel.input, outputs=headModel)

# # Dong bang cac lop cua mang Resnet
# for layer in baseModel.layers:
# 	layer.trainable = False

# opt = gradient_descent_v2(lr=1e-4, momentum=0.9, decay=1e-4 / ArgSpec["epochs"])
# model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])

# # Train model
# H = model.fit_generator(
# 	trainAug.flow(trainX, trainY, batch_size=32),	steps_per_epoch=len(trainX) // 32,	validation_data=valAug.flow(testX, testY),	validation_steps=len(testX) // 32,
# 	epochs=25)

# # Luu model
# model.save("models/sport.h5")

# # Luu ten class
# f = open("models/lb.pickle", "wb")
# f.write(pickle.dumps(lb))
# f.close()

# Tao model
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
filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

trainAug = ImageDataGenerator(rotation_range=30,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean
# construct the training image generator for data augmentation
H = my_model.fit_generator(
    trainAug.flow(X_train, y_train, batch_size=32),	steps_per_epoch=len(X_train) // 32,	validation_data=valAug.flow(X_test, y_test),	validation_steps=len(X_test) // 32,
    epochs=25)

my_model.save("vggmodel.h5")
