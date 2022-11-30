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


# Tao model
model_resnet_conv = ResNet50(weights="imagenet", include_top=False,	input_tensor=Input(shape=(224, 224, 3)))
output_resnet_conv = model_resnet_conv.output
# Them cac layer FC va Dropout
output_resnet_conv = AveragePooling2D(pool_size=(7, 7))(output_resnet_conv)
output_resnet_conv = Flatten(name="flatten")(output_resnet_conv)
output_resnet_conv = Dense(4096, activation="relu")(output_resnet_conv)
output_resnet_conv = Dropout(0.5)(output_resnet_conv)
output_resnet_conv = Dense(4096, activation="relu")(output_resnet_conv)
output_resnet_conv = Dropout(0.5)(output_resnet_conv)
output_resnet_conv = Dense(7, activation="softmax")(output_resnet_conv)

   # Dong bang cac layer
for layer in model_resnet_conv.layers:
    layer.trainable = False
# Compile
my_model = Model(inputs=model_resnet_conv.input,outputs=output_resnet_conv)
opt = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9,decay=1e-4)
my_model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

trainAug = ImageDataGenerator(rotation_range=30,	zoom_range=0.15,	width_shift_range=0.2,	height_shift_range=0.2,	shear_range=0.15,	horizontal_flip=True,
                              fill_mode="nearest")

valAug = ImageDataGenerator()


Resnet = my_model.fit_generator(trainAug.flow(X_train, y_train, batch_size=32),	steps_per_epoch=len(X_train) // 32,	validation_data=valAug.flow(X_test, y_test),	validation_steps=len(X_test) // 32,epochs=25,callbacks=callbacks_list)

my_model.save("resnetmodel.h5")
def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) // 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) // 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    plt.savefig('roc.png')


plot_model_history(Resnet)