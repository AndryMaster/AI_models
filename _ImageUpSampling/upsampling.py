from keras.utils.image_dataset import image_dataset_from_directory
from keras.layers import Dense, Flatten, Reshape, Input, BatchNormalization, MaxPooling2D,\
                         Conv2D, Conv2DTranspose, LeakyReLU, concatenate, Multiply, Add, Resizing
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.losses import mse
import tensorflow as tf
import keras

import time
import random
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def resize(img, side): return tf.image.resize(img, (side, side), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Params
SIZE_FROM = 64
SIZE_TO = 128
# hidden_dim = 60
batch_size = 32
SEED = random.randint(0, 1000)
EPOCHS = 3

# Dataset
dataset128 = image_dataset_from_directory("../dataset/class_img",
                                          labels="inferred", label_mode="categorical",
                                          batch_size=batch_size, shuffle=True, seed=SEED,
                                          validation_split=0.1, subset="training",
                                          image_size=(SIZE_TO, SIZE_TO), crop_to_aspect_ratio=True,
                                          interpolation="bilinear")
dataset128 = dataset128.map(lambda img, lab: (img / 255, lab))
dataset64 = dataset128.map(lambda img, lab: (resize(img, SIZE_FROM), lab))


def show_ds_faces(count=1):
    for batch, lab in dataset64.take(count):
        print(batch.shape, lab[0], np.max(batch), np.min(batch))
        for i in range(count):
            plt.imshow(batch[i])
            # plt.axis("off")
            plt.show()
        break


show_ds_faces(count=3)

# Model
input_img = Input(shape=(SIZE_FROM, SIZE_FROM, 3))
conv_img = Conv2D(64, kernel_size=3, padding='same')(input_img)
conv_tran_img = Conv2DTranspose(64, kernel_size=5, strides=(2, 2), padding='same')(conv_img)
conv_tran_img = Conv2DTranspose(1, kernel_size=3, padding='same')(conv_tran_img)
output_img = Resizing(SIZE_TO, SIZE_TO, interpolation="nearest")(input_img)
output_img = Add()([output_img, Multiply()([conv_tran_img, 0.1])])

# Compile
opt = Adam(learning_rate=0.001)
model = Model(inputs=input_img, outputs=output_img)
model.compile(opt, loss=mse, metrics=['accuracy'])
model.summary()
