from keras.layers import Dense, Flatten, Reshape, Input, BatchNormalization, MaxPooling2D,\
                         Conv2D, Conv2DTranspose, ReLU, Dropout
from keras.initializers.initializers_v2 import RandomNormal
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.datasets import cifar10

import random
import numpy as np
import matplotlib.pyplot as plt
# from functools import partial

import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

from WGAN_GP import WGANGP

# Params
hidden_dim = 64
batch_size = 30
EPOCHS = 3

# Dataset
dataset = np.load('../book_dataset/full_numpy_bitmap_camel.npy')
dataset = np.reshape(dataset / 127.5 - 1, newshape=(-1, 28, 28, 1))

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x = np.vstack([x_test, x_train])
# y = np.vstack([y_test, y_train])
# print(x.shape, y.shape)
# dataset = x[np.where(y_train == 7)]
# print(dataset.shape)
# dataset = np.reshape(dataset / 255., newshape=(-1, 32, 32, 3))  # 127.5 - 1
# print(dataset.shape)

# Funcs
def show_dataset(count=3):
    for i in range(count):
        plt.imshow(dataset[random.randint(0, dataset.shape[0])])
        plt.axis('off')
        plt.show()
def progress_bar(n_iter, n_total, prefix='Progress: ', suffix='', length=55, fill='█', lost='-'):
    percent = f"{100 * (n_iter / float(n_total)) :.1f}"
    filled_length = round(length * n_iter // n_total)
    bar = fill * filled_length + lost * (length - filled_length)
    print(f'\r{prefix}[{n_iter}/{n_total}] |{bar}| {percent}% {suffix}', end='')
    if n_iter == n_total:
        print()
show_dataset(count=2)
print(np.max(dataset[0]), np.min(dataset[0]))

# Models
generator = Sequential([
    Input(shape=(hidden_dim,)),
    Dense(7 * 7 * 40, activation='relu', **WGANGP.KWARGS_INIT),
    BatchNormalization(),
    Reshape((7, 7, 40)),  # 7x7
    Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), **WGANGP.KWARGS_CONV),  # 14x14
    BatchNormalization(momentum=0.8), ReLU(),
    Conv2DTranspose(96, kernel_size=(5, 5), strides=(2, 2), **WGANGP.KWARGS_CONV),  # 28x28
    BatchNormalization(momentum=0.8), ReLU(),
    Conv2DTranspose(50, kernel_size=(3, 3), strides=(1, 1), **WGANGP.KWARGS_CONV),  # 28x28
    BatchNormalization(momentum=0.8), ReLU(),
    Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')])  # 28x28

discriminator = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), strides=(1, 1), **WGANGP.KWARGS_CONV_LReLu),
    Conv2D(64, kernel_size=(3, 3), strides=(2, 2), **WGANGP.KWARGS_CONV_LReLu),
    # MaxPooling2D(),  # 14x14
    Conv2D(96, kernel_size=(3, 3), strides=(1, 1), **WGANGP.KWARGS_CONV_LReLu),
    Conv2D(128, kernel_size=(3, 3), strides=(2, 2), **WGANGP.KWARGS_CONV_LReLu),
    # MaxPooling2D(),  # 7x7
    Conv2D(192, kernel_size=(3, 3), strides=(1, 1), **WGANGP.KWARGS_CONV_LReLu),
    Conv2D(256, kernel_size=(3, 3), strides=(1, 1), **WGANGP.KWARGS_CONV_LReLu),
    Flatten(),
    Dropout(0.1),
    Dense(1, **WGANGP.KWARGS_INIT)])

generator.summary()
discriminator.summary()

# Optimizers
generator_optimizer = Adam(0.00005, beta_1=0.5)
discriminator_optimizer = Adam(0.00005, beta_1=0.5)

model = WGANGP(wgan_generator=generator, gen_opt=generator_optimizer,
               wgan_critic=discriminator, critic_opt=discriminator_optimizer)

model.train(dataset, batch_size=50, epochs=1, critic_loops=4, batches_before_swap=12)
# model.load_weights()

# Testing
# side = 4
# for i in range(3):
#     images = model.generate_images(side*side)
#
#     num = 0
#     plt.figure(figsize=(side*2, side*2))
#     for i in range(side):
#         for j in range(side):
#             plt.subplot(side, side, num+1)
#             plt.imshow(images[num].squeeze(), cmap='gray')
#             plt.axis('off')
#             num += 1
#
#     plt.show()
