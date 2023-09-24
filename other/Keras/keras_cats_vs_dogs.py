import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.python.keras.layers import DepthwiseConv2D  # !!!
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, InputLayer

from keras.models import Sequential, load_model, save_model
from keras.activations import relu, sigmoid, softmax, relu6, linear
from keras.applications import MobileNetV2
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import os

SIZE = 224
PATH = 'models/cats_vs_dogs_3.h5'

# DATASET
dataset, info = tfds.load('cats_vs_dogs', data_dir='datasets', split=['train[:90%]', 'train[90%:]'],
                          download=False, with_info=True, as_supervised=True)
classes = info.features['label'].names

def preprocess_image(img, label=''):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, size=(SIZE, SIZE))
    img = img / 128. - 1
    return img, label

def un_preprocess_image(img):
    return (img + 1) / 2

def show_train_images():
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    axs = np.ravel(axs)
    for images, labels in train_batches.take(1):
        for i, (image, label) in enumerate(zip(images, labels)):
            plt.sca(axs[i])
            plt.imshow(un_preprocess_image(image.numpy()))
            plt.title(f"{classes[label]} - {image.numpy().shape}")
            plt.axis('off')
    plt.show()

def test_model(batches=1):
    for images, labels in test_batches.take(batches):
        predictions = model.predict(images, batch_size=16, verbose=0).squeeze()
        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
        axs = np.ravel(axs)
        for i, (image, label) in enumerate(zip(images, labels)):
            plt.sca(axs[i])
            plt.imshow(un_preprocess_image(image.numpy()))
            predict = np.argmax(predictions[i])
            plt.title(f"{classes[predict]} {predictions[i]}", c='g' if predict == int(label) else 'r')
            # print(f'{classes[int(label)]} -> {classes[predict]}\t({label}) -> {predictions[i]}')
            plt.axis('off')
        plt.show()

def my_cats():
    all_imgs = os.listdir(os.path.join(os.getcwd(), 'datasets/my_cats'))
    all_path_imgs = [os.path.join(os.getcwd(), 'datasets/my_cats', path_img) for path_img in all_imgs]
    plt.figure(figsize=(10, 10))
    for path in all_path_imgs:
        img = preprocess_image(np.array([plt.imread(path)]))[0]
        prediction = model.predict(img, verbose=0).squeeze()
        plt.title(f"{classes[np.argmax(prediction)]} {prediction}", c='b')
        plt.imshow(un_preprocess_image(img[0]))
        plt.axis('off')
        plt.show()

# dataset (image, label)
train = dataset[0].map(preprocess_image)
test = dataset[1].map(preprocess_image)
train_batches = train.shuffle(buffer_size=1000).batch(16)
test_batches = test.shuffle(buffer_size=1000).batch(16)
# show_train_images()

# MODEL
mobile_net2_layer = MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False, alpha=1)
mobile_net2_layer.trainable = False
t_depth = 1
model_1 = Sequential([  # SIZE == 128
    InputLayer(input_shape=(SIZE, SIZE, 3)),
    Conv2D(64, kernel_size=4, activation=relu6),  # padding='same'
    Conv2D(64, kernel_size=4, activation=relu6),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(96, kernel_size=3, activation=relu6),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(128, kernel_size=3, activation=relu6),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(256, kernel_size=3, activation=relu6),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(512, kernel_size=3, activation=relu6),
    MaxPooling2D(pool_size=(3, 3)),
    BatchNormalization(),
    Flatten(),
    Dense(360, activation=relu),
    Dropout(0.45),
    Dense(128, activation=relu),
    Dropout(0.25),
    Dense(2, activation=softmax)
])
model_2 = Sequential([  # SIZE == 100
     Conv2D(64, kernel_size=3, activation=relu, input_shape=(SIZE, SIZE, 3)),
     BatchNormalization(),
     MaxPooling2D(),
     Conv2D(128, kernel_size=3, activation=relu),
     BatchNormalization(),
     MaxPooling2D(),
     Conv2D(256, kernel_size=3, activation=relu),
     BatchNormalization(),
     MaxPooling2D(),
     Conv2D(512, kernel_size=3, activation=relu),
     BatchNormalization(),
     MaxPooling2D(),
     Flatten(),
     Dense(512, activation=relu),
     Dropout(0.5),
     Dense(128, activation=relu),
     Dropout(0.25),
     Dense(2, activation=softmax)
])
model_3 = Sequential([  # SIZE == 224; un_preprocess_image
    mobile_net2_layer,
    GlobalMaxPooling2D(),  # Flatten(),
    Dropout(0.2),
    Dense(512, activation=sigmoid),
    Dropout(0.15),
    Dense(2, activation=softmax)
])
model_4 = Sequential([  # SIZE == 224; un_preprocess_image
    InputLayer(input_shape=(SIZE, SIZE, 3)),

    Conv2D(32, kernel_size=3, activation=relu6, padding='same', strides=(2, 2)),
    Dropout(0.1),

    DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    Conv2D(64, kernel_size=1, activation=relu6, padding='same'),
    DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth, strides=(2, 2)),
    BatchNormalization(),
    Dropout(0.15),

    Conv2D(128, kernel_size=1, activation=relu6, padding='same'),
    # DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    # Conv2D(128, kernel_size=1, activation=relu6, padding='same'),
    DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth, strides=(2, 2)),
    BatchNormalization(),
    Dropout(0.15),

    Conv2D(256, kernel_size=1, activation=relu6, padding='same'),
    # DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    # Conv2D(256, kernel_size=1, activation=relu6, padding='same'),
    DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth, strides=(2, 2)),
    BatchNormalization(),

    Conv2D(512, kernel_size=1, activation=relu6, padding='same'),
    # DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    # Conv2D(512, kernel_size=1, activation=relu6, padding='same'),
    # DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    # Conv2D(512, kernel_size=1, activation=relu6, padding='same'),
    # DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    # Conv2D(512, kernel_size=1, activation=relu6, padding='same'),
    # DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    # Conv2D(512, kernel_size=1, activation=relu6, padding='same'),
    DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth),
    Conv2D(512, kernel_size=1, activation=relu6, padding='same'),
    DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth, strides=(2, 2)),
    BatchNormalization(),

    Conv2D(1024, kernel_size=1, activation=relu6, padding='same'),
    DepthwiseConv2D(kernel_size=3, activation=relu6, padding='same', depth_multiplier=t_depth, strides=(2, 2)),
    Conv2D(1024, kernel_size=1, activation=relu6, padding='same'),
    BatchNormalization(),

    # BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(1000, activation=relu),
    Dropout(0.3),
    Dense(2, activation=softmax)
])

model = model_3
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.build()
model.summary()

# self_model.fit(train_batches, validation_data=test_batches, epochs=3)
# self_model.save(PATH)
model = load_model(PATH)


# TEST
test_loss, test_acc = model.evaluate(test_batches, verbose=2)
print(f'Accuracy: {test_acc * 100 :.2f}%')
# test_model(batches=5)
# my_cats()
