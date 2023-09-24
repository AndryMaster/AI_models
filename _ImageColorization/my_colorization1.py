from keras.layers import Conv2D, InputLayer, UpSampling2D, MaxPooling2D, AveragePooling2D,\
    BatchNormalization, Resizing, Reshape, Concatenate, Input
from keras.activations import relu, tanh
from keras.applications import InceptionResNetV2, MobileNetV2
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.losses import mse
import tensorflow as tf

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from load_dataset import load_data, preprocess_image_old


def get_output_old(x):
    output = model.predict(x)[0] * 128
    ab = np.clip(output, -128, 127)
    size = 0  # !!!
    # print(ab.shape, ab)
    res = np.zeros(size, dtype=np.float32)
    res[:, :, 0] = np.squeeze(x)
    res[:, :, 1:] = ab
    res = lab2rgb(res)
    # print(res.shape, res)
    return res


def get_output_lab(img_l):
    output = model.predict(np.array([img_l]), verbose=0)[0] * 128
    ab = np.clip(output, -128, 127)
    res = np.zeros((*SIZE, 3), dtype=np.float32)
    res[:, :, 0] = img_l
    res[:, :, 1:] = ab
    return res


def imshow_lab(image, title='', show=False):
    image[:, :, 0] = np.clip(image[:, :, 0], 0, 100)
    image[:, :, 1:] = np.clip(image[:, :, 1:]*128, -128, 127)
    plt.imshow(lab2rgb(image))
    if title:
        plt.title(title)
    if show:
        plt.show()


def compare_images(image, res, save_path=None, gray=0):
    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2+gray, 1)
    imshow_lab(image, title='Original')
    plt.axis('off')
    plt.subplot(1, 2+gray, 2)
    imshow_lab(res, title='Result')
    plt.axis('off')
    if gray == 1:
        plt.subplot(1, 3, 3)
        plt.imshow(np.mean(lab2rgb(image), axis=2), cmap='gray')
        plt.axis('off')
        plt.title('Gray')
    plt.show()

    if save_path:
        img2save = Image.fromarray(np.array(lab2rgb(res) * 255).astype(np.uint8))
        img2save.save(save_path)


def test_random(arr, count=5):
    for i in range(count):
        n = np.random.randint(0, len(arr))
        result = get_output_lab(arr[n, :, :, 0])
        compare_images(arr[n], result, gray=1)


SIZE = (224, 224)

inception_res_net_model = MobileNetV2(include_top=True)
inception_res_net_model.trainable = False
# inception_res_net_model.summary()

# train, test = load_data(train_range=(0, 4000), test_range=(20000, 20500))
# train_X, train_Y = train[:, :, :, 0], train[:, :, :, 1:]
# test_X, test_Y = test[:, :, :, 0], test[:, :, :, 1:]

# imshow_lab(train[10], show=True)
# imshow_lab(train[100], show=True)
# imshow_lab(train[1000], show=True)
# print(train[100], np.max(train[100, :, :, 1:]), np.min(train[100, :, :, 1:]), train[100].shape)
# print(lab2rgb(train[100]))

interpolation = 'bilinear'  # UpSampling2D(interpolation)
# 'nearest'-, 'bicubic'-, 'lanczos5'-, 'bilinear'++-, 'gaussian'++-, 'mitchellcubic'++-

model1 = Sequential([
    InputLayer(input_shape=(None, None, 1)),
    Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', strides=2),
    # AveragePooling2D(pool_size=(2, 2)),  # or strides=2
    Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same', strides=2),
    # AveragePooling2D(pool_size=(2, 2)),  # or strides=2
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same', strides=2),
    # AveragePooling2D(pool_size=(2, 2)),  # or strides=2
    Conv2D(512, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same'),
    UpSampling2D(size=(2, 2), interpolation=interpolation),
    Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same'),
    UpSampling2D(size=(2, 2), interpolation=interpolation),
    Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(2, kernel_size=(3, 3), activation=tanh, padding='same'),
    UpSampling2D(size=(2, 2), interpolation=interpolation),
])

model2 = Sequential([
    InputLayer(input_shape=(*SIZE, 1)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same'),
    AveragePooling2D(pool_size=(2, 2)),  # or , strides=2
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same'),
    AveragePooling2D(pool_size=(2, 2)),  # or strides=2
    BatchNormalization(),
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(512, kernel_size=(3, 3), activation=relu, padding='same'),
    BatchNormalization(),
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same'),
    UpSampling2D(size=(2, 2), interpolation=interpolation),
    Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same'),
    UpSampling2D(size=(2, 2), interpolation=interpolation),
    Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same'),
    Conv2D(2, kernel_size=(3, 3), activation=tanh, padding='same')])

# model3
input_layer = InputLayer(input_shape=(*SIZE, 1))
x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same')(input_layer)
x = Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = AveragePooling2D()(x)
x = Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = AveragePooling2D()(x)
x = Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = Conv2D(512, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = Conv2D(512, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same')(x)

res_net_output = inception_res_net_model.predict(tf.image.grayscale_to_rgb(tf.multiply(input_layer, 2.55)))
res_net_output = Reshape(-1, 1, 1, 1000)(res_net_output)
res_net_output = Resizing(224/4, 224/4)(res_net_output)
x = tf.concat((x, res_net_output), axis=-1)

x = Conv2D(256, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = Conv2D(128, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same')(x)
x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
x = Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same')(x)
output_layer = Conv2D(2, kernel_size=(3, 3), activation=tanh, padding='same')(x)

model3 = Model(inputs=input_layer, outputs=output_layer)


model = model3
model.compile(optimizer=Adam(), loss=mse, metrics=['accuracy'])
model.summary()

# self_model.fit(x=train_X, y=train_Y, validation_data=(test_X, test_Y), BATCH_SIZE=8, epochs=3, verbose=1)
# self_model.save('models/color_model2_2.h5', save_format='h5')
# self_model = load_model('models/color_model2_2.h5')

# test_random(train, 2)
# test_random(test, 3)

# result = get_output_old(X)
# compare_images(image=img_name, res=result, save_path='results/res1.jpg', gray=1)
