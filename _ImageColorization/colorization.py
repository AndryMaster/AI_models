from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, UpSampling2D, Reshape, concatenate, Input, RepeatVector
from keras.activations import relu, tanh
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import mse
import tensorflow as tf

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from load_dataset import load_data


def get_output_lab(img_l):
    # print((np.array([np.expand_dims(img_l, axis=-1)]), create_inception_embedding([img_l])))
    output = model.predict((np.array([np.expand_dims(img_l, axis=-1)]), create_inception_embedding([img_l])),
                           batch_size=1, verbose=0)[0] * 128
    ab = np.clip(output, -128, 127)
    result = np.zeros((224, 224, 3), dtype=np.float32)
    result[:, :, 0] = img_l
    result[:, :, 1:] = ab
    return result


def imshow_lab(image, title='', show=False):
    image[:, :, 0] = np.clip(image[:, :, 0], 0, 100)
    image[:, :, 1:] = np.clip(image[:, :, 1:]*128, -128, 127)
    plt.imshow(lab2rgb(image))
    if title:
        plt.title(title)
    if show:
        plt.show()


def compare_images_lab(image, res, save_path=None, is_add_gray=0):
    plt.figure(figsize=(3, 9))
    plt.subplot(2 + is_add_gray, 1, 1)
    imshow_lab(image, title='Original')
    plt.axis('off')
    plt.subplot(2 + is_add_gray, 1, 2)
    imshow_lab(res, title='Result')
    plt.axis('off')
    if is_add_gray:
        plt.subplot(3, 1, 3)
        image_gray = np.clip(image[:, :, 0] / 100, 0, 1)
        plt.imshow(image_gray, cmap='gray')
        plt.axis('off')
        plt.title('Gray')
    plt.show()

    if save_path:
        img2save = Image.fromarray(np.array(lab2rgb(res) * 255).astype(np.uint8))
        img2save.save(save_path)


def test_random(arr, count=5):
    for n in np.random.randint(0, len(arr), count):
        result = get_output_lab(arr[n, :, :, 0])
        compare_images_lab(arr[n], result, is_add_gray=1)


def create_inception_embedding(grayscale_batch):
    rgb_gray_images = []
    for img in grayscale_batch:
        rgb_img = np.stack((img,) * 3, axis=-1)
        img_resized = tf.image.resize(rgb_img, (299, 299))
        rgb_gray_images.append(img_resized)
    rgb_gray_images = np.array(rgb_gray_images)
    rgb_gray_images = preprocess_input(rgb_gray_images)
    result = inception_res_net_model.predict(rgb_gray_images, verbose=0)
    return result


def image_a_b_gen(batch_size):
    for batch in data_generator.flow(train, batch_size=batch_size):  # numpy
        batch_X = batch[:, :, :, 0]
        batch_Y = batch[:, :, :, 1:]
        yield [np.expand_dims(batch_X, axis=-1), create_inception_embedding(batch_X * 2.55)], batch_Y


BATCH_SIZE = 10
train, test = load_data(train_range=(0, 5000), test_range=(20000, 20100))

data_generator = ImageDataGenerator(  # validation_split=0.1
    shear_range=0,        # 0.2
    zoom_range=0,         # 0.2
    rotation_range=0,      # 0.15
    horizontal_flip=False)   # True

interpolation = 'bilinear'
inception_res_net_model = InceptionResNetV2()
inception_res_net_model.trainable = False
# 'nearest'-, 'bicubic'-, 'lanczos5'+-, 'bilinear'++-, 'gaussian'++-, 'mitchellcubic'++-

image_input = Input(shape=(224, 224, 1), batch_size=BATCH_SIZE)
encoder_output = Conv2D(64, kernel_size=3, activation=relu, padding='same', strides=2)(image_input)
encoder_output = Conv2D(128, kernel_size=3, activation=relu, padding='same')(encoder_output)
encoder_output = Conv2D(128, kernel_size=3, activation=relu, padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, kernel_size=3, activation=relu, padding='same')(encoder_output)
encoder_output = Conv2D(256, kernel_size=3, activation=relu, padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, kernel_size=3, activation=relu, padding='same')(encoder_output)
encoder_output = Conv2D(512, kernel_size=3, activation=relu, padding='same')(encoder_output)
encoder_output = Conv2D(256, kernel_size=3, activation=relu, padding='same')(encoder_output)

inception_output = Input(shape=(1000,), batch_size=BATCH_SIZE)
fusion_output = RepeatVector(28 * 28)(inception_output)
fusion_output = Reshape(([28, 28, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3)
fusion_output = Conv2D(256, kernel_size=1, activation=relu, padding='same')(fusion_output)

decoder_output = Conv2D(128, kernel_size=3, activation=relu, padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2), interpolation=interpolation)(decoder_output)
decoder_output = Conv2D(64, kernel_size=3, activation=relu, padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2), interpolation=interpolation)(decoder_output)
decoder_output = Conv2D(32, kernel_size=3, activation=relu, padding='same')(decoder_output)
decoder_output = Conv2D(16, kernel_size=3, activation=relu, padding='same')(decoder_output)
decoder_output = Conv2D(2, kernel_size=3, activation=tanh, padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2), interpolation=interpolation)(decoder_output)

model = Model(inputs=[image_input, inception_output], outputs=decoder_output)
model.compile(optimizer=Adam(), loss=mse, metrics=['accuracy'])
model.summary()

model.fit_generator(image_a_b_gen(BATCH_SIZE), steps_per_epoch=len(train) // BATCH_SIZE, epochs=5, verbose=1)
# model.fit(x=train_X, y=train_Y, validation_data=(test_X, test_Y), batch_size=BATCH_SIZE, epochs=3, verbose=1)

model.save('models/color_model5.h5', save_format='h5')
# model = load_model('models/color_model5.h5')

test_random(train, 5)
test_random(test, 2)
