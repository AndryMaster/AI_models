import keras
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dropout, Concatenate, Layer, InputSpec, \
                         MaxPooling2D, Activation, ZeroPadding2D, Add
from keras.layers.activation import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Model, load_model
from keras.initializers.initializers_v2 import RandomNormal
from keras.optimizers import Adam

import tensorflow as tf

import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random
import time
import os


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class CycleGAN:
    def __init__(self
                 , input_dim
                 , learning_rate
                 , weight_validation
                 , weight_reconstr
                 , weight_id
                 , generator_type
                 , gen_n_filters
                 , disc_n_filters
                 , buffer_max_length=50
                 , max_gen_count=200
                 , data_loader=None
                 , summary=False):

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_validation = weight_validation
        self.weight_reconstr = weight_reconstr
        self.weight_id = weight_id
        self.generator_type = generator_type
        self.gen_n_filters = gen_n_filters
        self.disc_n_filters = disc_n_filters

        self.data_loader = data_loader
        self.zip_data_loader = None
        self.max_gen_count = self.gen_count = max_gen_count

        # Input shape
        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self.buffer_A = deque(maxlen=buffer_max_length)
        self.buffer_B = deque(maxlen=buffer_max_length)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 3)
        self.disc_patch = (patch, patch, 1)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.path = 'models/ap2or_unet-36-44-model.h5'
        self.compile_models()

        if summary:
            self.d_A.summary()
            self.d_B.summary()
            self.g_AB.summary()
            self.g_BA.summary()

    def compile_models(self):

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        self.d_A.compile(loss='mse',
                         optimizer=Adam(self.learning_rate, 0.5),
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=Adam(self.learning_rate, 0.5),
                         metrics=['accuracy'])

        # Build the generators
        if self.generator_type == 'unet':
            self.g_AB = self.build_generator_unet()
            self.g_BA = self.build_generator_unet()
        elif self.generator_type == 'resnet':
            self.g_AB = self.build_generator_resnet()
            self.g_BA = self.build_generator_resnet()
        else:
            raise AttributeError("`generator_type` is not `unet` or `resnet`")

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[self.weight_validation, self.weight_validation,
                                            self.weight_reconstr, self.weight_reconstr,
                                            self.weight_id, self.weight_id],
                              optimizer=Adam(self.learning_rate, 0.5))

        self.d_A.trainable = True
        self.d_B.trainable = True

    def build_generator_unet(self):

        def downsample(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = ReLU()(d)
            return d

        def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
            u = InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = ReLU()(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)

            u = Concatenate()([u, skip_input])
            return u

        # Image input
        img = Input(shape=self.img_shape)

        # Downsampling
        d1 = downsample(img, self.gen_n_filters)
        d2 = downsample(d1, self.gen_n_filters * 2)
        d3 = downsample(d2, self.gen_n_filters * 4)
        d4 = downsample(d3, self.gen_n_filters * 8)

        # Upsampling
        u1 = upsample(d4, d3, self.gen_n_filters * 4)
        u2 = upsample(u1, d2, self.gen_n_filters * 2)
        u3 = upsample(u2, d1, self.gen_n_filters)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output_img)

    def build_generator_resnet(self):
        def conv7(layer_input, filters, final):
            y = ZeroPadding2D(padding=(3, 3))(layer_input)  # ReflectionPadding2D
            y = Conv2D(filters, kernel_size=(7, 7), strides=1, padding='valid', kernel_initializer=self.weight_init)(y)
            if final:
                y = Activation('tanh')(y)
            else:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
                y = ReLU()(y)
            return y

        def downsample(layer_input, filters):
            y = Conv2D(filters, kernel_size=(3, 3), strides=2, padding='same',
                       kernel_initializer=self.weight_init)(layer_input)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = ReLU()(y)
            return y

        def upsample(layer_input, filters):
            # y = Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same',
            #                     kernel_initializer=self.weight_init)(layer_input)
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer=self.weight_init)(layer_input)
            y = UpSampling2D()(y)

            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = ReLU()(y)
            return y

        def residual(layer_input, filters):
            shortcut = layer_input
            y = ZeroPadding2D(padding=(1, 1))(layer_input)  # ReflectionPadding2D
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer=self.weight_init)(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = ReLU()(y)

            y = ZeroPadding2D(padding=(1, 1))(y)  # ReflectionPadding2D
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer=self.weight_init)(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)

            return Add()([shortcut, y])  # add([shortcut, y])

        # Image input
        img = Input(shape=self.img_shape)

        y = img

        y = conv7(y, self.gen_n_filters, final=False)
        y = downsample(y, self.gen_n_filters * 2)
        y = downsample(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)  # 9 ResBlocks
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = upsample(y, self.gen_n_filters * 2)
        y = upsample(y, self.gen_n_filters)
        y = conv7(y, self.channels, final=True)
        output = y

        return Model(img, output)

    def build_discriminator(self):

        def conv4(layer_input, filters, stride=2, norm=True):
            c = Conv2D(filters, kernel_size=(4, 4), strides=stride, padding='same',
                       kernel_initializer=self.weight_init)(layer_input)

            if norm:
                c = InstanceNormalization(axis=-1, center=False, scale=False)(c)

            c = LeakyReLU(0.2)(c)

            return c

        img = Input(shape=self.img_shape)

        y = conv4(img, self.disc_n_filters, stride=2, norm=False)
        y = conv4(y, self.disc_n_filters * 2, stride=2)
        y = conv4(y, self.disc_n_filters * 4, stride=2)
        y = conv4(y, self.disc_n_filters * 8, stride=1)

        output = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=self.weight_init)(y)

        return Model(img, output)

    def train_discriminators(self, imgs_A, imgs_B, valid, fake):
        # Translate images to opposite domain
        fake_B = self.g_AB.predict(imgs_A, verbose=0)
        fake_A = self.g_BA.predict(imgs_B, verbose=0)

        self.buffer_B.append(fake_B)
        self.buffer_A.append(fake_A)

        fake_A_rnd = random.sample(self.buffer_A, min(len(self.buffer_A), len(imgs_A)))
        fake_B_rnd = random.sample(self.buffer_B, min(len(self.buffer_B), len(imgs_B)))

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0]
            , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
            , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
            , d_loss_total[1]
            , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
            , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )

    def train_generators(self, imgs_A, imgs_B, valid):
        # Train the generators
        return self.combined.train_on_batch([imgs_A, imgs_B],
                                            [valid, valid,
                                             imgs_A, imgs_B,
                                             imgs_A, imgs_B])

    def train(self, epochs=1, batch_size=1, count_batches=100, run_folder=''):
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch, epochs):
            print(f'Epoch: {epoch + 1}/{epochs}')
            start_time = time.time()

            for step in range(count_batches):
                imgs_A, imgs_B = self.next()

                d_loss = self.train_discriminators(imgs_A, imgs_B, valid, fake)
                g_loss = self.train_generators(imgs_A, imgs_B, valid)
                d_loss, g_loss = sum(d_loss), sum(g_loss)

                self.progress_bar(step, count_batches, prefix='', length=30,
                                  suffix=f'\tgen: {g_loss:.2f}\tcritic: {d_loss:.2f}'
                                         f'\ttime: {time.time() - start_time:.1f} sec')
                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

            self.progress_bar(count_batches, count_batches, prefix='', length=30,
                              suffix=f'\ttime: {time.time() - start_time:.1f} sec')
            self.sample_images()
            self.save_model(run_folder)

            self.epoch += 1

    def sample_images(self):
        imgs_A, imgs_B = self.next()

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A, verbose=0)
        fake_A = self.g_BA.predict(imgs_B, verbose=0)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B, verbose=0)
        reconstr_B = self.g_AB.predict(fake_A, verbose=0)

        # ID the images
        id_A = self.g_BA.predict(imgs_A, verbose=0)
        id_B = self.g_AB.predict(imgs_B, verbose=0)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.clip(gen_imgs, 0, 1)

        r, c = 2, 4
        titles = ['Original', 'Translated', 'Reconstructed', 'Identical']
        fig, axs = plt.subplots(r, c, figsize=(25, 12.5))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
        plt.close()

    def save_model(self, run_folder):
        self.combined.save(os.path.join(run_folder, self.path))
        # self.combined.save_weights(os.path.join(run_folder, 'w'+self.path))
        # self.d_A.save(os.path.join(run_folder, 'd_A.h5'))
        # self.d_B.save(os.path.join(run_folder, 'd_B.h5'))
        # self.g_BA.save(os.path.join(run_folder, 'g_BA.h5'))
        # self.g_AB.save(os.path.join(run_folder, 'g_AB.h5'))

    def load_weights(self, filepath):
        self.combined = load_model(filepath)
        # self.combined.load_weigths(filepath)

    def next(self):
        if self.gen_count == self.max_gen_count:
            self.gen_count = 0
            self.zip_data_loader = zip(*self.data_loader)  # update zip data loader
        self.gen_count += 1
        return next(self.zip_data_loader)

    @staticmethod
    def progress_bar(n_iter, n_total, prefix='Progress: ', suffix='', length=55, fill='█', lost='-'):
        percent = f"{100 * (n_iter / float(n_total)) :.1f}"
        filled_length = round(length * n_iter // n_total)
        bar = fill * filled_length + lost * (length - filled_length)
        print(f'\r{prefix}[{n_iter}/{n_total}] |{bar}| {percent}% {suffix}', end='')
        if n_iter == n_total:
            print()
