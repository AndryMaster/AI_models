from keras.layers import Dense, Flatten, Reshape, Input, BatchNormalization, MaxPooling2D,\
                         Conv2D, Conv2DTranspose, LeakyReLU, ReLU, Dropout, UpSampling2D
from keras.models import Sequential, load_model
# from keras.datasets import mnist
# from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
import tensorflow as tf
import keras.backend as K
import keras

import time
import random
import numpy as np
import matplotlib.pyplot as plt

# Params
hidden_dim = 70
batch_size = 50
EPOCHS = 1

# Dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()  # / 255
# y_train, y_test = to_categorical(y_train), to_categorical(y_test)
# X_train, X_test = X_train.reshape(*X_train.shape, 1), X_test.reshape(*X_test.shape, 1)  # 60_000(10_000), 28, 28, 1
dataset = np.load('../book_dataset/full_numpy_bitmap_camel.npy')
dataset = np.reshape(dataset / 255.0, newshape=(-1, 28, 28, 1))
dataset = dataset[:30_000]
print(dataset.shape)

# Funcs
def show_dataset(count=3):
    for i in range(count):
        plt.imshow(dataset[random.randint(0, dataset.shape[0])], cmap='gray')
        plt.axis('off')
        plt.show()
def progress_bar(n_iter, n_total, prefix='Progress: ', suffix='', length=55, fill='█', lost='-'):
    percent = f"{100 * (n_iter / float(n_total)) :.1f}"
    filled_length = round(length * n_iter // n_total)
    bar = fill * filled_length + lost * (length - filled_length)
    print(f'\r{prefix}[{n_iter}/{n_total}] |{bar}| {percent}% {suffix}', end='')
    if n_iter == n_total:
        print()

# show_dataset()
print(np.max(dataset[0]), np.min(dataset[0]))

# Models
generator = Sequential([
    Input(shape=(hidden_dim,)),
    Dense(7 * 7 * 64, activation='relu'),
    BatchNormalization(),
    Reshape((7, 7, 64)),  # 7x7
    Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same'),  # 14x14
    BatchNormalization(momentum=0.8), ReLU(),
    Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same'),  # 28x28
    BatchNormalization(momentum=0.8), ReLU(),
    Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),  # 28x28
    BatchNormalization(momentum=0.8), ReLU(),
    Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')])  # 28x28

discriminator = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(64, kernel_size=(4, 4), strides=(1, 1)), BatchNormalization(), LeakyReLU(0.2),
    Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same'), BatchNormalization(), LeakyReLU(0.2),
    Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), BatchNormalization(), LeakyReLU(0.2),
    Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'), BatchNormalization(), LeakyReLU(0.2),
    Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'), BatchNormalization(), LeakyReLU(0.2),
    Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'), BatchNormalization(), LeakyReLU(0.2),
    Flatten(),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1)])  # activation='sigmoid'

generator.summary()
discriminator.summary()  # critic

# Losses and optimizers
generator_optimizer = RMSprop(0.0000_3, rho=0.7)  # rho=0.7
discriminator_optimizer = RMSprop(0.0000_3, rho=0.97, momentum=0.35)  # rho=0.97, momentum=0.35
binary_crossentropy = keras.losses.BinaryCrossentropy(from_logits=False)

def generator_loss(fake_output):
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    return K.mean(real_output * fake_output)
    # real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    # fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    # return real_loss + fake_loss

def clip_weights(model, clip_val):
    for l in model.layers:  # Ограничение Липшица
        weights = l.get_weights()
        weights = [np.clip(w, -clip_val, clip_val) for w in weights]
        l.set_weights(weights)

# Train functions
# @tf.function
def train_step(train_images):
    noises = tf.random.normal([batch_size, hidden_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        for _ in range(1):
            generated_images = generator(noises, training=True)

            real_output = discriminator(train_images, training=True)
            fake_output = discriminator(generated_images, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            clip_weights(discriminator, 0.01)

        gen_loss = generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss

def train(epochs: int):
    all_steps = dataset.shape[0] // batch_size
    train_history = np.zeros((all_steps - 1, 2), dtype=float)

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        start_time = time.time()

        for step in range(all_steps-1):
            gen_loss, disc_loss = train_step(dataset[step*batch_size:(step+1)*batch_size])
            progress_bar(step, all_steps, suffix=f'\tgen: {gen_loss:.2f}\tdis: {disc_loss:.2f}'
                                                 f'\ttime: {time.time() - start_time:.1f} sec')
            train_history[step, :] = gen_loss, disc_loss

        generator.save('models/camel/gen1.h5')
        discriminator.save('models/camel/disk1.h5')
        progress_bar(all_steps, all_steps, prefix='', length=30,
                     suffix=f'\ttime: {time.time() - start_time:.1f} sec')  # \t loss: {train_history[-1]:.3f}

# Training
# wgan_generator = load_model('models/camel/gen.h5')
# discriminator = load_model('models/camel/disk.h5')
train(EPOCHS)

# Testing
side = 4
for i in range(3):
    images = generator.predict(tf.random.normal([side*side, hidden_dim]))

    num = 0
    plt.figure(figsize=(side*2, side*2))
    for i in range(side):
        for j in range(side):
            plt.subplot(side, side, num+1)
            plt.imshow(images[num].squeeze(), cmap='gray')
            plt.axis('off')
            num += 1

    plt.show()
