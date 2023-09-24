from keras.initializers.initializers_v2 import RandomNormal
from keras.layers import Input, LeakyReLU, Layer
from keras.activations import tanh
from keras.models import Model, Sequential
from keras.optimizers import Optimizer
import keras.backend as K

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


# WGAN-GP class
class WGANGP:
    """Wasserstein GAN Gradient Penalty"""
    KWARGS_INIT = {'kernel_initializer': RandomNormal(mean=0., stddev=0.02)}
    KWARGS_CONV = KWARGS_INIT | {'padding': 'same'}
    KWARGS_CONV_LReLu = KWARGS_CONV | {'activation': LeakyReLU(0.2)}

    class RandomWeightedAverage(Layer):
        """Provides a (random) weighted average between real and generated image samples"""
        def __init__(self):
            super().__init__()

        def call(self, inputs, **kwargs):
            alpha = tf.random.uniform((tf.shape(inputs[0])[0], 1, 1, 1))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

        def compute_output_shape(self, input_shape):
            return input_shape[0]

    def __init__(self,
                 wgan_generator: Sequential,
                 wgan_critic: Sequential,
                 gen_opt: Optimizer,
                 critic_opt: Optimizer,
                 gradient_penalty_weight=10.0,
                 summary=False):
        self.generator = wgan_generator
        self.critic = wgan_critic
        self.gen_optimizer = gen_opt
        self.critic_optimizer = critic_opt
        self.gradient_penalty_weight = gradient_penalty_weight

        self._is_tanh_gen = False
        if hasattr(wgan_generator.layers[-1], "activation"):
            self._is_tanh_gen = (wgan_generator.layers[-1].activation == tanh)

        self.input_dim = np.array(wgan_critic.layers[0].input.shape[1:])
        self.z_dim = np.array(wgan_generator.layers[0].input.shape[1:])[0]

        self.epochs = 0
        self.critic_losses_hist = []
        self.gen_losses_hist = []

        self._build_adversarial()

        if summary:
            self.critic_model.summary()
            self.generator_model.summary()

    @staticmethod
    def wasserstein(y_true, y_pred):
        """Computes wasserstein loss on models"""
        return -K.mean(y_true * y_pred)

    @staticmethod
    def gradient_penalty_loss(y_true, y_pred, interpolated_img_samples):
        """Computes gradient penalty based on prediction and weighted real / fake samples"""
        gradients = K.gradients(y_pred, interpolated_img_samples)[0]
        gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis=np.arange(1, len(gradients.shape))))  # Евклид len
        gradient_penalty = K.square(1 - gradient_l2_norm)  # y_true - gradient_l2_norm
        return K.mean(gradient_penalty)

    @staticmethod
    def set_trainable(self_model, bool_val):
        self_model.trainable = bool_val
        for layer in self_model.layers:
            layer.trainable = bool_val

    def _build_adversarial(self):
        # -------------------------------
        # Construct Computational Graph for the *Critic*
        # -------------------------------

        # Freeze wgan_generator'y layers while training critic
        self.set_trainable(self.critic, True)
        self.set_trainable(self.generator, False)

        # Image input (real sample)
        real_img = Input(shape=self.input_dim)

        # Fake image
        z_disc = Input(shape=(self.z_dim,))
        fake_img = self.generator(z_disc)

        # critic determines validity of the real and fake images
        fake_output = self.critic(fake_img)
        real_output = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = WGANGP.RandomWeightedAverage()(
            [real_img, fake_img])  # Lambda(self.interpol)([real_img, fake_img])
        # Determine validity of weighted sample
        interpolated_output = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional 'interpolated_img_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, interpolated_img_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty_loss'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[real_output, fake_output, interpolated_output])  # interpolated_output
        self.critic_model.compile(
            loss=[self.wasserstein, self.wasserstein, partial_gp_loss],  # partial_gp_loss
            loss_weights=[1, 1, self.gradient_penalty_weight],  # self.gradient_penalty_weight
            optimizer=self.critic_optimizer)

        # -------------------------------
        # Construct Computational Graph for *Generator*
        # -------------------------------

        # For the wgan_generator we freeze the critic layers
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Sampled noise for input to wgan_generator
        model_input = Input(shape=(self.z_dim,))
        # Generate images based of noise
        img = self.generator(model_input)
        # Discriminator determines validity
        model_output = self.critic(img)
        # Defines wgan_generator self_model
        self.generator_model = Model(model_input, model_output)
        self.generator_model.compile(optimizer=self.gen_optimizer, loss=self.wasserstein)

        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size, count_batches):
        valid = np.ones([batch_size, 1], dtype=np.float32)
        fake = -np.ones([batch_size, 1], dtype=np.float32)
        dummy = np.zeros([batch_size, 1], dtype=np.float32)  # Dummy for gradient penalty
        c_loss = 0
        for i in range(count_batches):
            true_imgs = x_train[i]
            noise = tf.random.normal([batch_size, self.z_dim])
            c_loss -= sum(self.critic_model.train_on_batch(x=[true_imgs, noise], y=[valid, fake, dummy]))
        return c_loss / count_batches

    def train_generator(self, batch_size, count_batches):
        g_loss = 0
        for i in range(count_batches):
            valid = np.ones([batch_size, 1], dtype=np.float32)
            noise = tf.random.normal([batch_size, self.z_dim])
            g_loss += self.generator_model.train_on_batch(x=noise, y=valid)
        return g_loss / count_batches

    def train(self, dataset: np.array, epochs: int, batch_size: int, batches_before_swap=3,
              run_folder='', every_n_epochs=1, critic_loops=5):
        # self._batch_size = BATCH_SIZE
        for epoch in range(self.epochs, self.epochs + epochs):
            print(f'Epoch: {epoch + 1}/{epochs}')
            all_steps = dataset.shape[0] // batch_size
            start_time = time.time()

            for step in range(0, all_steps, batches_before_swap):
                x_train = np.array(dataset[step * batch_size:(step + batches_before_swap) * batch_size])
                x_train = x_train.reshape([batches_before_swap, batch_size, *dataset[0].shape])

                # Critic train
                critic_loss = 0
                for _ in range(critic_loops):
                    critic_loss += self.train_critic(x_train, batch_size, batches_before_swap)
                critic_loss /= critic_loops

                # Generator train
                gen_loss = self.train_generator(batch_size, batches_before_swap)

                self.critic_losses_hist.append(critic_loss)
                self.gen_losses_hist.append(gen_loss)
                self.progress_bar(step, all_steps, suffix=f'\tgen: {gen_loss:.2f}\tcritic: {critic_loss:.2f}'
                                                          f'\ttime: {time.time() - start_time:.1f} sec')

            self.progress_bar(all_steps, all_steps, prefix='', length=30,
                              suffix=f'\ttime: {time.time() - start_time:.1f} sec')

            if epoch % every_n_epochs == 0:
                self.save_model(run_folder)

            self.epochs += 1

    def generate_images(self, count):
        noise = tf.random.normal([count, self.z_dim])
        gen_imgs = self.generator.predict(noise, steps=1+count//30)
        if self._is_tanh_gen:
            gen_imgs = (gen_imgs + 1) / 2
        return np.clip(gen_imgs, 0, 1)

    def sample_images(self, run_folder, r=4, c=4):
        gen_imgs = self.generate_images(r * c)

        fig, axs = plt.subplots(r, c, figsize=(r*4, c*4))
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[(i * c + j), :, :, :]), cmap='rgb')
                axs[i, j].axis('off')

        # fig.savefig(os.path.join(run_folder, f"images/sample_{self.epochs}.png"))
        plt.close()

    def save_model(self, folder_path):
        # self.generator_model.save(os.path.join(folder_path, "models/model_generator.h5"))
        # self.critic_model.save(os.path.join(folder_path, "models/model_critic.h5"))
        self.critic.save(os.path.join(folder_path, "models/critic.h5"))
        self.generator.save(os.path.join(folder_path, "models/generator.h5"))
        # pickle.dump(self, open(os.path.join(folder_path, "obj.pkl"), "wb" ))

    def load_weights(self, filepath=''):
        # self.generator_model.load_weights(filepath)
        self.critic.load_weights(os.path.join(filepath, "models/critic.h5"))
        self.generator.load_weights(os.path.join(filepath, "models/generator.h5"))

    @staticmethod
    def progress_bar(n_iter, n_total, prefix='Progress: ', suffix='', length=55, fill='█', lost='-'):
        percent = f"{100 * (n_iter / float(n_total)) :.1f}"
        filled_length = round(length * n_iter // n_total)
        bar = fill * filled_length + lost * (length - filled_length)
        print(f'\r{prefix}[{n_iter}/{n_total}] |{bar}| {percent}% {suffix}', end='')
        if n_iter == n_total:
            print()

