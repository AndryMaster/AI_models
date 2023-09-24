import os
import time
import pickle
import numpy as np
# import matplotlib.pyplot as plt

from keras.layers import Input, Flatten, Dense, Conv2DTranspose, Lambda, \
    Activation, BatchNormalization, LeakyReLU, ReLU, Reshape, Concatenate, Conv3D

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.initializers.initializers_v2 import RandomNormal

from music21 import note, stream, duration, tempo


class MuseGAN:
    def __init__(self,
                 input_dim,
                 critic_learning_rate,
                 generator_learning_rate,
                 optimiser,
                 z_dim,
                 n_tracks,
                 n_bars,
                 n_steps_per_bar,
                 n_pitches):

        self.name = 'gan'

        self.input_dim = input_dim

        self.critic_learning_rate = critic_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.optimiser = optimiser

        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches

        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self.binary_crossentropy = BinaryCrossentropy(from_logits=False)
        # self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self._build_critic()
        self._build_generator()

        self._build_adversarial()

        self.generator.summary()
        self.critic.summary()

    @staticmethod
    def get_activation(activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha=0.2)
        else:
            layer = Activation(activation)
        return layer

    @staticmethod
    def set_activation(layer, activation):
        if activation == 'lrelu':
            return LeakyReLU(alpha=0.2)(layer)
        elif activation == 'relu':
            return ReLU()(layer)
        return layer

    def conv(self, x, f, k, s, p, bn=False):
        x = Conv3D(filters=f, kernel_size=k, padding=p, strides=s, kernel_initializer=self.weight_init)(x)
        if bn:
            x = BatchNormalization(momentum=0.9)(x)
        return LeakyReLU(alpha=0.2)(x)

    def _build_critic(self):

        critic_input = Input(shape=self.input_dim, name='critic_input')
        x = critic_input

        x = self.conv(x, f=128, k=(2, 1, 1), s=(1, 1, 1), p='valid')
        x = self.conv(x, f=128, k=(self.n_bars - 1, 1, 1), s=(1, 1, 1), p='valid')

        x = self.conv(x, f=128, k=(1, 1, 12), s=(1, 1, 12), p='same', bn=True)
        x = self.conv(x, f=128, k=(1, 1, 7), s=(1, 1, 7), p='same', bn=True)
        x = self.conv(x, f=128, k=(1, 2, 1), s=(1, 2, 1), p='same', bn=True)
        x = self.conv(x, f=128, k=(1, 2, 1), s=(1, 2, 1), p='same', bn=True)
        x = self.conv(x, f=256, k=(1, 4, 1), s=(1, 2, 1), p='same', bn=True)
        x = self.conv(x, f=512, k=(1, 3, 1), s=(1, 2, 1), p='same', bn=True)

        x = Flatten()(x)

        x = Dense(1024, kernel_initializer=self.weight_init)(x)
        x = LeakyReLU()(x)

        critic_output = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)

        self.critic = Model(critic_input, critic_output)

    def conv_t(self, x, f, k, s, a, p, bn):
        x = Conv2DTranspose(filters=f, kernel_size=k, padding=p, strides=s, kernel_initializer=self.weight_init)(x)
        if bn:
            x = BatchNormalization(momentum=0.9)(x)
        return Activation(a)(x)

    def TemporalNetwork(self, name):

        input_layer = Input(shape=(self.z_dim,), name='temporal_input')

        x = Reshape([1, 1, self.z_dim])(input_layer)
        x = self.conv_t(x, f=1024, k=(2, 1), s=(1, 1), a='relu', p='valid', bn=True)
        x = self.conv_t(x, f=self.z_dim, k=(self.n_bars - 1, 1), s=(1, 1), a='relu', p='valid', bn=True)

        output_layer = Reshape([self.n_bars, self.z_dim])(x)

        return Model(input_layer, output_layer, name=name)

    def BarGenerator(self):

        input_layer = Input(shape=(self.z_dim * 4,), name='bar_generator_input')

        x = Dense(1024)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        x = Reshape([2, 1, 512])(x)
        x = self.conv_t(x, f=512, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=256, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=256, k=(2, 1), s=(2, 1), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=256, k=(1, 7), s=(1, 7), a='relu', p='same', bn=True)
        x = self.conv_t(x, f=1, k=(1, 12), s=(1, 12), a='tanh', p='same', bn=False)

        output_layer = Reshape([1, self.n_steps_per_bar, self.n_pitches, 1])(x)

        return Model(input_layer, output_layer)

    def _build_generator(self):

        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')

        # CHORDS -> TEMPORAL NETWORK
        self.chords_tempNetwork = self.TemporalNetwork("temporal_network")
        chords_over_time = self.chords_tempNetwork(chords_input)  # [n_bars, z_dim]

        # MELODY -> TEMPORAL NETWORK
        melody_over_time = [None] * self.n_tracks  # list of n_tracks [n_bars, z_dim] tensors
        self.melody_tempNetwork = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.melody_tempNetwork[track] = self.TemporalNetwork(f"temporal{track}")
            melody_track = Lambda(lambda x: x[:, track, :])(melody_input)
            melody_over_time[track] = self.melody_tempNetwork[track](melody_track)

        # CREATE BAR GENERATOR FOR EACH TRACK
        self.barGen = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.barGen[track] = self.BarGenerator()

        # CREATE OUTPUT FOR EVERY TRACK AND BAR
        bars_output = [None] * self.n_bars
        for bar in range(self.n_bars):
            track_output = [None] * self.n_tracks

            c = Lambda(lambda x: x[:, bar, :], name='chords_input_bar_' + str(bar))(chords_over_time)  # [z_dim]
            s = style_input  # [z_dim]

            for track in range(self.n_tracks):
                m = Lambda(lambda x: x[:, bar, :])(melody_over_time[track])  # [z_dim]
                g = Lambda(lambda x: x[:, track, :])(groove_input)  # [z_dim]

                z_input = Concatenate(axis=1, name=f'total_input_bar_{bar}_track_{track}')([c, s, m, g])

                track_output[track] = self.barGen[track](z_input)

            bars_output[bar] = Concatenate(axis=-1)(track_output)

        generator_output = Concatenate(axis=1, name='concat_bars')(bars_output)

        self.generator = Model([chords_input, style_input, melody_input, groove_input], generator_output)

    @staticmethod
    def get_opti(lr):
        # if self.optimiser == 'adam':
        #     opti = Adam(lr=lr, beta_1=0.5, beta_2=0.9)
        # elif self.optimiser == 'rmsprop':
        #     opti = RMSprop(lr=lr)
        # else:
        #     opti = Adam(lr=lr)
        return Adam(lr=lr, beta_1=0.5, beta_2=0.9)

    @staticmethod
    def set_trainable(m, val):
        m.trainable = val
        for layer in m.layers:
            layer.trainable = val

    def _build_adversarial(self):

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.set_trainable(self.generator, False)

        # Image input (real sample)
        real_img = Input(shape=self.input_dim)

        # Fake image
        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')

        fake_img = self.generator([chords_input, style_input, melody_input, groove_input])

        # critic determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        self.critic_model = Model(inputs=[real_img, chords_input, style_input, melody_input, groove_input],
                                  outputs=[valid, fake])

        self.critic_model.compile(
            loss=[self.binary_crossentropy, self.binary_crossentropy],
            optimizer=self.get_opti(self.critic_learning_rate))

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Sampled noise for input to generator
        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')

        # Generate images based of noise
        img = self.generator([chords_input, style_input, melody_input, groove_input])
        # Discriminator determines validity
        model_output = self.critic(img)
        # Defines generator model
        self.model = Model([chords_input, style_input, melody_input, groove_input], model_output)

        self.model.compile(optimizer=self.get_opti(self.generator_learning_rate), loss=self.binary_crossentropy)
        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = np.zeros((batch_size, 1), dtype=np.float32)

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]

        chords_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        style_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        melody_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))

        d_loss = self.critic_model.train_on_batch(
            [true_imgs, chords_noise, style_noise, melody_noise, groove_noise], [valid, fake])
        return d_loss

    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)

        chords_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        style_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        melody_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))

        return self.model.train_on_batch([chords_noise, style_noise, melody_noise, groove_noise], valid)

    def train(self, x_train, batch_size, epochs):
        all_steps = 1000
        for epoch in range(self.epoch, self.epoch + epochs):
            print(f'Epoch: {epoch + 1}/{epochs}')
            start_time = time.time()
            for step in range(all_steps):
                # critic_loops = n_critic
                # for _ in range(critic_loops):
                d_loss = sum(self.train_critic(x_train, batch_size))
                g_loss = self.train_generator(batch_size)
                self.progress_bar(step, all_steps, prefix='', length=30,
                                  suffix=f'\tgen: {g_loss:.2f}\tcritic: {d_loss:.2f}'
                                         f'\ttime: {time.time() - start_time:.1f} sec')

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

            self.progress_bar(all_steps, all_steps, prefix='', length=30,
                              suffix=f'\ttime: {time.time() - start_time:.1f} sec')
            self.save_model('../models/')
            self.epoch += 1

    def sample_images(self, filename, run_folder=''):
        r = 5

        chords_noise = np.random.normal(0, 1, (r, self.z_dim))
        style_noise = np.random.normal(0, 1, (r, self.z_dim))
        melody_noise = np.random.normal(0, 1, (r, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (r, self.n_tracks, self.z_dim))

        gen_scores = self.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

        # np.save(os.path.join(run_folder, "images/sample_%d.npy" % self.epoch), gen_scores)
        self.notes_to_midi(run_folder, gen_scores, filepath=f'../output/{filename}.mid')

    @staticmethod
    def binarise_output(output):
        return np.argmax(output, axis=3)  # output is a set of scores: [batch size , steps , pitches , tracks]

    def notes_to_midi(self, run_folder, output, filepath=None):
        for score_num in range(len(output)):
            max_pitches = self.binarise_output(output)

            midi_note_score = max_pitches[score_num].reshape([self.n_bars * self.n_steps_per_bar, self.n_tracks])
            parts = stream.Score()
            parts.append(tempo.MetronomeMark(number=66))

            for i in range(self.n_tracks):
                last_x = int(midi_note_score[:, i][0])
                s = stream.Part()
                dur = 0

                for idx, x in enumerate(midi_note_score[:, i]):
                    x = int(x)

                    if (x != last_x or idx % 4 == 0) and idx > 0:
                        n = note.Note(last_x)
                        n.duration = duration.Duration(dur)
                        s.append(n)
                        dur = 0

                    last_x = x
                    dur = dur + 0.25

                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)

                parts.append(s)

            parts.write('midi', fp=os.path.join(run_folder, filepath))

    def save(self, folder):
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.critic_learning_rate
                , self.generator_learning_rate
                , self.optimiser
                , self.z_dim
                , self.n_tracks
                , self.n_bars
                , self.n_steps_per_bar
                , self.n_pitches
            ], f)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'muse_model.h5'))
        # self.critic.save(os.path.join(run_folder, 'critic.h5'))
        # self.generator.save(os.path.join(run_folder, 'generator.h5'))
        # pickle.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

    def load_model(self, run_folder):
        self.model = load_model(os.path.join(run_folder, 'muse_model.h5'))
        # self.generator.load_weights(os.path.join(run_folder, 'weights', 'weights-g.h5'))
        # self.critic.load_weights(os.path.join(run_folder, 'weights', 'weights-c.h5'))

    @staticmethod
    def progress_bar(n_iter, n_total, prefix='Progress: ', suffix='', length=55, fill='█', lost='-'):
        percent = f"{100 * (n_iter / float(n_total)) :.1f}"
        filled_length = round(length * n_iter // n_total)
        bar = fill * filled_length + lost * (length - filled_length)
        print(f'\r{prefix}[{n_iter}/{n_total}] |{bar}| {percent}% {suffix}', end='')
        if n_iter == n_total:
            print()

    # def draw_bar(self, data, score_num, bar, part):
    #     plt.imshow(data[score_num, bar, :, :, part].transpose([1, 0]), origin='lower', cmap='Greys', vmin=-1, vmax=1)

    # def draw_score(self, data, score_num):
    #     fig, axes = plt.subplots(ncols=self.n_bars, nrows=self.n_tracks, figsize=(12, 8), sharey=True, sharex=True)
    #     fig.subplots_adjust(0, 0, 0.2, 1.5, 0, 0)
    #
    #     for bar in range(self.n_bars):
    #         for track in range(self.n_tracks):
    #
    #             if self.n_bars > 1:
    #                 axes[track, bar].imshow(data[score_num, bar, :, :, track].transpose([1, 0]), origin='lower',
    #                                         cmap='Greys', vmin=-1, vmax=1)
    #             else:
    #                 axes[track].imshow(data[score_num, bar, :, :, track].transpose([1, 0]), origin='lower',
    #                                    cmap='Greys', vmin=-1, vmax=1)

