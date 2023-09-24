import os
import numpy as np
from ModelMuseGAN import MuseGAN


def load_music(n_bars_, n_steps_per_bar_, filename='Jsb16Separated.npz'):
    file = os.path.abspath(os.path.join("../dataset/JSB-Chorales", filename))
    with np.load(file, encoding='bytes', allow_pickle=True) as f:
        data = f['train']
    # with open('jsb-chorales-16th.pkl', 'rb') as p:
    #     data = pickle.load(p, encoding="latin1")

    data_ints_ = []
    for x in data:
        counter = 0
        cont = True
        while cont:
            if not np.any(np.isnan(x[counter:(counter + 4)])):
                cont = False
            else:
                counter += 4

        if n_bars_ * n_steps_per_bar_ < x.shape[0]:
            data_ints_.append(x[counter:(counter + (n_bars_ * n_steps_per_bar_)), :])

    data_ints_ = np.array(data_ints_)
    n_songs = data_ints_.shape[0]
    n_tracks_ = data_ints_.shape[2]
    data_ints_ = data_ints_.reshape([n_songs, n_bars_, n_steps_per_bar_, n_tracks_])

    max_note = 83

    where_are_NaNs = np.isnan(data_ints_)
    data_ints_[where_are_NaNs] = max_note + 1
    max_note = max_note + 1

    data_ints_ = data_ints_.astype(int)
    num_classes = max_note + 1

    data_binary_ = np.eye(num_classes)[data_ints_]
    data_binary_[data_binary_ == 0] = -1
    data_binary_ = np.delete(data_binary_, max_note, -1)
    data_binary_ = data_binary_.transpose([0, 1, 2, 4, 3])

    return data_binary_, data_ints_, data


BATCH_SIZE = 64
n_bars = 2            # 32
n_steps_per_bar = 16  # 256
n_pitches = 84        # max_notes 83+Nan
n_tracks = 4          # roads num

data_binary, data_ints, raw_data = load_music(n_bars, n_steps_per_bar)
data_binary = np.squeeze(data_binary)

print(data_ints.shape, data_binary.shape, raw_data.shape, sep='\n')
# 229, n_bars, n_steps_per_bar, b(n_pitches), n_tracks


gan = MuseGAN(
    input_dim=data_binary.shape[1:],
    critic_learning_rate=0.0008,
    generator_learning_rate=0.0008,
    optimiser='adam',
    z_dim=32,
    n_tracks=n_tracks,
    n_bars=n_bars,
    n_steps_per_bar=n_steps_per_bar,
    n_pitches=n_pitches)

# gan.train(data_binary, batch_size=BATCH_SIZE, epochs=10)

gan.load_model('../models/')
gan.sample_images('muse_1')
