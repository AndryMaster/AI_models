import scipy
import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np
from time import time
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

start = time()
last_dist = None

filenames = tf.train.match_filenames_once('./audio_dataset/*.wav')  # ./*.wav - one big file
count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
filename, file_content = reader.read(filename_queue)

chroma = tf.placeholder(tf.float32)
max_freqs = tf.argmax(chroma, 0)

k = 2
max_iterations = 200


def initial_cluster_centroids(X, k):
    return X[0:k, :]


def assign_cluster(X, centroids, iter):
    global converged, last_dist
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), axis=2)
    minis = tf.argmin(distances, 0)
    if iter % 20 == 0:
        print(f'Dist{i}:\t{distances.eval()}\t{minis.eval()}')
    return minis


def recompute_centroids(X, Y):
    sums = tf.unsorted_segment_sum(X, Y, num_segments=k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, num_segments=k)
    return sums / counts


def get_next_chromatogram():
    audio_file = str(sess.run(filename))[2:-1]
    y, sr = librosa.load(audio_file)  # chroma_(cqt, stft, cens)

    chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)  # 1.6 sec

    y_harm = librosa.effects.harmonic(y=y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)

    chroma_filter = np.minimum(chroma_harm, librosa.decompose. nn_filter(
        chroma_harm, aggregate=np.median, metric='cosine'))

    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))  # 3.8 sec

    # # Show on display!
    # fig, ax = plt.subplots(nrows=4, sharex=True, sharey=True)
    # idx = tuple([slice(None), slice(*list(librosa.time_to_frames([0, 15])))])
    # librosa.display.specshow(chroma_orig[idx], y_axis='chroma', x_axis='time', ax=ax[0])
    # ax[0].set(ylabel='Default chroma')
    # ax[0].label_outer()
    # librosa.display.specshow(chroma_harm[idx], y_axis='chroma', x_axis='time', ax=ax[1])
    # ax[1].set(ylabel='Harmonic')
    # ax[1].label_outer()
    # librosa.display.specshow(chroma_filter[idx], y_axis='chroma', x_axis='time', ax=ax[2])
    # ax[2].set(ylabel='Non-local')
    # ax[2].label_outer()
    # librosa.display.specshow(chroma_smooth[idx], y_axis='chroma', x_axis='time', ax=ax[3])
    # ax[3].set(ylabel='Median-filtered')
    # ax[3].label_outer()
    # plt.show()

    chromatogram = chroma_orig
    print(audio_file, chromatogram.shape)
    return chromatogram, audio_file.split('\\')[-1]


def extract_feature_vector(chroma_data):
    num_features, num_samples = np.shape(chroma_data)  # notes, length
    freq_vals = sess.run(max_freqs, feed_dict={chroma: chroma_data})
    hist, bins = np.histogram(freq_vals, bins=range(num_features + 1))
    return hist.astype(float) / num_samples


def get_dataset():
    num_files = sess.run(count_num_files)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    xs, name = None, []
    for _ in range(num_files):
        chroma_data, fname = get_next_chromatogram()
        x = np.matrix([extract_feature_vector(chroma_data)])
        xs = x if xs is None else np.vstack((xs, x))
        name.append(fname)
    return xs, name


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # for sound need (global and local) initializer: data - local; variables - global
    print(f'Run init: {time() - start :.2f}')
    X, names = get_dataset()
    print(f'Dataset ready: {X}\nTime: {time() - start :.2f}')
    centroids = initial_cluster_centroids(X, k)
    i, converged = 0, False
    while not converged and i < max_iterations:
        i += 1
        Y = assign_cluster(X, centroids, i)
        centroids = sess.run(recompute_centroids(X, Y))
    print(f'Names: {names}\nCentroids: {centroids}\nTime: {time() - start :.2f}')
