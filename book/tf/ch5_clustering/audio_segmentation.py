import scipy
import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

k = 2
segment_size = 60
max_iterations = 200

chroma = tf.placeholder(tf.float32)
max_freqs = tf.argmax(chroma, 0)


def extract_feature_vector(chroma_data):
    num_features, num_samples = np.shape(chroma_data)  # notes, length
    freq_vals = sess.run(max_freqs, feed_dict={chroma: chroma_data})
    hist, bins = np.histogram(freq_vals, bins=range(num_features + 1))
    return hist.astype(float) / num_samples


def get_chromatogram(filename):
    y, sr = librosa.load(filename)  # wave, clock[22050Hz]

    # chroma_(cqt, stft, cens)
    chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)  # 1.6 sec

    chroma_orig_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_orig_cens = librosa.feature.chroma_cens(y=y, sr=sr)

    y_harm = librosa.effects.harmonic(y=y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)

    chroma_filter = np.minimum(chroma_harm, librosa.decompose.nn_filter(chroma_harm, aggregate=np.median, metric='cosine'))
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))  # 5 sec

    # # Show on display!
    # fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True)
    # idx = tuple([slice(None), slice(*list(librosa.time_to_frames([0, 31])))])
    # librosa.display.specshow(chroma_orig[idx], y_axis='chroma', x_axis='time', ax=ax[0])
    # ax[0].set(ylabel='Default chroma')
    # ax[0].label_outer()
    # librosa.display.specshow(chroma_orig_stft[idx], y_axis='chroma', x_axis='time', ax=ax[1])
    # ax[1].set(ylabel='Stft chroma')
    # ax[1].label_outer()
    # librosa.display.specshow(chroma_orig_cens[idx], y_axis='chroma', x_axis='time', ax=ax[2])
    # ax[2].set(ylabel='Cens chroma')
    # ax[2].label_outer()
    # librosa.display.specshow(chroma_harm[idx], y_axis='chroma', x_axis='time', ax=ax[3])
    # ax[3].set(ylabel='Harmonic')
    # ax[3].label_outer()
    # librosa.display.specshow(chroma_smooth[idx], y_axis='chroma', x_axis='time', ax=ax[4])
    # ax[4].set(ylabel='Median-filtered')
    # ax[4].label_outer()

    chromatogram = chroma_smooth
    return chromatogram


def get_dataset(audio_file):
    chroma_data = get_chromatogram(audio_file)
    chroma_length = chroma_data.shape[1]
    print(f"Chroma_shape <{audio_file}> ({chroma_length}): {chroma_data.shape}")
    xs = None
    for j in range(chroma_length // segment_size):
        chroma_segment = chroma_data[:, j*segment_size:(j+1)*segment_size]
        x = extract_feature_vector(chroma_segment)
        xs = x if xs is None else np.vstack((xs, x))
    print(f"Audio dataset {xs.shape}: {xs}")
    return xs


def initial_cluster_centroids(X, k):
    return X[0:k, :]


def assign_cluster(X, centroids, iter):
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), axis=2)
    minis = tf.argmin(distances, 0)
    if iter % 20 == 0:
        print(f"Iter {i}")
    if iter == max_iterations:
        global dist
        dist = distances.eval()
    return minis


def recompute_centroids(X, Y):
    sums = tf.unsorted_segment_sum(X, Y, num_segments=k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, num_segments=k)
    return sums / counts


with tf.Session() as sess:
    X = get_dataset('TalkingMachinesPodcast.wav')
    centroids = initial_cluster_centroids(X, k)

    for i in range(1, max_iterations+1):
        Y = assign_cluster(X, centroids, i)
        centroids = sess.run(recompute_centroids(X, Y))

    time = []
    segments = sess.run(Y)
    for i in range(len(segments)):
        seconds = i * segment_size / 41.95
        print(f"{int(seconds // 60)}min {round(seconds % 60, 1)}sec :"
              f"\t{'Bob   1' if segments[i] else 'Alisa 2'}\t{100 - round(dist[segments[i], i] * 100)}%")
        time.append(seconds)

    print(f'Centroids: {centroids}')

    plt.scatter(time, segments)
    plt.plot(time, segments)

    plt.show()
