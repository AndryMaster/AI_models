import matplotlib.pyplot as plt
import numpy as np
import pickle, gzip, sys, random


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


def show_img(data, label, shape=(24, 24), rows=5, cols=8):
    plt.figure()
    for i in range(rows * cols):
        n = random.randint(1, len(label))
        img = data[n].reshape(*shape)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.title(label[n], color='b')
        plt.axis('off')
    plt.show()


def clean(data):
    # imgs = data.reshape(data.shape[0], 3, 28, 28)
    cropped_imgs = data[:, 2:26, 2:26]  # Crop
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    # Mean
    img_size = img_data.shape[1]
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1 / np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds

    out = normalized.astype(np.float32)
    return out


def read_data(filename):
    with gzip.open(filename, 'rb') as f:
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='bytes')

    data = np.array(data, dtype=object)
    (x_train, y_train), (x_test, y_test) = data
    x_train, x_test = clean(x_train), clean(x_test)
    # x_train, y_train, x_test, y_test = np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.int8),\
    #                                    np.array(x_test, dtype=np.float32), np.array(y_test, dtype=np.int8)
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = read_data("../datasets/mnist/mnist.pkl.gz")
