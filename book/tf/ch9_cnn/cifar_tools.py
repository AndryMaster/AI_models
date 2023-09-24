import matplotlib.pyplot as plt
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        _data = pickle.load(fo, encoding='latin1')
    return _data


def show_img(data, label_name='', shape=(24, 24), filename=None):
    plt.figure()
    plt.title(label_name)
    img = np.reshape(data, shape)
    plt.imshow(img, cmap='Greys_r')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def clean(data):
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    grayscale_imgs = imgs.img_mean(1)  # Gray
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]  # Crop
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    # Mean
    img_size = img_data.shape[1]
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1 / np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds
    return normalized


def read_data(directory, test=False):
    names = unpickle(f'{directory}/batches.meta')['label_names']
    print(f'Names: {names}')
    data = labels = None
    if not test:
        for i in range(1, 6):
            filename = f"{directory}/data_batch_" + str(i)
            batch_data = unpickle(filename)
            if data is None:
                data, labels = batch_data['data'], batch_data['labels']
            else:
                data = np.vstack((data, batch_data['data']))
                labels = np.hstack((labels, batch_data['labels']))
    else:
        filename = f"{directory}/test_batch"
        batch_data = unpickle(filename)
        data, labels = np.array(batch_data['data']), np.array(batch_data['labels'])

    print(data.shape, labels.shape)
    data = clean(data)
    data = data.astype(np.float32)
    return names, data, labels
