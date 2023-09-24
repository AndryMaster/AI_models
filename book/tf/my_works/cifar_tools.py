import matplotlib.pyplot as plt
import numpy as np
import pickle, random


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


def show_img(data, label, shape=(32, 32, 3), rows=2, cols=3):
    plt.figure()
    for i in range(rows * cols):
        n = random.randint(0, len(label))
        img = data[n]  # .reshape(*shape)
        # img = np.moveaxis(img, 0, -1)  # For colored
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f'{label[n]} {names_[label[n]]}', color='b')
        plt.axis('off')
    plt.show()


def read_data100(directory, **cleans):
    names = unpickle(f'{directory}/meta')['fine_label_names']
    print(f'Names: {names}')

    load_train_data = unpickle(f"{directory}/train")
    train_data, train_labels = np.array(load_train_data['data']), np.array(load_train_data['fine_labels'])

    load_test_data = unpickle(f"{directory}/test")
    test_data, test_labels = np.array(load_test_data['data']), np.array(load_test_data['fine_labels'])

    train_data, test_data = clean(train_data, **cleans), clean(test_data, **cleans)
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

    return names, train_data, train_labels, test_data, test_labels


def clean(data, gray=False, crop=0, mean=False):
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    imgs = np.moveaxis(imgs, 1, -1)  # 32, 32, 3
    if gray:  # grayscale_imgs
        imgs = imgs.mean(3)
    if crop:  # cropped_imgs
        imgs = imgs[:, crop:32-crop, crop:32-crop]
    if mean:  # Mean
        shape, img_data = imgs.shape, imgs.reshape(len(data), -1)
        img_size = img_data.shape[1]
        means = np.mean(img_data, axis=1)
        means = means.reshape(len(means), 1)
        stds = np.std(img_data, axis=1)
        stds = stds.reshape(len(stds), 1)
        adj_stds = np.maximum(stds, 1 / np.sqrt(img_size))
        normalized = (img_data - means) / adj_stds  # normalized
        return normalized.reshape(shape)
    return imgs / 256


def read_data(directory, **cleans):
    names = unpickle(f'{directory}/batches.meta')['label_names']
    print(f'Names: {names}')
    data = labels = None
    for i in range(1, 6):
        filename = f"{directory}/data_batch_" + str(i)
        batch_data = unpickle(filename)
        if data is None:
            data, labels = batch_data['data'], batch_data['labels']
        else:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels']))

    filename_test = f"{directory}/test_batch"
    batch_data = unpickle(filename_test)
    data_test, labels_test = np.array(batch_data['data']), np.array(batch_data['labels'])

    print(data.shape, labels.shape, data_test.shape, labels_test.shape)
    data, data_test = clean(data, **cleans), clean(data_test, **cleans)

    return names, data, labels, data_test, labels_test


if __name__ == "__main__":
    # X_train, y_train, X_test, y_test = read_data("../datasets/mnist/mnist.pkl.gz")
    names_, x_train, y_train, x_test, y_test = read_data100("../datasets/cifar/cifar-100-python")
    # print(*x_train[6])
    show_img(x_train, y_train)
    show_img(x_train, y_train)
    show_img(x_train, y_train)
