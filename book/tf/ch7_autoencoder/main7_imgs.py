from autoencoder import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        _data = pickle.load(fo, encoding='latin1')
    return _data


def grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).img_mean(1).reshape(a.shape[0], -1)


names = unpickle("../datasets/cifar/cifar-10-python/batches.meta")['label_names']
data = labels = None
for i in range(1, 6):
    filename = "../datasets/cifar/cifar-10-python/data_batch_" + str(i)
    batch_data = unpickle(filename)
    if data is None:
        data, labels = batch_data['data'], batch_data['labels']
    else:
        data = np.vstack((data, batch_data['data']))
        labels = np.vstack((labels, batch_data['labels']))


labels = labels.reshape((50000,))
print(data.shape, labels.shape)
data = grayscale(data)
print(data.shape)

x = np.matrix(data)
y = np.array(labels)

horse_x = np.array([x[i, :] for i in range(len(x)) if labels[i] == 2])  # 7
horse_x = horse_x.reshape((horse_x.shape[::2]))
print(horse_x.shape)

input_dim = horse_x.shape[1]
hidden_dim = 100

ae = AutoEncoder(input_dim, hidden_dim, epoch=300, batch_size=10)
ae.train(horse_x)

for h in horse_x:
    res = ae.test(h)

    real, test = h.reshape((32, 32)), res.reshape((32, 32))
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(real)  # cmap='gray' Greys_r
    axes[0].set_title('Real img 32x32')
    axes[1].imshow(test)
    axes[1].set_title('Test img x10 small')
    plt.show()
