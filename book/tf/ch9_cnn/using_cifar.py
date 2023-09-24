import random
import numpy as np
import matplotlib.pyplot as plt
import cifar_tools

_names, _data, _labels = cifar_tools.read_data("../datasets/cifar/cifar-10-python")
n = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def show_some_examples(names, data, labels):
    plt.figure()
    rows, cols = 4, 5
    random_idxs = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        idx = random_idxs[i]
        plt.title(names[labels[idx]])
        img = np.reshape(data[idx, :], (24, 24))
        plt.imshow(img, cmap='Greys_r')  # Greys_r
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('img/cifar_examples.png')


show_some_examples(_names, _data, _labels)
