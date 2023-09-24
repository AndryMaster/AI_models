import cifar_tools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# import os  # ERROR: -1073740791 (0xC0000409)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

names, data, labels = cifar_tools.read_data("../datasets/cifar/cifar-10-python")


def show_conv_results(_data, filename=None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(_data)[3]):
        img = _data[0, :, :, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def show_weights(w, filename=None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(w)[3]):
        img = w[:, :, 0, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


raw_data = data[4, :]
cifar_tools.show_img(np.reshape(raw_data, (24, 24)), names[labels[4]], filename='img/input_image.png')

x = tf.reshape(raw_data, shape=(-1, 24, 24, 1))
W = tf.Variable(tf.random_normal([5, 5, 1, 32]))
b = tf.Variable(tf.random_normal([32]))

conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
conv_with_b = tf.nn.bias_add(conv, b)
conv_out = tf.nn.relu(conv_with_b)

k = 2
max_pool = tf.nn.max_pool(conv_out, ksize=k, strides=k, padding='SAME')  # k or [1, k, k, 1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W_val = sess.run(W)
    show_weights(W_val, 'img/step0_weights.png')  # img/step0_weights.png

    conv_val = sess.run(conv)
    show_conv_results(conv_val, 'img/step1_conv.png')  # img/step1_conv.png
    print(np.shape(conv_val))

    conv_out_val = sess.run(conv_out)
    show_conv_results(conv_out_val, 'img/step2_conv_outs.png')  # img/step2_conv_outs.png
    print(np.shape(conv_out_val))

    max_pool_val = sess.run(max_pool)
    show_conv_results(max_pool_val, 'img/step3_max_pool.png')  # img/step2_conv_outs.png
    print(np.shape(max_pool_val))
