import cifar_tools
import random
import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def progress_bar(iteration, total, prefix='Progress:', suffix='', length=50, fill='█', lost='-'):
    percent = f"{100 * (iteration / float(total)) :.1f}"
    filled_length = round(length * iteration // total)
    bar = fill * filled_length + lost * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def show_pred_and_img(data, label, prediction, iters=5, rows=3, cols=4):
    plt.figure()
    for _ in range(iters):
        for i in range(rows * cols):
            n = random.randint(1, len(label))
            img = data[n]  # .reshape(*shape)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            color = 'b' if label[n] == prediction[n] else 'r'
            plt.title(f"{names[label[n]]}>{names[int(prediction[n])]} {label[n]}-{int(prediction[n])}", color=color)
            plt.axis('off')
        plt.show()


class CnnCifar10:
    def __init__(self, epochs=100, learning_rate=0.001, batch_size=100, num_outputs=100):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.init_weights()
        self.model = self.model_op()
        self.prob = tf.nn.softmax(self.model)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver()
        self.path = './models/cifar/model_cif100_v1.ckpt'

    def init_weights(self):
        # Input
        self.imgs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_images')
        self.y = tf.placeholder(tf.float32, shape=(None, self.num_outputs), name='output')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        self.img_mean = tf.constant([123.68 / 256, 116.779 / 256, 103.939 / 256], dtype=tf.float32, shape=[1, 1, 1, 3])

        # Convert layer 1_1
        self.W1_1 = tf.Variable(tf.truncated_normal([4, 4, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights1_1')
        self.b1_1 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases1_1')
        # Convert layer 1_2
        self.W1_2 = tf.Variable(tf.truncated_normal([4, 4, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights1_2')
        self.b1_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases1_2')

        # Convert layer 2_1
        self.W2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights2_1')
        self.b2_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases2_1')
        # Convert layer 2_2
        self.W2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights2_2')
        self.b2_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases2_2')

        # Hidden layer 1 (512)
        self.hW1 = tf.Variable(tf.truncated_normal([8*8*128, 1024], dtype=tf.float32, stddev=1e-1), name='hid_weights1')
        self.hb1 = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='hid_biases1')
        # Hidden layer 2  (128)
        self.hW2 = tf.Variable(tf.truncated_normal([1024, 256], dtype=tf.float32, stddev=1e-1), name='hid_weights2')
        self.hb2 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='hid_biases2')
        # Output layer
        self.W_out = tf.Variable(tf.truncated_normal([256, self.num_outputs], dtype=tf.float32, stddev=1e-1),
                                 name='weights_out')
        self.b_out = tf.Variable(tf.constant(0.0, shape=[self.num_outputs], dtype=tf.float32), trainable=True,
                                 name='biases_out')

    def load(self, sess):
        self.saver.restore(sess, self.path)
        print(f"Model loaded")

    @staticmethod
    def conv_layer(x, W, b):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_with_b = tf.nn.bias_add(conv, b)
        conv_out = tf.nn.relu(conv_with_b)
        return conv_out

    @staticmethod
    def maxpool_layer(conv, k=2):
        max_pool = tf.nn.max_pool(conv, ksize=k, strides=k, padding='SAME')  # k or [1, k, k, 1]
        return max_pool

    def model_op(self):
        # Input
        images = self.imgs - self.img_mean

        conv_1_1 = self.conv_layer(images, self.W1_1, self.b1_1)  # Convert layer 1_1
        conv_1_2 = self.conv_layer(conv_1_1, self.W1_2, self.b1_2)  # Convert layer 1_2
        max_pool_1 = self.maxpool_layer(conv_1_2, k=2)  # Pooling 1  32x32 -> 16x16

        conv_2_1 = self.conv_layer(max_pool_1, self.W2_1, self.b2_1)  # Convert layer 2_1
        conv_2_2 = self.conv_layer(conv_2_1, self.W2_2, self.b2_2)  # Convert layer 2_2
        max_pool_2 = self.maxpool_layer(conv_2_2, k=2)  # Pooling 2  16x16 -> 8x8

        # Hidden layer 1
        length = int(np.prod(max_pool_2.get_shape()[1:]))
        max_pool_2_flat = tf.reshape(max_pool_2, [-1, length])
        hidden_1 = tf.nn.sigmoid(tf.matmul(max_pool_2_flat, self.hW1) + self.hb1)
        hidden_1 = tf.nn.dropout(hidden_1, rate=self.dropout_rate)
        # Hidden layer 2
        hidden_2 = tf.nn.sigmoid(tf.matmul(hidden_1, self.hW2) + self.hb2)
        hidden_2 = tf.nn.dropout(hidden_2, rate=self.dropout_rate)

        # Output layer
        out = tf.matmul(hidden_2, self.W_out) + self.b_out
        return out

    def test(self, data, labels, sess):
        one_hot_labels = sess.run(tf.one_hot(labels, self.num_outputs, on_value=1., off_value=0., axis=-1))
        prediction, accuracy, loss = np.zeros(len(labels)), [], 0
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batch_labels = one_hot_labels[i:i + self.batch_size]
            pred, acc, lo = sess.run([tf.cast(tf.argmax(self.prob, axis=1), tf.int8), self.accuracy, self.cost],
                                     feed_dict={self.imgs: batch_data, self.y: batch_labels, self.dropout_rate: 0.})
            loss += lo
            accuracy.append(acc)
            prediction[i:i + self.batch_size] = pred
        return prediction, sum(accuracy) / len(accuracy), loss

    def train(self, data, labels, test_data, test_labels):
        start = time()
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            one_hot_labels = sess.run(tf.one_hot(labels, self.num_outputs, on_value=1., off_value=0., axis=-1))
            max_accuracy_test = 0
            for epoch in range(1, self.epochs+1):
                s = time()
                iters = len(data) // self.batch_size
                accuracy_total = loss_total = 0
                for i in range(0, len(data), self.batch_size):
                    batch_data = data[i:i + self.batch_size, :]
                    batch_labels = one_hot_labels[i:i + self.batch_size]  # :
                    _, accuracy_val, loss = sess.run([self.train_op, self.accuracy, self.cost],
                                                     feed_dict={self.imgs: batch_data, self.y: batch_labels,
                                                                self.dropout_rate: 0.2})

                    accuracy_total += accuracy_val
                    loss_total += loss
                    progress_bar(i, len(data), prefix=f"EPOCH-{epoch}:\t",
                                 suffix=f"[{i}-{len(data)}]\tAccuracy: {accuracy_val * 100 :.1f}%  Loss: {loss:.2f}")

                    if i + self.batch_size == len(data):  # end for
                        progress_bar(1, 1, prefix=f"EPOCH-{epoch}:\t",
                                     suffix=f"[{len(data)}-{len(data)}]\tTime: {time() - s :.2f} sec")
                        _, accuracy_test, loss_test = self.test(test_data, test_labels, sess)
                        print(f"Accuracy: {accuracy_total / iters * 100 :.2f}%   Accuracy_test: "
                              f"{accuracy_test * 100 :.2f}%   Loss: {loss_total / 5:.3f}   Loss_test: {loss_test:.3f}")
                        if accuracy_test > max_accuracy_test:
                            max_accuracy_test = accuracy_test
                            self.saver.save(sess, self.path)
        print(f"Total time: {time() - start :.0f} sec")


if __name__ == "__main__":
    names, imgs_train, labels_train, imgs_test, labels_test = \
        cifar_tools.read_data100("../datasets/cifar/cifar-100-python")
    # len(names) = 100; [...]

    cnn = CnnCifar10(epochs=200,
                     learning_rate=0.0003,
                     batch_size=250,
                     num_outputs=len(names))

    # cnn.train(imgs_train, labels_train, imgs_test, labels_test)

    with tf.Session() as session:
        cnn.load(session)
        train_predict, train_accuracy_, _ = cnn.test(imgs_train, labels_train, session)
        predict, accuracy_, _ = cnn.test(imgs_test, labels_test, session)

    print(f"Accuracy train: {train_accuracy_ * 100 :.2f}%\nAccuracy: {accuracy_ * 100 :.2f}%")
    show_pred_and_img(imgs_test, labels_test, predict)
    show_pred_and_img(imgs_test, labels_test, predict, iters=10, rows=1, cols=1)
    show_pred_and_img(imgs_train, labels_train, train_predict)
