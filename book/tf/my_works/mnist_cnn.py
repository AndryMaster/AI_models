import mnist_tools
import random
import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class CnnMnist:
    def __init__(self, epochs=100, learning_rate=0.001, batch_size=500):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_outputs = 10  # numbers 0..9
        self.batch_size = batch_size
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # Input
        self.x = tf.placeholder(tf.float32, shape=(None, 24 * 24))
        self.y = tf.placeholder(tf.float32, shape=(None, self.num_outputs))
        # Convert layer 1
        self.W1 = tf.Variable(tf.random_normal([6, 6, 1, 75]))
        self.b1 = tf.Variable(tf.random_normal([75]))
        # Convert layer 2
        self.W2 = tf.Variable(tf.random_normal([5, 5, 75, 60]))
        self.b2 = tf.Variable(tf.random_normal([60]))
        # Hidden layer
        self.W3 = tf.Variable(tf.random_normal([6*6*60, 1536]))  # big_x1024
        self.b3 = tf.Variable(tf.random_normal([1536]))
        # Output layer
        self.W_out = tf.Variable(tf.random_normal([1536, self.num_outputs]))  # 1024x10
        self.b_out = tf.Variable(tf.random_normal([self.num_outputs]))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model_op(), labels=self.y))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.correct_prediction = tf.equal(tf.argmax(self.model_op(), 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver()
        self.path = 'models/mnist/model1.ckpt'
        # model 97.7 - common (5x5x64 5x5x64 x1024); model1 98.5 - long_very good[+++] (6x6x72 5x5x60 x1536);
        # model2 97.7 - common[0]; model_lite 98 - very fast[+] (5x5x64 4x4x50 x512)

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
        x_reshaped = tf.reshape(self.x, shape=(-1, 24, 24, 1))

        conv_out1 = self.conv_layer(x_reshaped, self.W1, self.b1)  # NxNx64
        maxpool_out1 = self.maxpool_layer(conv_out1)  # 24x24 -> 12x12
        norm1 = tf.nn.lrn(maxpool_out1, depth_radius=4, bias=1., alpha=0.001 / 9, beta=0.75)

        conv_out2 = self.conv_layer(norm1, self.W2, self.b2)  # NxNx64
        norm2 = tf.nn.lrn(conv_out2, depth_radius=4, bias=1., alpha=0.001 / 9, beta=0.75)
        maxpool_out2 = self.maxpool_layer(norm2)  # 12x12 -> 6x6

        maxpool_reshaped = tf.reshape(maxpool_out2, shape=(-1, self.W3.get_shape().as_list()[0]))
        local = tf.matmul(maxpool_reshaped, self.W3) + self.b3  # Hidden layer 1024
        local_out = tf.nn.relu(local)

        out = tf.matmul(local_out, self.W_out) + self.b_out  # Out number 0..9 (softmax)
        return out

    def test(self, data, labels, sess):
        one_hot_labels = sess.run(tf.one_hot(labels, self.num_outputs, on_value=1., off_value=0., axis=-1))
        prediction = np.zeros(len(labels))
        accuracy = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size, :]
            batch_labels = one_hot_labels[i:i + self.batch_size]
            model, acc = sess.run([tf.cast(tf.argmax(self.model_op(), axis=1), tf.int8), self.accuracy],
                                  feed_dict={self.x: batch_data, self.y: batch_labels})
            accuracy.append(acc)
            prediction[i:i + self.batch_size] = model
        return prediction, sum(accuracy) / len(accuracy)

    def train(self, data, labels, test_data, test_labels):
        start = time()
        with tf.Session() as sess:
            # summary_writer = tf.summary.FileWriter('../logs', sess.graph)
            sess.run(tf.global_variables_initializer())
            one_hot_labels = sess.run(tf.one_hot(labels, self.num_outputs, on_value=1., off_value=0., axis=-1))
            for epoch in range(1, self.epochs+1):
                s = time()
                iters = len(data) // self.batch_size
                accuracy_total = loss_total = 0
                for i in range(0, len(data), self.batch_size):
                    batch_data = data[i:i + self.batch_size, :]
                    batch_labels = one_hot_labels[i:i + self.batch_size]  # :
                    _, accuracy_val, cost = sess.run([self.train_op, self.accuracy, self.cost],
                                                     feed_dict={self.x: batch_data, self.y: batch_labels})
                    # summary_writer.add_summary(summary, i)
                    accuracy_total += accuracy_val
                    loss_total += cost

                    progress_bar(i, len(data), prefix=f"EPOCH-{epoch}:\t",
                                 suffix=f"[{i}-{len(data)}]\tAccuracy: {accuracy_val * 100 :.2f}%  Loss: {cost:.2f}")

                    if i + self.batch_size == len(data):  # end
                        progress_bar(1, 1, prefix=f"EPOCH-{epoch}:\t",
                                     suffix=f"[{len(data)}-{len(data)}]\tTime: {time() - s :.2f} sec")
                        _, accuracy_test = self.test(test_data, test_labels, sess)
                        print(f"Accuracy: {accuracy_total / iters * 100 :.2f}%\tAccuracy_test:"
                              f" {accuracy_test * 100 :.2f}%\tLoss: {loss_total / iters:.3f}\tModel saved")

                # self.saver.save(sess, self.path)
        # summary_writer.close()
        print(f"Total time: {time() - start :.0f} sec")


def progress_bar(iteration, total, prefix='Progress:', suffix='', length=50, fill='█', lost='-'):
    percent = f"{100 * (iteration / float(total)) :.1f}"
    filled_length = round(length * iteration // total)
    bar = fill * filled_length + lost * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def show_all_imgs(data, label, prediction, shape=(24, 24), rows=5, cols=8):
    plt.figure()
    while True:
        for i in range(rows * cols):
            n = random.randint(1, len(label))
            img = data[n].reshape(*shape)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray', interpolation='none')
            color = 'b' if label[n] == prediction[n] else 'r'
            plt.title(f"{label[n]}_{prediction[n] :.0f}", color=color)
            plt.axis('off')
        plt.show()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = mnist_tools.read_data("../datasets/mnist/mnist.pkl.gz")
    cnn = CnnMnist(epochs=80, batch_size=500)

    # cnn.train(x_train, y_train, x_test, y_test)  !!!

    with tf.Session() as session:
        # print(1e-1, 1e-1 + 5, float(1e-1), round(1e-1))
        # print(session.run(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=0.1)))
        cnn.load(session)
        predict, acu = cnn.test(x_test, y_test, session)
    print(f"Accuracy {acu * 100 :.2f}%")
    show_all_imgs(x_test, y_test, prediction=predict)
