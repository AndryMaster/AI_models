import os, glob
import numpy as np
from time import time
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

DATASET_DIR = "../datasets/cats_vs_dogs/"


def progress_bar(iteration, total, prefix='Progress:', suffix='', length=50, fill='█', lost='-'):
    percent = f"{100 * iteration / float(total) :.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + lost * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def show_imgs(imgs, label, place=(3, 6)):
    plt.figure()
    rows, cols = place
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i])
        plt.title(label[i])  # color='b'
        plt.axis('off')
    plt.show()


def get_image(path, resize=(int, int)):
    img = np.array(ImageOps.fit(Image.open(path), resize, Image.ANTIALIAS)) / 256  # Image.ANTIALIAS = quality(&fast)
    # show_imgs([img], '', place=(1, 1))
    return img  # (128, 128, 3)


def get_dataset(path_animal, shape=(96, 96), train_rate=0.8, max_imgs=0):  # "128" 10_GB, 100, '96' 5_GB, 64 1.7_GB, 50
    img_files = sorted(glob.glob(os.path.join(DATASET_DIR, path_animal, '*.jpg')),
                       key=lambda path: int(path.split('\\')[-1].split('.')[0]))
    res = np.zeros((len(img_files), *shape, 3))
    s = time()
    for i, filename in enumerate(img_files):
        res[i] = get_image(filename, resize=shape)
        if max_imgs and i == max_imgs-1:
            res = res[:max_imgs]
            break
        if i % 100 == 0:
            progress_bar(i, max_imgs if max_imgs else len(img_files), prefix=f'Loading {path_animal}:',
                         suffix=f'[{i}/{max_imgs if max_imgs else len(img_files)}] time {time() - s :.1f} sec')
    progress_bar(1, 1, prefix=f'Dataset {path_animal}:', suffix=f'[{max_imgs if max_imgs else len(img_files)}]'
                                                                f' time {time() - s :.1f} sec')

    train_idx_end = round(len(res) * train_rate)
    train_res, test_res = res[:train_idx_end], res[train_idx_end:]
    return train_res, test_res


class Cnn_CvsD:
    def __init__(self, epochs=100, learning_rate=0.001, batch_size=10):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_outputs = 2  # cat or dog
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
        self.path = './models/CvsD_model1.ckpt'

    def init_weights(self):
        # Input
        self.imgs = tf.placeholder(tf.float32, shape=(None, 96, 96, 3), name='input_images')
        self.y = tf.placeholder(tf.float32, shape=(None, self.num_outputs), name='output')
        self.img_mean = tf.constant([123.68 / 256, 116.779 / 256, 103.939 / 256], dtype=tf.float32,
                                    shape=[1, 1, 1, 3], name='img_mean')
        # Convert layer 1_1
        self.W1_1 = tf.Variable(tf.truncated_normal([6, 6, 3, 32], dtype=tf.float32, stddev=0.1), name='weights1_1')
        self.b1_1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases1_1')
        # Convert layer 1_2
        self.W1_2 = tf.Variable(tf.truncated_normal([6, 6, 32, 32], dtype=tf.float32, stddev=0.1), name='weights1_2')
        self.b1_2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases1_2')
        # Pooling 1
        # Convert layer 2_1
        self.W2_1 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], dtype=tf.float32, stddev=0.1), name='weights2_1')
        self.b2_1 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases2_1')
        # Convert layer 2_2
        self.W2_2 = tf.Variable(tf.truncated_normal([4, 4, 64, 64], dtype=tf.float32, stddev=0.1), name='weights2_2')
        self.b2_2 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases2_2')
        # Pooling 2
        # Convert layer 3_1
        self.W3_1 = tf.Variable(tf.truncated_normal([4, 4, 64, 128], dtype=tf.float32, stddev=0.1), name='weights3_1')
        self.b3_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases3_1')
        # Convert layer 3_2
        self.W3_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=0.1), name='weights3_2')
        self.b3_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases3_2')
        # # Convert layer 3_3
        # self.W3_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=0.1), name='weights3_3'
        # self.b3_3 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases3_3')
        # Pooling 3
        # Convert layer 4_1
        self.W4_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=0.1), name='weights4_1')
        self.b4_1 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases4_1')
        # Convert layer 4_2
        self.W4_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=0.1), name='weights4_2')
        self.b4_2 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases4_2')
        # # Convert layer 4_3
        # self.W4_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 128], dtype=tf.float32, stddev=0.1), name='weights4_3'
        # self.b4_3 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases4_3')
        # Pooling 4
        # Hidden layer 1
        self.hW1 = tf.Variable(tf.truncated_normal([6*6*256, 512], dtype=tf.float32, stddev=1e-1), name='hid_weights1')
        self.hb1 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='hid_biases1')
        # Hidden layer 2
        self.hW2 = tf.Variable(tf.truncated_normal([512, 256], dtype=tf.float32, stddev=1e-1), name='hid_weights2')
        self.hb2 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='hid_biases2')
        # Output layer
        self.W_out = tf.Variable(tf.truncated_normal([256, self.num_outputs], dtype=tf.float32, stddev=1e-1),
                                 name='weights_out')  # tf.random_normal
        self.b_out = tf.Variable(tf.constant(0.0, shape=[self.num_outputs], dtype=tf.float32), trainable=True,
                                 name='biases_out')  # tf.random_normal

    @staticmethod
    def conv_layer(last_conv, weights, biases):
        conv = tf.nn.conv2d(last_conv, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv_with_b = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(conv_with_b)
        return conv_out

    @staticmethod
    def maxpool_layer(conv, k=2):
        max_pool = tf.nn.max_pool(conv, ksize=k, strides=k, padding='SAME')  # k or [1, k, k, 1]
        return max_pool

    def model_op(self):
        # Input
        images = self.imgs - self.img_mean

        # Convert layer 1_1
        conv_1_1 = self.conv_layer(images, self.W1_1, self.b1_1)
        # Convert layer 1_2
        conv_1_2 = self.conv_layer(conv_1_1, self.W1_2, self.b1_2)
        # Pooling 1  96x96 -> 48x48
        max_pool_1 = self.maxpool_layer(conv_1_2, k=2)

        # Convert layer 2_1
        conv_2_1 = self.conv_layer(max_pool_1, self.W2_1, self.b2_1)
        # Convert layer 2_2
        conv_2_2 = self.conv_layer(conv_2_1, self.W2_2, self.b2_2)
        # Pooling 2  48x48 -> 24x24
        max_pool_2 = self.maxpool_layer(conv_2_2, k=2)

        # Convert layer 3_1
        conv_3_1 = self.conv_layer(max_pool_2, self.W3_1, self.b3_1)
        # Convert layer 3_2
        conv_3_2 = self.conv_layer(conv_3_1, self.W3_2, self.b3_2)
        # Convert layer 3_3
        # conv_3_3 = self.conv_layer(conv_3_2, self.W3_3, self.b3_3)
        # # Pooling 3  24x24 -> 12x12
        max_pool_3 = self.maxpool_layer(conv_3_2, k=2)

        # Convert layer 4_1
        conv_4_1 = self.conv_layer(max_pool_3, self.W4_1, self.b4_1)
        # Convert layer 4_2
        conv_4_2 = self.conv_layer(conv_4_1, self.W4_2, self.b4_2)
        # # Convert layer 4_3
        # conv_4_3 = self.conv_layer(conv_4_2, self.W4_3, self.b4_3)
        # Pooling 3  24x24 -> 12x12
        max_pool_4 = self.maxpool_layer(conv_4_2, k=2)

        # Hidden layer 1
        shape = int(np.prod(max_pool_4.get_shape()[1:]))
        max_pool_4_flat = tf.reshape(max_pool_4, [-1, shape])
        hidden_1 = tf.nn.bias_add(tf.matmul(max_pool_4_flat, self.hW1), self.hb1)
        hidden_1 = tf.nn.sigmoid(hidden_1)  # 55% 15 epochs
        # Hidden layer 2
        hidden_2 = tf.nn.bias_add(tf.matmul(hidden_1, self.hW2), self.hb2)
        hidden_2 = tf.nn.sigmoid(hidden_2)

        # Output layer
        out = tf.nn.bias_add(tf.matmul(hidden_2, self.W_out), self.b_out)
        return out

    def load(self, sess):
        self.saver.restore(sess, self.path)
        print(f"Model loaded")

    def test(self, data, labels, sess):
        prediction = np.zeros((len(labels), self.num_outputs))
        accuracy_average, cost_all = [], []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size, :]
            batch_labels = labels[i:i + self.batch_size]
            model, accuracy, cost = sess.run([self.prob, self.accuracy, self.cost],
                                             feed_dict={self.imgs: batch_data, self.y: batch_labels})
            cost_all.append(cost)
            accuracy_average.append(accuracy)
            prediction[i:i + self.batch_size] = model
        return prediction, sum(accuracy_average) / len(accuracy_average), cost_all

    def train(self, data, labels, test_data, test_labels):  # test_data, test_labels
        start = time()
        with tf.Session(config=self.config) as sess:
            # summary_writer = tf.summary.FileWriter('../logs', sess.graph)
            sess.run(tf.global_variables_initializer())
            length = len(data)
            iters = length // self.batch_size
            best_test_accuracy = 0
            for epoch in range(1, self.epochs + 1):
                s = time()
                accuracy_total = cost_total = 0
                for i in range(0, len(data), self.batch_size):
                    batch_data = data[i:i + self.batch_size]
                    batch_labels = labels[i:i + self.batch_size]
                    _, accuracy_val, cost = sess.run([self.train_op, self.accuracy, self.cost],
                                                     feed_dict={self.imgs: batch_data, self.y: batch_labels})
                    # summary_writer.add_summary(summary, i)
                    accuracy_total += accuracy_val
                    cost_total += cost

                    progress_bar(i, length, prefix=f"EPOCH-{epoch}:\t",
                                 suffix=f"[{i}/{length}]\tAccuracy: {accuracy_val * 100 :.2f}%  Loss: {cost:.2f}")

                    if i == (iters - 1) * self.batch_size:  # end
                        progress_bar(1, 1, prefix=f"EPOCH-{epoch}:\t",
                                     suffix=f"[{length}/{length}]\tTime: {time() - s :.2f} sec")
                        print(f"Accuracy: {accuracy_total / iters * 100 :.2f}%\tLoss: {cost_total / iters:.3f}")

                if epoch % 2 == 0:
                    print("\nTest model on test images...")
                    _, test_accuracy, test_cost = self.test(test_data, test_labels, sess)
                    print(f"Accuracy: {test_accuracy * 100 :.2f}%\tCost: {sum(test_cost) / len(test_cost) :.3f}\n")
                    # if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    print("Model saved")
                    # self.saver.save(sess, self.path)
        # summary_writer.close()
        print(f"Total time: {time() - start :.0f} sec")


if __name__ == "__main__":
    cats_train, cats_test = get_dataset('cat', max_imgs=10_000, train_rate=0.9)
    # print(cats_train.shape, cats_test.shape)
    dogs_train, dogs_test = get_dataset('dog', max_imgs=10_000, train_rate=0.9)
    # print(dogs_train.shape, dogs_test.shape)

    train_labels = np.matrix([[1., 0.]] * cats_train.shape[0] + [[0., 1.]] * dogs_train.shape[0])
    test_labels = np.matrix([[1., 0.]] * cats_test.shape[0] + [[0., 1.]] * dogs_test.shape[0])
    train_imgs = np.vstack((cats_train, dogs_train))
    test_imgs = np.vstack((cats_test, dogs_test))
    del cats_train, cats_test, dogs_train, dogs_test  # clear memory
    print(train_labels, train_imgs.shape, train_imgs.shape)

    arr_train = np.arange(train_imgs.shape[0])
    print(arr_train)
    np.random.shuffle(arr_train)
    print(arr_train)
    train_imgs, train_labels = train_imgs[arr_train], train_labels[arr_train]

    arr_test = np.arange(test_imgs.shape[0])
    np.random.shuffle(arr_test)
    test_imgs, test_labels = test_imgs[arr_test], test_labels[arr_test]

    cnn = Cnn_CvsD(epochs=100, learning_rate=0.001, batch_size=40)
    cnn.train(train_imgs, train_labels, test_imgs, test_labels)
    # with tf.Session() as session:
    #     cnn.load(session)
    #     prob, accuracy_, cost_average = cnn.test(test_imgs, test_labels, session)
    #
    # print(f"Accuracy: {accuracy_ * 100 :.2f}%")
    # for j in range(0, len(prob), 4):
    #     imgs_ = test_imgs[j:j + 4]
    #     labels_, predict = test_labels[j:j + 4], prob[j:j + 4]
    #     lab = []
    #     print(labels_, predict)
    #     for l, p in zip(labels_, predict):
    #         # a = "КОТ" if l[0] == 1. else "СОБАКА"
    #         # if l.index(max(l)) == p.index(max(p)):
    #         #     c = 'g'
    #         # else:
    #         #     c = 'r'
    #         lab.append(f'[cat/dog] {p}')
    #     show_imgs(imgs_, lab, place=(2, 2))
