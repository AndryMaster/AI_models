import load_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.python.ops import rnn, rnn_cell
tf.disable_v2_behavior()


class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_size = seq_size
        self.epochs = 1000

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Самая распространенняа среднеквадратичная стоимость
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        self.saver = tf.train.Saver()

    def model(self):
        cell = rnn_cell.BasicLSTMCell(self.hidden_dim, reuse=tf.get_variable_scope().reuse)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        print(states)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])  # W * n_e[3] = WWW
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)  # shape [1, 2, 1, 3, 1, 1] to [2, 3]
        return out

    def save(self, sess, filename):
        path = self.saver.save(sess, filename)
        print(f"Model saved to {path}")

    def load(self, sess, path):
        self.saver.restore(sess, path)

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            max_patience = 5
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            while patience > 0:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print(f"Step: {step}\ttrain data err: {train_err}\ttest data err: {test_err}")
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1
            # self.save(sess, 'model.ckpt')

    def test(self, sess, test_x):
        self.load(sess, './model.ckpt')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output


def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == '__main__':
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1,
                                seq_size=seq_size,
                                hidden_dim=100)

    data = load_data.load_series('international-airline-passengers.csv')
    train_data, actual_vals = load_data.split_data(data, percent_train=0.76)

    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i + seq_size], axis=1).tolist())
        train_y.append(train_data[i + 1:i + seq_size + 1])

    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i + seq_size], axis=1).tolist())
        test_y.append(actual_vals[i + 1:i + seq_size + 1])

    print(len(train_x), len(train_y))
    print(len(test_x), len(test_y))

    predictor.train(train_x, train_y, test_x, test_y)
    with tf.Session() as sess:
        predicted_vals = predictor.test(sess, test_x)[:, 0]
        print('predicted_vals', np.shape(predicted_vals))
        plot_results(train_data, predicted_vals, actual_vals, 'predictions.png')

        prev_seq = train_x[-1]
        predicted_vals = []
        for i in range(1000):
            next_seq = predictor.test(sess, [prev_seq])
            predicted_vals.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        plot_results(train_data, predicted_vals, actual_vals, 'hallucinations.png')
