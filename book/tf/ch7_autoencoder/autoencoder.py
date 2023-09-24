import numpy as np
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def get_batch(X, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a]


class AutoEncoder:
    def __init__(self, input_dim, hidden_dim, epoch=250, batch_size=10, learning_rate=0.001):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        x = tf.placeholder(tf.float32, shape=(None, input_dim))

        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal((input_dim, hidden_dim), dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros(hidden_dim), name='biases')
            encoded = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal((hidden_dim, input_dim), dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros(input_dim), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        # RMSPropOptimizer; AdamOptimizer;
        self.saver = tf.train.Saver()

    def train(self, data):
        num_samples = len(data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, self.epoch+1):
                for i in range(num_samples // self.batch_size):
                    batch_data = data[i*self.batch_size:(i+1)*self.batch_size]
                    loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data})
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}:\tloss = {loss :.4f}")
                    self.saver.save(sess, './model.ckpt')

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: [data]})
        print(f"Input: {data}\nCompressed: {hidden}\nReconstructed: {reconstructed}\n")
        return reconstructed
