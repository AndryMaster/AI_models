import numpy as np
from main import split_dataset
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

learning_rate = 0.01
training_epochs = 50
reg_lambda = 0.

x_dataset = np.linspace(-1, 1, 101)

num_coeffs = 9
y_dataset_coeffs = [0.] * num_coeffs
y_dataset_coeffs[2] = 1.
y_dataset_coeffs[5] = 5.
y_dataset_coeffs[0] = -2.

y_dataset = 0
for i in range(num_coeffs):
    y_dataset += y_dataset_coeffs[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

# plt.scatter(x_dataset, y_dataset)
# plt.show()

x_train, x_test, y_train, y_test = split_dataset(x_dataset, y_dataset, 0.7)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)

cost = (tf.reduce_sum(tf.square(Y-y_model)) + reg_lambda * tf.reduce_sum(tf.square(w))) / 2 * x_train.size
# cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y - y_model)),
#                      tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))), 2 * x_train.size)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for reg_lambda in np.linspace(0, 1, 10):
    for epoch in range(training_epochs):
        for x, y in zip(x_train, y_train):
            sess.run(train_op, feed_dict={X: x, Y: y})
    final_cost = sess.run(train_op, feed_dict={X: x_test, Y: y_test})
    print(f"Regulation: {reg_lambda :.2f}\tParams: {sess.run(w)}\tFinal cost: {final_cost}")

# w_val = sess.run(w)
# print(f"Parameters: {w_val}"
#       f"Real (no random) parameters: {y_dataset_coeffs}")

sess.close()
