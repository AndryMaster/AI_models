import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
start = time()

learning_rate = 0.1
training_epochs = 2000

x1_label1 = np.random.normal(3, 1, 1000)
x2_label1 = np.random.normal(2, 1, 1000)
x1_label2 = np.random.normal(7, 1, 1000)
x2_label2 = np.random.normal(6, 1, 1000)
x1s = np.append(x1_label1, x1_label2)
x2s = np.append(x2_label1, x2_label2)
labels = np.asarray([0.] * len(x1_label1) + [1.] * len(x1_label2))

X1 = tf.placeholder(tf.float32, shape=(None,), name='x1_pos')
X2 = tf.placeholder(tf.float32, shape=(None,), name='x2_pos')
Y = tf.placeholder(tf.float32, shape=(None,), name='label')
w = tf.Variable([0., 0., 0.], name='parameters', trainable=True)

y_model = tf.sigmoid(-(w[2] * X2 + w[1] * X1 + w[0]))
cost = tf.reduce_mean(-tf.log(y_model * Y + (1 - y_model) * (1 - Y)))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    prev_cost = 0
    for epoch in range(1, training_epochs + 1):
        curr_cost, _ = sess.run([cost, train_op], {X1: x1s, X2: x2s, Y: labels})
        print(f"Epoch {epoch}\tCost: {curr_cost :.4f}")
        if abs(prev_cost - curr_cost) < 0.0001:
            break
        prev_cost = curr_cost

    w_val = sess.run(w)


print(f"Time: {time() - start :.3f}\tParams: {w_val}")

sigmoid = lambda x: 1. / (1. + np.exp(-x))
x1_boundary, x2_boundary = [], []
for x1_test in np.linspace(0, 10, 101):
    for x2_test in np.linspace(0, 10, 101):
        z = sigmoid(-x2_test * w_val[2] - x1_test * w_val[1] - w_val[0])
        if abs(z - 0.61) < 0.01:
            x1_boundary.append(x1_test)
            x2_boundary.append(x2_test)

plt.scatter(x1_boundary, x2_boundary, c='b', marker='o', s=20, alpha=0.8)
plt.scatter(x1_label1, x2_label1, c='r', marker='x', s=20, alpha=0.6)
plt.scatter(x1_label2, x2_label2, c='g', marker='1', s=20, alpha=0.7)

plt.show()
