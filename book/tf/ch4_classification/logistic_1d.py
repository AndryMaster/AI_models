import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
start = time()

learning_rate = 0.005  # 0.01
training_epochs = 2000

x0 = np.random.normal(-4, 2, 1000)
x1 = np.random.normal(4, 2, 1000)
xs = np.append(x0, x1)
ys = np.asarray([0.] * len(x0) + [1.] * len(x1))

X = tf.placeholder(tf.float32, shape=(None,), name='x')
Y = tf.placeholder(tf.float32, shape=(None,), name='y')
w = tf.Variable([0., 0.], name='parameters')  # trainable=True
y_model = tf.sigmoid(w[1] * X + w[0])
cost = tf.reduce_mean(-tf.log(y_model * Y + (1 - y_model) * (1 - Y)))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_model, Y))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    prev_cost = 0
    for epoch in range(1, training_epochs + 1):
        curr_cost, _ = sess.run([cost, train_op], {X: xs, Y: ys})
        print(f"Epoch {epoch}\tCost: {curr_cost :.4f}")
        if abs(prev_cost - curr_cost) < 0.0001:
            break
        prev_cost = curr_cost
    w_val = sess.run(w)


print(f"Time: {time() - start :.3f}\tParams: {w_val}")
sigmoid = lambda x: 1. / (1. + np.exp(-x))
all_xs = np.linspace(-10, 10, 101)
plt.scatter(xs, ys, alpha=0.15)
plt.plot(all_xs, sigmoid(all_xs * w_val[1] + w_val[0]))
plt.show()
