import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

start = time()

learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1, 1, 101)
num_coeffs = 9
y_train_coeffs = [2, 6, -7, 4, 0, -2, 4, 1, 3]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
y_train = 0
for i in range(num_coeffs):
    y_train += y_train_coeffs[i] * np.power(x_train, i)
y_train += np.random.randn(*x_train.shape) * 0.8

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

cost = tf.pow(Y - y_model, 2)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    for x, y in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})

w_val = sess.run(w)
print(f"Parameters: {w_val}\nReal (no random) parameters: {y_train_coeffs}\nTime: {time() - start :.1f}")

print(sess.run(y_model, feed_dict={X: 1.172}))
print(sess.run(y_model, feed_dict={X: 1.5}))

sess.close()

plt.scatter(x_train, y_train)
y_train2 = 0
for i in range(num_coeffs):
    y_train2 += w_val[i] * np.power(x_train, i)
plt.plot(x_train, y_train2, 'r')
plt.show()
