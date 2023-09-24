import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

start = time()

x_label0 = np.append(np.random.normal(5, 1, 10), 20)
x_label1 = np.random.normal(2, 1, 10)
xs = np.append(x_label0, x_label1)
labels = [0.] * len(x_label0) + [1.] * len(x_label1)

print(x_label0)
print(x_label1)
print(labels)
plt.scatter(xs, labels)

learning_rate = 0.001  # 0.0007
training_epochs = 1000

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
    return X * w[1] + w[0]  # 1.95 -> 1.6
    # return tf.add(tf.multiply(w[1], X), w[0])
    # return tf.add(tf.multiply(w[1], tf.pow(X, 1)),
    #               tf.multiply(w[0], tf.pow(X, 0)))


w = tf.Variable([0., 0.], name="parameters")
y_model = model(X, w)
cost = tf.reduce_sum(tf.square(Y-y_model))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for epoch in range(training_epochs):
    sess.run(train_op, feed_dict={X: xs, Y: labels})

    if (epoch + 1) % 100 == 0:
        current_cost = sess.run(cost, feed_dict={X: xs, Y: labels})
        print(epoch+1, current_cost)

correct_prediction = tf.equal(Y, tf.to_float(tf.greater(y_model, 0.5)))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

w_val = sess.run(w)
print(f"Accuracy: {sess.run(accuracy, feed_dict={X: xs, Y: labels}) :.3f}\n"
      f"W: {w_val}\nTime: {time() - start :.2f}")

sess.close()

all_xs = np.linspace(0, 10, 101)
plt.plot(all_xs, all_xs * w_val[1] + w_val[0])
plt.show()
