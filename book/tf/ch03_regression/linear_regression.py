import numpy as np
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

learning_rate = 0.01
training_epochs = 100

real_w, real_d = 2, 1.3
x_train = np.linspace(-1, 1, 101)
y_train = real_w * x_train + np.random.randn(*x_train.shape) * 0.33 + real_d

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def model(X, w, d):
    return tf.add(tf.multiply(X, w), d)  # X * w + d


w = tf.Variable(0.0, name="W_weight")
d = tf.Variable(0.0, name="D_weight")

y_model = model(X, w, d)
cost = tf.square(Y - y_model)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# cost_hist = tf.summary.scalar("Cost", cost)
# w_hist = tf.summary.scalar("W_parameter", w)
# d_hist = tf.summary.scalar("D_parameter", d)
# merge = tf.summary.merge_all()
# writer = tf.summary.FileWriter("../logs", sess.graph)

for epoch in range(training_epochs):
    for x, y in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X: x, Y: y})
        # summary_str, _ = sess.run([merge, train_op], feed_dict={X: x, Y: y})
        # writer.add_summary(summary_str, epoch)

# writer.close()

w_val = sess.run(w)
d_val = sess.run(d)
print(f"W: {w_val} [{real_w}]\nD: {d_val} [{real_d}]")

sess.close()

plt.scatter(x_train, y_train)
y_train = x_train * w_val + d_val
plt.plot(x_train, y_train, 'r')
plt.show()
