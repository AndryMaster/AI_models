import numpy as np
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def get_data():
    data_a = np.random.rand(12, n_features) + 1
    data_b = np.random.rand(12, n_features)

    plt.scatter(data_a[:, 0], data_a[:, 1], c='g', marker='o')
    plt.scatter(data_b[:, 0], data_b[:, 1], c='r', marker='x')
    plt.show()
    return data_a, data_b


n_hidden = 10
n_features = 2
learning_rate = 0.01  # 0.001
a_data, b_data = get_data()

with tf.name_scope("input"):
    x1 = tf.placeholder(tf.float32, shape=[None, n_features], name="preferred")
    x2 = tf.placeholder(tf.float32, shape=[None, n_features], name="non-preferred")
    dropout_rate = tf.placeholder(tf.float32, name="dropout")

with tf.name_scope("hidden_layer"):
    with tf.name_scope("weights"):
        W1 = tf.Variable(tf.random_normal([n_features, n_hidden]), name="W1")
        b1 = tf.Variable(tf.random_normal([n_hidden]), name="b1")
        tf.summary.histogram("W1", W1)
        tf.summary.histogram("b1", b1)

    with tf.name_scope("output"):
        # tf.nn.dropout - older; keep_prob = (rate-1); tf.nn.experimental.stateless_dropout
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x1, W1) + b1), rate=dropout_rate)
        h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(x2, W1) + b1), rate=dropout_rate)
        tf.summary.histogram("h1", h1)
        tf.summary.histogram("h2", h2)

with tf.name_scope("output_layer"):
    with tf.name_scope("weights"):
        W2 = tf.Variable(tf.random_normal([n_hidden, 1]), name="W2")
        b2 = tf.Variable(tf.random_normal([1]), name="b2")
        tf.summary.histogram("W2", W2)
        tf.summary.histogram("b2", b2)

    with tf.name_scope("output"):
        s1 = tf.matmul(h1, W2) + b2
        s2 = tf.matmul(h2, W2) + b2

with tf.name_scope("loss"):
    s12 = s2 - s1  # s1 - s2 old (for correct land_shaft)
    s12_flat = tf.reshape(s12, shape=[-1])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.zeros_like(s12_flat),
        logits=s12_flat + 1)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("loss", loss)

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


sess = tf.InteractiveSession()
summary_op = tf.summary.merge_all()
# writer = tf.summary.FileWriter("./logs", sess.graph)
sess.run(tf.global_variables_initializer())

for epoch in range(2001):  # 10001
    loss_val, _ = sess.run([loss, train_op], feed_dict={x1: a_data, x2: b_data, dropout_rate: 0.45})
    if epoch % 100 == 0:
        summary = sess.run(summary_op, feed_dict={x1: a_data, x2: b_data, dropout_rate: 0.0})
        print(f"Epoch {epoch}: {loss_val}")
        # writer.add_summary(summary, epoch)

# writer.close()


def visualize_results(data, out, elem):
    plt.figure()
    scores_data = sess.run(out, feed_dict={elem: data, dropout_rate: 0.0})
    scores_img = np.reshape(scores_data, [grid_size, grid_size])
    plt.imshow(scores_img, origin="lower")
    plt.colorbar()
    plt.show()


grid_size = 20
data_test = [(x, y) for y in np.linspace(0., 1., grid_size) for x in np.linspace(0., 1., grid_size)]
visualize_results(data_test, s1, x1)
visualize_results(data_test, s2, x2)

sess.close()
