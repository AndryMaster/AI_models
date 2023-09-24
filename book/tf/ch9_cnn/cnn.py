import os
import cifar_tools
from time import time
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ERROR: -1073740791 (0xC0000409)

learning_rate = 0.001
names, data, labels = cifar_tools.read_data("../datasets/cifar/cifar-10-python", test=False)
# len(names) = 10; ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

x = tf.placeholder(tf.float32, shape=(None, 24 * 24))
y = tf.placeholder(tf.float32, shape=(None, len(names)))

W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
b1 = tf.Variable(tf.random_normal([64]))

W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
b2 = tf.Variable(tf.random_normal([64]))

W3 = tf.Variable(tf.random_normal([6*6*64, 1024]))  # (9216)2304x1024
b3 = tf.Variable(tf.random_normal([1024]))

W_out = tf.Variable(tf.random_normal([1024, len(names)]))  # 1024x10
b_out = tf.Variable(tf.random_normal([len(names)]))

# W1_summary = tf.summary.image('W1_img', tf.reshape(W1, (64, 5, 5, 1)))  # !!!
# merged = tf.summary.merge_all()  # !!!


def conv_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out


def maxpool_layer(conv, k=2):
    max_pool = tf.nn.max_pool(conv, ksize=k, strides=k, padding='SAME')  # k or [1, k, k, 1]
    return max_pool


def model():
    x_reshaped = tf.reshape(x, shape=(-1, 24, 24, 1))

    conv_out1 = conv_layer(x_reshaped, W1, b1)  # NxNx64
    maxpool_out1 = maxpool_layer(conv_out1)  # 24x24 -> 12x12
    norm1 = tf.nn.lrn(maxpool_out1, depth_radius=4, bias=1., alpha=0.001 / 9, beta=0.75)

    conv_out2 = conv_layer(norm1, W2, b2)  # NxNx64
    norm2 = tf.nn.lrn(conv_out2, depth_radius=4, bias=1., alpha=0.001 / 9, beta=0.75)
    maxpool_out2 = maxpool_layer(norm2)  # 12x12 -> 6x6; 3.5 -> 4 sec; accuracy + 15%

    maxpool_reshaped = tf.reshape(maxpool_out2, shape=(-1, W3.get_shape().as_list()[0]))  # maxpool_out2
    local = tf.matmul(maxpool_reshaped, W3) + b3
    local_out = tf.nn.relu(local)

    out = tf.matmul(local_out, W_out) + b_out
    return out


model_op = model()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_op, labels=y))  # v1 -> v2; 3.7 -> 3.5
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

start = time()
epochs = 300
config = tf.ConfigProto()  # 4.0 -> 3.8 + low memory; 3.8 -> 3.7 msi afterburner
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # summary_writer = tf.summary.FileWriter('../logs', sess.graph)  # !!!
    sess.run(tf.global_variables_initializer())
    one_hot_labels = tf.one_hot(labels, len(names), on_value=1., off_value=0., axis=-1)
    one_hot_vals = sess.run(one_hot_labels)
    batch_size = 250  # len(data) / 200
    saver = tf.train.Saver()
    # saver.restore(sess, './cnn_model.ckpt')
    for epoch in range(epochs):
        s = time()
        print(f"EPOCH: {epoch+1}")
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size, :]
            batch_labels = one_hot_vals[i:i+batch_size, :]
            _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: batch_data, y: batch_labels})
            # summary_writer.add_summary(summary, i)  # !!! up
            if i % 2000 == 0:
                print(f"\t{i+2000}-img accuracy: {accuracy_val * 100 :.1f}%")

        # saver.save(sess, './cnn_model.ckpt')
        print(f"END EPOCH  Time: {time() - s :.2f} sec\nSaved model cnn")

# summary_writer.close()  # !!!  tensorboard --logdir=./logs
print(f"Total time: {time() - start :.0f} sec")
