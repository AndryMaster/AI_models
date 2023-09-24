from vgg16 import vgg16
# from imagenet_classes import class_names

import glob, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

DATASET_DIR = '../datasets/cloth_folding_rgb_vids/'
NUM_VIDS = 45


def get_image(path, resize=(224, 224), ch3=True):
    img = np.array(Image.Image.resize(Image.open(path, mode='RGB'), resize))
    # if ch3:
    #     img = img[:, :, :3]
    return img


def get_img_seq(video_id):
    img_files = sorted(glob.glob(os.path.join(DATASET_DIR, str(video_id), '*.png')))
    imgs = [get_image(image_file) for image_file in img_files]
    return imgs


def get_img_pair(video_id):
    img_files = sorted(glob.glob(os.path.join(DATASET_DIR, str(video_id), '*.png')))
    start_img_ = img_files[0]
    end_img_ = img_files[-1]
    return tuple([get_image(start_img_), get_image(end_img_)])


start_imgs = []
end_imgs = []
for vid_id in range(1, NUM_VIDS+1):
    start_img, end_img = get_img_pair(vid_id)
    start_imgs.append(start_img)
    end_imgs.append(end_img)

print(f'Images of starting state {np.shape(start_imgs)}')
print(f'Images of ending state {np.shape(end_imgs)}')

n_features = 4096
n_hidden = n_features * 2  # n_features * 2

imgs_plc = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

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
    s12 = s1 - s2
    s12_flat = tf.reshape(s1 - s2, [-1])
    # pred = tf.sigmoid(s12)
    # lable_p = tf.sigmoid(-tf.ones_like(s12))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.zeros_like(s12_flat),
        logits=s12_flat + 1)

    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("loss", loss)

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)


sess = tf.InteractiveSession()
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs", sess.graph)
sess.run(tf.global_variables_initializer())

print('Loading model...')
vgg = vgg16(imgs_plc, sess, 'vgg16_weights.npz')
print('Done loading!')

start_imgs_embedded = sess.run(vgg.fc1, feed_dict={vgg.imgs: start_imgs})
end_imgs_embedded = sess.run(vgg.fc1, feed_dict={vgg.imgs: end_imgs})

idx = np.random.choice(NUM_VIDS, NUM_VIDS, replace=False)
train_idx, test_idx = idx[:round(NUM_VIDS * 0.75)], idx[round(NUM_VIDS * 0.75):]

train_start_imgs = start_imgs_embedded[train_idx]
train_end_imgs = end_imgs_embedded[train_idx]
test_start_imgs = start_imgs_embedded[test_idx]
test_end_imgs = end_imgs_embedded[test_idx]

print(f'Train start imgs {np.shape(train_start_imgs)}')
print(f'Train end imgs {np.shape(train_end_imgs)}')
print(f'Test start imgs {np.shape(test_start_imgs)}')
print(f'Test end imgs {np.shape(test_end_imgs)}')

train_y1 = np.expand_dims(np.zeros(np.shape(train_start_imgs)[0]), axis=1)
train_y2 = np.expand_dims(np.ones(np.shape(train_end_imgs)[0]), axis=1)
for epoch in range(100):
    for i in range(np.shape(train_start_imgs)[0]):
        _, loss_val = sess.run([train_op, loss],
                               feed_dict={x1: train_start_imgs[i:i+1,:],
                                          x2: train_end_imgs[i:i+1,:],
                                          dropout_rate: 0.5})
        # print(f'{epoch + 1}: {loss_val}')
    s1_val, s2_val = sess.run([s1, s2], feed_dict={x1: test_start_imgs, x2: test_end_imgs, dropout_rate: 0.})
    print(f'{epoch} Accuracy: {np.mean(s1_val < s2_val) * 100 :.2f}%')

writer.close()


def test_vid(vid_id):
    imgs = get_img_seq(vid_id)
    print(np.shape(imgs))
    imgs_embedded = []
    for img in imgs:
        imgs_embedded.append(sess.run(vgg.fc1, feed_dict={vgg.imgs: [img]}))
    imgs_embedded = np.squeeze(imgs_embedded)
    print(imgs_embedded.shape, imgs_embedded[0])
    scores = sess.run([s1], feed_dict={x1: imgs_embedded, dropout_rate: 0.})
    # print(scores[-1])

    plt.figure()
    plt.title('Utility of cloth-folding over time')
    plt.xlabel(f'Time (video_{vid_id} frame #)')
    plt.ylabel('Utility')
    plt.plot(scores[-1])
    plt.show()


for i in range(1, NUM_VIDS+1):
    test_vid(i)

sess.close()
