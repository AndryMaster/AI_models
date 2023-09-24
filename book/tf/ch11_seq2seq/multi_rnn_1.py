import tensorflow._api.v2.compat.v1 as tf
from tensorflow.python.ops import rnn, rnn_cell
tf.disable_v2_behavior()

input_dim = 1
seq_size = 3  # 6

input_placeholder = tf.placeholder(tf.float32, shape=[None, seq_size, input_dim])


def make_cell(state_dim):
    return tf.nn.rnn_cell.LSTMCell(state_dim)


# with tf.variable_scope('first_cell') as scope:
#     cell1 = make_cell(state_dim=10)
#     outputs1, states1 = tf.nn.dynamic_rnn(cell1, input_placeholder, dtype=tf.float32)
#
#
# with tf.variable_scope('second_cell') as scope:
#     cell2 = make_cell(state_dim=10)
#     outputs2, states2 = tf.nn.dynamic_rnn(cell2, outputs1, dtype=tf.float32)


def make_multi_cell(state_dim, num_layers):
    cells = [make_cell(state_dim) for _ in range(num_layers)]
    return tf.nn.rnn_cell.MultiRNNCell(cells)


multi_cell = make_multi_cell(state_dim=10, num_layers=4)
outputs4, states4 = tf.nn.dynamic_rnn(multi_cell, input_placeholder, dtype=tf.float32)
