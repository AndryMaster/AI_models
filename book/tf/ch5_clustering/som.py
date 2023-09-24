import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class SOM:
    num_iters = 500

    def __init__(self, width, height, dim):  # ш, в, размерность
        self.width = width
        self.height = height
        self.dim = dim
        self.node_locs = self.get_locs()

        # Each node is a vector of dimension `dim`
        # For a 2D grid, there are `width * height` nodes
        self.nodes = tf.Variable(tf.random_normal([width*height, dim]))

        # These two ops are inputs at each iteration
        self.x = tf.placeholder(tf.float32, [dim])
        self.iter = tf.placeholder(tf.float32)

        # Find the node that matches closest to the input
        # bmu_loc = self.get_bmu_loc(self.x)  # bmu - лучшая еденица соответствия
        self.propagate_nodes = self.get_propagation(self.x, self.iter)

    def get_locs(self):
        locs = [[x, y] for y in range(self.height) for x in range(self.width)]
        return tf.to_float(locs)

    def get_bmu_loc(self, x):
        expanded_x = tf.expand_dims(x, 0)
        sqr_diff = tf.square(tf.subtract(expanded_x, self.nodes))
        dists = tf.reduce_sum(sqr_diff, 1)
        bmu_idx = tf.cast(tf.argmin(dists, 0), tf.int32)  # tf.mod(tf.int32 only)
        bmu_loc = tf.stack([tf.mod(bmu_idx, self.width), tf.div(bmu_idx, self.width)])  # !!!
        return bmu_loc

    # !!! -> tf.pack() -> tf.stack()
    def get_propagation(self, x, iter):
        num_nodes = self.height * self.width
        rate = 1.0 - tf.div(iter, self.num_iters)
        sigma = rate * tf.to_float(tf.maximum(self.height, self.width)) / 2.
        bmu_loc = self.get_bmu_loc(x)
        expanded_bmu_loc = tf.expand_dims(tf.cast(bmu_loc, tf.float32), 0)
        sqr_dists_from_bmu = tf.reduce_sum(tf.square(tf.subtract(expanded_bmu_loc, self.node_locs)), 1)
        neighbor_factor = tf.exp(-tf.div(sqr_dists_from_bmu, 2 * tf.square(sigma)))
        rate = tf.multiply(rate * 0.5, neighbor_factor)
        rate_factor = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.dim]) for i in range(num_nodes)])  # !!!
        nodes_diff = tf.multiply(rate_factor, tf.subtract(tf.stack([x for _ in range(num_nodes)]), self.nodes))  # !!!
        update_nodes = tf.add(self.nodes, nodes_diff)
        return tf.assign(self.nodes, update_nodes)

    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_iters):
                print(f"Iter_{i+1}", f":\t{self.nodes.eval()}" if (i+1) % 50 == 0 else "")
                for data_x in data:
                    sess.run(self.propagate_nodes, feed_dict={self.x: data_x, self.iter: i})
            centroid_grid = [[] for _ in range(self.width)]
            self.nodes_val = list(sess.run(self.nodes))
            self.locs_val = list(sess.run(self.node_locs))
            for i, l in enumerate(self.locs_val):
                centroid_grid[int(l[0])].append(self.nodes_val[i])
            self.centroid_grid = centroid_grid
