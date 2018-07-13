import tensorflow as tf

def get_network(x):
    '''Gets a discriminator network with the shared base of the VGG19 network.

    x(tf.Tensor) -- the VGG19 base network
    '''

    with tf.variable_scope('VGG19_top', None, [x], reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(x, 512, 3, activation=tf.nn.leaky_relu, padding='same')
        conv2 = tf.layers.conv2d(conv1, 512, 3, activation=tf.nn.leaky_relu, padding='same') + x
        pool = tf.layers.max_pooling2d(conv2, 3, 2, padding='valid')
        flat = tf.reshape(pool, [pool.shape[0], -1])
        d1 = tf.layers.dense(flat, 1024, activation=tf.nn.leaky_relu)
        dropout1 = tf.layers.dropout(d1, 0.8)
        d2 = tf.layers.dense(dropout1, 1024, activation=tf.nn.leaky_relu)
        dropout2 = tf.layers.dropout(d2, 0.8)
        d3 = tf.layers.dense(dropout2, 1)
        return d3