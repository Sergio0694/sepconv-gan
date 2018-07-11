import tensorflow as tf

def get_network(x):
    '''Gets a discriminator network with the shared base of the VGG19 network.

    x(tf.Tensor) -- the VGG19 base network
    '''

    with tf.variable_scope('VGG19_top', None, [x], reuse=tf.AUTO_REUSE):
        d1 = tf.layers.dense(x, 4096, activation=tf.nn.leaky_relu)
        dropout1 = tf.layers.dropout(d1, 0.9)
        d2 = tf.layers.dense(dropout1, 2048, activation=tf.nn.leaky_relu)
        dropout2 = tf.layers.dropout(d2, 0.9)
        d3 = tf.layers.dense(dropout2, 1)
        return d3