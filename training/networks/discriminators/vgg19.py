import tensorflow as tf
import networks.pretrained.vgg19 as vgg19

def get_network(x):
    '''Gets a discriminator network with the shared base of the VGG19 network.

    x(tf.Tensor) -- the network input
    '''

    x_base = vgg19.get_base(x)

    with tf.variable_scope('VGG19_top', None, [x_base]):
        d1 = tf.layers.dense(x_base, 4096, activation=tf.nn.leaky_relu)
        dropout1 = tf.layers.dropout(d1, 0.9)
        d2 = tf.layers.dense(dropout1, 2048, activation=tf.nn.leaky_relu)
        dropout2 = tf.layers.dropout(d2, 0.9)
        d3 = tf.layers.dense(dropout2, 1)
        return d3