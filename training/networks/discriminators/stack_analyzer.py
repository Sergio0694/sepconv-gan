import tensorflow as tf
from __MACRO__ import BATCH_SIZE, TRAINING_IMAGES_SIZE

def get_network(x):

    x.set_shape([BATCH_SIZE, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 9]) # ensure the shape is valid

    def block(tensor, filters):
        down_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_conv2 = tf.layers.conv2d(down_conv1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_conv3 = tf.layers.conv2d(down_conv2, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        return down_conv3 + tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')

    with tf.variable_scope('analyzer', None, [x]):

        with tf.variable_scope('encoder', None, [x]):
            with tf.variable_scope('downscale_1', None, [x]):
                conv1 = block(x, 32)
                pool1 = tf.layers.average_pooling2d(conv1, 2, 2, padding='same')

            with tf.variable_scope('downscale_2', None, [pool1]):
                conv2 = block(pool1, 64)
                pool2 = tf.layers.average_pooling2d(conv2, 2, 2, padding='same')

            with tf.variable_scope('downscale_3', None, [pool2]):
                conv3 = block(pool2, 128)
                pool3 = tf.layers.average_pooling2d(conv3, 2, 2, padding='same')

            with tf.variable_scope('downscale_4', None, [pool3]):
                conv4 = block(pool3, 256)
                pool4 = tf.layers.average_pooling2d(conv4, 2, 2, padding='same')

            with tf.variable_scope('downscale_5', None, [pool4]):
                conv5 = block(pool4, 512)
                pool5 = tf.layers.average_pooling2d(conv5, 2, 2, padding='same')
            
            with tf.variable_scope('downscale_6', None, [pool5]):
                conv6 = block(pool5, 512)
                pool6 = tf.layers.average_pooling2d(conv6, 2, 2, padding='same')
            
            with tf.variable_scope('fc', None, [pool6]):
                flat = tf.reshape(pool6, [pool6.shape[0], -1])
                d1 = tf.layers.dense(flat, 2048, activation=tf.nn.leaky_relu)
                d2 = tf.layers.dense(d1, 1024, activation=tf.nn.leaky_relu)
                dropout = tf.layers.dropout(d2, 0.8)
                d3 = tf.layers.dense(dropout, 1)
                return d3