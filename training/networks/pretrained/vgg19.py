import numpy as np
import tensorflow as tf
from __MACRO__ import BATCH_SIZE, TRAINING_IMAGES_SIZE
from networks._tf import ROOT_SCOPE

BGR_MEAN_PIXELS = np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3)).astype(np.float32)
RELU4_4_NAME = 'block4_conv4/Relu:0'
OUTPUT_POOL_NAME = 'block5_pool/MaxPool:0'

def get_base(x):
    '''Gets the base of the VGG19 network, without the fully connected layers.

    x(tf.Tensor) -- the input tensor
    '''

    x.set_shape([BATCH_SIZE, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3])
    with tf.variable_scope(ROOT_SCOPE, None, [x]):
        with tf.variable_scope('VGG19', None, [x], reuse=tf.AUTO_REUSE):
            processed_x = x - BGR_MEAN_PIXELS
            tf.contrib.keras.applications.VGG19(weights='imagenet', include_top=False, input_tensor=processed_x)
            name = '{}/{}'.format(tf.contrib.framework.get_name_scope(), OUTPUT_POOL_NAME)
            return tf.get_default_graph().get_tensor_by_name(name)

def get_loss(yHat, y):

    yHat.set_shape([BATCH_SIZE, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3])
    y.set_shape([BATCH_SIZE, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3])

    # VGG19 setup
    with tf.variable_scope(ROOT_SCOPE, None, [yHat, y]):
        with tf.variable_scope('VGG19', None, [yHat, y], reuse=tf.AUTO_REUSE):
            with tf.name_scope('yHat', [yHat]):
                processed_yHat = yHat - BGR_MEAN_PIXELS
                tf.contrib.keras.applications.VGG19(weights='imagenet', include_top=False, input_tensor=processed_yHat)
                name = '{}/{}'.format(tf.contrib.framework.get_name_scope(), RELU4_4_NAME)
                relu4_4_yHat = tf.get_default_graph().get_tensor_by_name(name)
            with tf.name_scope('y', [y]):
                processed_y = y - BGR_MEAN_PIXELS
                tf.contrib.keras.applications.VGG19(weights='imagenet', include_top=False, input_tensor=processed_y)
                name = '{}/{}'.format(tf.contrib.framework.get_name_scope(), RELU4_4_NAME)
                relu4_4_y = tf.get_default_graph().get_tensor_by_name(name)

    # perceptual loss
    return tf.reduce_mean((relu4_4_yHat - relu4_4_y) ** 2)