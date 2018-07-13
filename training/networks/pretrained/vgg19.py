import numpy as np
import tensorflow as tf
from __MACRO__ import BATCH_SIZE, TRAINING_IMAGES_SIZE
import networks._tf as _tf

BGR_MEAN_PIXELS = np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3)).astype(np.float32)
RELU4_4_NAME_TOKENS = 'block4_conv4/Relu:0'.split('/')

def get_networks(args):
    '''Creates multiple instances of the VGG19, with shared weights.

    args(dict<str, tf.Tensor>) -- the map of input tensors and their name (used for the scope)
    '''

    if not args: return dict()

    # load the reusable VGG19 model
    with tf.variable_scope('VGG19', None, list(args.values()), reuse=tf.AUTO_REUSE):
        outputs = dict()
        model = tf.contrib.keras.applications.VGG19(weights='imagenet', include_top=False)
        
        # build the requested branches
        for key in args:
            args[key].set_shape([BATCH_SIZE, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3])
            with tf.name_scope(key, None, [args[key]]):
                processed_x = args[key] - BGR_MEAN_PIXELS
            branch = model(processed_x)
            with tf.name_scope(key, None, [branch]):
                relu4 = _tf.get_parent_by_match(branch, RELU4_4_NAME_TOKENS)
                outputs[key] = (relu4, branch)
        return outputs

def get_loss(yHat, y):

    # perceptual loss
    return tf.reduce_mean((yHat[0] - y[0]) ** 2)
