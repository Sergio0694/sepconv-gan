import tensorflow as tf

def get_network(x):
    '''Generates a simple CNN to perform image interpolation, based on the FI_CNN_model
    network defined here: https://github.com/neil454/deep-motion/blob/master/src/FI_CNN.py#L6.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    # setup the inputs
    x_shape = tf.shape(x, name='batch_shape')
    with tf.name_scope('frames'):
        x_r = tf.reshape(x, [x_shape[0], x_shape[2], x_shape[3], 6], name='frames')
        x_norm = x_r / 255.0

    conv1 = tf.layers.conv2d(x_norm, 32, 3, activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')
    conv2 = tf.layers.conv2d(pool1, 32, 3, activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')

    conv3 = tf.layers.conv2d(pool2, 32, 3, activation=tf.nn.relu, padding='same')
    up1_s = tf.shape(conv2)
    up1 = tf.image.resize_nearest_neighbor(conv3, [up1_s[1], up1_s[2]])
    conv4 = tf.layers.conv2d(up1, 32, 3, activation=tf.nn.relu, padding='same')
    up2_s = tf.shape(conv1)
    up2 = tf.image.resize_nearest_neighbor(conv4, [up2_s[1], up2_s[2]])

    y = tf.layers.conv2d(up2, 3, 3, activation=tf.nn.sigmoid, padding='same')
    return y
