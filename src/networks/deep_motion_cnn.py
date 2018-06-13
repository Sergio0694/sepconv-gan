import tensorflow as tf

def stack_images(x):
    '''Stacks n images over the channels dimension.

    x(tf.tensor<tf.float32>) -- the input [batch, images, h, w, channels] tensor
    '''

    with tf.name_scope('frames'):
        x_t = tf.transpose(x, [0, 2, 3, 1, 4])
        x_shape = tf.shape(x_t, name='batch_shape')
        return tf.reshape(x_t, [x_shape[0], x_shape[1], x_shape[2], 6], name='frames')

def get_network_v1(x):
    '''Generates a simple CNN to perform image interpolation, based on the FI_CNN_model
    network defined here: https://github.com/neil454/deep-motion/blob/master/src/FI_CNN.py#L6.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    x_stack = stack_images(x)
    conv1 = tf.layers.conv2d(x_stack, 32, 3, activation=tf.nn.relu, padding='same')
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

def get_network_v2(x):
    '''Generates a variant of the simple CNN to perform image interpolation.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    x_stack = stack_images(x)
    conv1_a = tf.layers.conv2d(x_stack, 32, 3, activation=tf.nn.relu, padding='same')
    conv1_b = tf.layers.conv2d(conv1_a, 32, 3, activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(conv1_b, 2, 2, padding='same')

    conv2_a = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same')
    conv2_b = tf.layers.conv2d(conv2_a, 64, 3, activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(conv2_b, 2, 2, padding='same')

    conv3_a = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu, padding='same')
    conv3_b = tf.layers.conv2d(conv3_a, 128, 3, activation=tf.nn.relu, padding='same') + conv3_a
    up1_s = tf.shape(conv2_b)
    up1 = tf.image.resize_nearest_neighbor(conv3_b, [up1_s[1], up1_s[2]])

    conv4_a = tf.layers.conv2d(up1, 64, 3, activation=tf.nn.relu, padding='same') + conv2_b
    conv4_b = tf.layers.conv2d(conv4_a, 64, 3, activation=tf.nn.relu, padding='same')
    up2_s = tf.shape(conv1_b)
    up2 = tf.image.resize_nearest_neighbor(conv4_b, [up2_s[1], up2_s[2]])

    conv5_a = tf.layers.conv2d(up2, 32, 3, activation=tf.nn.relu, padding='same') + conv1_b
    conv5_b = tf.layers.conv2d(conv5_a, 32, 3, activation=tf.nn.relu, padding='same')

    y = tf.layers.conv2d(conv5_b, 3, 5, activation=tf.nn.sigmoid, padding='same')
    return y
