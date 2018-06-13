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

def get_network_v3(x):
    '''Generates a deeper variant of the simple CNN to perform image interpolation.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''
    
    # [batch, images, h, w, 3]
    x_stack = stack_images(x)
    norm_x = tf.layers.batch_normalization(x_stack)

    # [batch, h, w, 3 * images]
    with tf.name_scope('stem_a'):
        conv1_a = tf.layers.conv2d(norm_x, 32, 3, activation=tf.nn.relu, padding='same')
        conv1_b = tf.layers.conv2d(conv1_a, 32, 3, activation=tf.nn.relu, padding='same')
        conv1_c = tf.layers.conv2d(conv1_b, 32, 3, activation=tf.nn.relu, padding='same')
        conv1_d = tf.layers.conv2d(conv1_c, 32, 2, 2, activation=tf.nn.relu, padding='same')
        pool1 = tf.layers.max_pooling2d(conv1_c, 2, 2, padding='same')
        stack1 = tf.concat([conv1_d, pool1], 3)
        norm1 = tf.layers.batch_normalization(stack1)

    # [batch, h / 2, w / 2, 64]
    with tf.name_scope('stem_b'):
        conv2_b1_a = tf.layers.conv2d(norm1, 64, 1, activation=tf.nn.relu)
        conv2_b1_b = tf.layers.conv2d(conv2_b1_a, 64, 3, activation=tf.nn.relu, padding='same')
        conv2_b1_c = tf.layers.conv2d(conv2_b1_b, 46, 1, activation=tf.nn.relu)
        conv2_b2_a = tf.layers.conv2d(norm1, 64, 1, activation=tf.nn.relu)
        conv2_b2_b = tf.layers.conv2d(conv2_b2_a, 64, [7, 1], activation=tf.nn.relu, padding='same')
        conv2_b2_c = tf.layers.conv2d(conv2_b2_b, 64, [1, 7], activation=tf.nn.relu, padding='same')
        conv2_b2_d = tf.layers.conv2d(conv2_b2_c, 46, 3, activation=tf.nn.relu, padding='same')
        stack2 = tf.concat([conv2_b1_c, conv2_b2_d], 3)
        norm2 = tf.layers.batch_normalization(stack2)

    # [batch, h / 2, w / 2, 92]
    with tf.name_scope('stem_c'):
        conv3_b1 = tf.layers.conv2d(norm2, 128, 3, 2, activation=tf.nn.relu, padding='same')
        pool2 = tf.layers.max_pooling2d(norm2, 2, 2, padding='same')
        conv3_b2 = tf.layers.conv2d(pool2, 32, 1, activation=tf.nn.relu)
        stack3 = tf.concat([conv3_b1, conv3_b2], 3)
        norm3 = tf.layers.batch_normalization(stack3)

    # [batch, h / 4, w / 4, 160]
    with tf.name_scope('conv_a'):
        conv4_a = tf.layers.conv2d(norm3, 192, 3, activation=tf.nn.relu, padding='same')
        norm4_a = tf.layers.batch_normalization(conv4_a)
        conv4_b = tf.layers.conv2d(norm4_a, 192, 3, activation=tf.nn.relu, padding='same') + norm4_a
        norm4_b = tf.layers.batch_normalization(conv4_b)

    # [batch, h / 4, w / 4, 192] >> (h / 2, w / 2)
    with tf.name_scope('deconv_a'):
        up1_s = tf.shape(conv3_b1)
        up1 = tf.image.resize_nearest_neighbor(norm4_b, [up1_s[1], up1_s[2]])
        conv5_a = tf.layers.conv2d(up1, 64, 3, activation=tf.nn.relu, padding='same')
        norm5_a = tf.layers.batch_normalization(conv5_a)
        conv5_b = tf.layers.conv2d(norm5_a, 64, 3, activation=tf.nn.relu, padding='same')
        norm5_b = tf.layers.batch_normalization(conv5_b)

    # [batch, h / 2, w / 2, 64] >> (h, w)
    with tf.name_scope('deconv_b'):
        up2_s = tf.shape(x_stack)
        up2 = tf.image.resize_nearest_neighbor(norm5_b, [up2_s[1], up2_s[2]])
        conv6_a = tf.layers.conv2d(up2, 32, 3, activation=tf.nn.relu, padding='same')
        norm6_a = tf.layers.batch_normalization(conv6_a)
        conv6_b = tf.layers.conv2d(norm6_a, 32, 3, activation=tf.nn.relu, padding='same')
        conv6_c = tf.layers.conv2d(conv6_b, 32, 3, activation=tf.nn.relu, padding='same')

    y = tf.layers.conv2d(conv6_c, 3, 5, activation=tf.nn.sigmoid, padding='same')
    return y
