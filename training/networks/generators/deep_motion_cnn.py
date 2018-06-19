import tensorflow as tf
import networks._tf as _tf

def get_network_v1(x):
    '''Generates a simple CNN to perform image interpolation, based on the FI_CNN_model
    network defined here: https://github.com/neil454/deep-motion/blob/master/src/FI_CNN.py#L6.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    with tf.variable_scope('CNN_v1', None, [x]):
        with tf.variable_scope('encoder', None, [x]):
            conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
            pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')

            conv2 = tf.layers.conv2d(pool1, 32, 3, activation=tf.nn.relu, padding='same')
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')
            conv3 = tf.layers.conv2d(pool2, 32, 3, activation=tf.nn.relu, padding='same')
        
        with tf.variable_scope('decoder', None, [conv3, conv2, conv1]):
            up1_s = tf.shape(conv2)
            up1 = tf.image.resize_nearest_neighbor(conv3, [up1_s[1], up1_s[2]])
            conv4 = tf.layers.conv2d(up1, 32, 3, activation=tf.nn.relu, padding='same')

            up2_s = tf.shape(conv1)
            up2 = tf.image.resize_nearest_neighbor(conv4, [up2_s[1], up2_s[2]])
            return tf.layers.conv2d(up2, 3, 3, activation=tf.nn.sigmoid, padding='same')

def get_network_v2(x):
    '''Generates a variant of the simple CNN to perform image interpolation.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    with tf.variable_scope('CNN_v2', None, [x]):
        with tf.variable_scope('encoder', None, [x]):
            conv1_a = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
            conv1_b = tf.layers.conv2d(conv1_a, 32, 3, activation=tf.nn.relu, padding='same')
            pool1 = tf.layers.max_pooling2d(conv1_b, 2, 2, padding='same')

            conv2_a = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same')
            conv2_b = tf.layers.conv2d(conv2_a, 64, 3, activation=tf.nn.relu, padding='same')
            pool2 = tf.layers.max_pooling2d(conv2_b, 2, 2, padding='same')

            conv3_a = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu, padding='same')
            conv3_b = tf.layers.conv2d(conv3_a, 128, 3, activation=tf.nn.relu, padding='same') + conv3_a
        
        with tf.variable_scope('decoder', None, [conv3_b, conv2_b, conv1_b]):
            up1_s = tf.shape(conv2_b)
            up1 = tf.image.resize_nearest_neighbor(conv3_b, [up1_s[1], up1_s[2]])

            conv4_a = tf.layers.conv2d(up1, 64, 3, activation=tf.nn.relu, padding='same') + conv2_b
            conv4_b = tf.layers.conv2d(conv4_a, 64, 3, activation=tf.nn.relu, padding='same')
            up2_s = tf.shape(conv1_b)
            up2 = tf.image.resize_nearest_neighbor(conv4_b, [up2_s[1], up2_s[2]])

            conv5_a = tf.layers.conv2d(up2, 32, 3, activation=tf.nn.relu, padding='same') + conv1_b
            conv5_b = tf.layers.conv2d(conv5_a, 32, 3, activation=tf.nn.relu, padding='same')

            return tf.layers.conv2d(conv5_b, 3, 5, activation=tf.nn.sigmoid, padding='same')

def get_network_v3(x):
    '''Generates a deeper variant of the simple CNN to perform image interpolation.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    with tf.variable_scope('CNN_v3', None, [x]):
        with tf.variable_scope('encoder', None, [x]):

            # [batch, h, w, 3 * images]
            conv1_a = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
            conv1_b = tf.layers.conv2d(conv1_a, 32, 3, activation=tf.nn.relu, padding='same')
            conv1_c = tf.layers.conv2d(conv1_b, 32, 3, activation=tf.nn.relu, padding='same')
            norm1_a = tf.layers.batch_normalization(conv1_c)
            conv1_d = tf.layers.conv2d(norm1_a, 32, 2, 2, activation=tf.nn.relu, padding='same')
            pool1 = tf.layers.max_pooling2d(norm1_a, 2, 2, padding='same')
            stack1 = tf.concat([conv1_d, pool1], 3)
            norm1_b = tf.layers.batch_normalization(stack1)

            # [batch, h / 2, w / 2, 64]
            conv2_b1_a = tf.layers.conv2d(norm1_b, 64, 1, activation=tf.nn.relu)
            conv2_b1_b = tf.layers.conv2d(conv2_b1_a, 64, 3, activation=tf.nn.relu, padding='same')
            conv2_b1_c = tf.layers.conv2d(conv2_b1_b, 46, 1, activation=tf.nn.relu)
            conv2_b2_a = tf.layers.conv2d(norm1_b, 64, 1, activation=tf.nn.relu)
            conv2_b2_b = tf.layers.conv2d(conv2_b2_a, 64, [7, 1], activation=tf.nn.relu, padding='same')
            conv2_b2_c = tf.layers.conv2d(conv2_b2_b, 64, [1, 7], activation=tf.nn.relu, padding='same')
            conv2_b2_d = tf.layers.conv2d(conv2_b2_c, 46, 3, activation=tf.nn.relu, padding='same')
            stack2 = tf.concat([conv2_b1_c, conv2_b2_d], 3)
            norm2 = tf.layers.batch_normalization(stack2)

            # [batch, h / 2, w / 2, 92]
            conv3_b1 = tf.layers.conv2d(norm2, 128, 3, 2, activation=tf.nn.relu, padding='same')
            pool3 = tf.layers.max_pooling2d(norm2, 2, 2, padding='same')
            conv3_b2 = tf.layers.conv2d(pool3, 32, 1, activation=tf.nn.relu)
            stack3 = tf.concat([conv3_b1, conv3_b2], 3)
            norm3 = tf.layers.batch_normalization(stack3)

            # [batch, h / 4, w / 4, 160]
            conv4_a = tf.layers.conv2d(norm3, 192, 3, activation=tf.nn.relu, padding='same')
            norm4_a = tf.layers.batch_normalization(conv4_a)
            conv4_b = tf.layers.conv2d(norm4_a, 192, 3, activation=tf.nn.relu, padding='same') + norm4_a
            norm4_b = tf.layers.batch_normalization(conv4_b)
            conv4_c = tf.layers.conv2d(norm4_b, 192, 3, activation=tf.nn.relu, padding='same') + norm4_b
            norm4_c = tf.layers.batch_normalization(conv4_c)

        with tf.variable_scope('decoder', None, [norm2, norm4_c, norm1_a]):

            # [batch, h / 4, w / 4, 192] >> (h / 2, w / 2)
            up1_s = tf.shape(norm2)
            up1 = tf.image.resize_bilinear(norm4_c, [up1_s[1], up1_s[2]])
            norm2_res = tf.layers.conv2d(norm2, 64, 1, activation=tf.nn.relu)
            conv5_a = tf.layers.conv2d(up1, 64, 3, activation=tf.nn.relu, padding='same') + norm2_res
            norm5_a = tf.layers.batch_normalization(conv5_a)
            conv5_b = tf.layers.conv2d(norm5_a, 64, 3, activation=tf.nn.relu, padding='same')
            norm5_b = tf.layers.batch_normalization(conv5_b)

            # [batch, h / 2, w / 2, 64] >> (h, w)
            up2_s = tf.shape(x)
            up2 = tf.image.resize_bilinear(norm5_b, [up2_s[1], up2_s[2]])
            conv6_a = tf.layers.conv2d(up2, 32, 3, activation=tf.nn.relu, padding='same') + norm1_a
            norm6_a = tf.layers.batch_normalization(conv6_a)
            conv6_b = tf.layers.conv2d(norm6_a, 32, 3, activation=tf.nn.relu, padding='same')
            conv6_c = tf.layers.conv2d(conv6_b, 32, 3, activation=tf.nn.relu, padding='same')

            return tf.layers.conv2d(conv6_c, 3, 5, activation=tf.nn.sigmoid, padding='same')
