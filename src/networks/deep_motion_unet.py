import tensorflow as tf
import networks.deep_motion_cnn as cnn

def get_network(x):
    '''Generates a CNN to perform image interpolation

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    x_stack = cnn.stack_images(x)
    conv1_a = tf.layers.conv2d(x_stack, 32, 3, activation=tf.nn.leaky_relu, padding='same')
    norm1_a = tf.layers.batch_normalization(conv1_a)
    conv1_b = tf.layers.conv2d(norm1_a, 32, 3, activation=tf.nn.leaky_relu, padding='same')
    norm1_b = tf.layers.batch_normalization(conv1_b)
    pool1 = tf.layers.max_pooling2d(norm1_b, 2, 2, padding='same')

    conv2_a = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.leaky_relu, padding='same')
    norm2_a = tf.layers.batch_normalization(conv2_a)
    conv2_b = tf.layers.conv2d(norm2_a, 64, 3, activation=tf.nn.leaky_relu, padding='same')
    norm2_b = tf.layers.batch_normalization(conv2_b)
    pool2 = tf.layers.max_pooling2d(norm2_b, 2, 2, padding='same')

    conv3_a = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.leaky_relu, padding='same')
    norm3_a = tf.layers.batch_normalization(conv3_a)
    conv3_b = tf.layers.conv2d(norm3_a, 128, 3, activation=tf.nn.leaky_relu, padding='same')
    norm3_b = tf.layers.batch_normalization(conv3_b)
    pool3 = tf.layers.max_pooling2d(norm3_b, 2, 2, padding='same')

    conv4_a = tf.layers.conv2d(pool3, 256, 3, activation=tf.nn.leaky_relu, padding='same')
    norm4_a = tf.layers.batch_normalization(conv4_a)
    conv4_b = tf.layers.conv2d(norm4_a, 256, 3, activation=tf.nn.leaky_relu, padding='same')
    norm4_b = tf.layers.batch_normalization(conv4_b)
    pool4 = tf.layers.max_pooling2d(norm4_b, 2, 2, padding='same')

    conv5_a = tf.layers.conv2d(pool4, 512, 3, activation=tf.nn.leaky_relu, padding='same')
    norm5_a = tf.layers.batch_normalization(conv5_a)
    conv5_b = tf.layers.conv2d(norm5_a, 512, 3, activation=tf.nn.leaky_relu, padding='same')
    norm5_b = tf.layers.batch_normalization(conv5_b)

    up6_s = tf.shape(conv4_b)
    up6 = tf.concat([tf.image.resize_bilinear(norm5_b, [up6_s[1], up6_s[2]]), norm4_b], -1)
    conv6_a = tf.layers.conv2d(up6, 256, 3, activation=tf.nn.leaky_relu, padding='same')
    norm6_a = tf.layers.batch_normalization(conv6_a)
    conv6_b = tf.layers.conv2d(norm6_a, 256, 3, activation=tf.nn.leaky_relu, padding='same')
    norm6_b = tf.layers.batch_normalization(conv6_b)

    up7_s = tf.shape(conv3_b)
    up7 = tf.concat([tf.image.resize_bilinear(norm6_b, [up7_s[1], up7_s[2]]), norm3_b], -1)
    conv7_a = tf.layers.conv2d(up7, 128, 3, activation=tf.nn.leaky_relu, padding='same')
    norm7_a = tf.layers.batch_normalization(conv7_a)
    conv7_b = tf.layers.conv2d(norm7_a, 128, 3, activation=tf.nn.leaky_relu, padding='same')
    norm7_b = tf.layers.batch_normalization(conv7_b)

    up8_s = tf.shape(conv2_b)
    up8 = tf.concat([tf.image.resize_bilinear(norm7_b, [up8_s[1], up8_s[2]]), norm2_b], -1)
    conv8_a = tf.layers.conv2d(up8, 64, 3, activation=tf.nn.leaky_relu, padding='same')
    norm8_a = tf.layers.batch_normalization(conv8_a)
    conv8_b = tf.layers.conv2d(norm8_a, 64, 3, activation=tf.nn.leaky_relu, padding='same')
    norm8_b = tf.layers.batch_normalization(conv8_b)

    up9_s = tf.shape(conv1_b)
    up9 = tf.concat([tf.image.resize_bilinear(norm8_b, [up9_s[1], up9_s[2]]), norm1_b], -1)
    conv9_a = tf.layers.conv2d(up9, 32, 3, activation=tf.nn.leaky_relu, padding='same')
    norm9_a = tf.layers.batch_normalization(conv9_a)
    conv9_b = tf.layers.conv2d(norm9_a, 32, 3, activation=tf.nn.leaky_relu, padding='same')
    norm9_b = tf.layers.batch_normalization(conv9_b)
    return tf.layers.conv2d(norm9_b, 3, 1, activation=tf.nn.sigmoid)
