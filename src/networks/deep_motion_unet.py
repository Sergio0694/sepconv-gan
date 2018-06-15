import tensorflow as tf
import networks.deep_motion_cnn as cnn

def get_network(x):
    '''Generates a CNN to perform image interpolation

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    def downscale_block(tensor, filters):
        down_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_norm1 = tf.layers.batch_normalization(down_conv1)
        down_conv2 = tf.layers.conv2d(down_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_norm2 = tf.layers.batch_normalization(down_conv2)
        down_pool = tf.layers.max_pooling2d(down_norm2, 2, 2, padding='same')
        return down_norm2, down_pool

    def upscale_block(tensor, residue, filters):
        up_shape = tf.shape(residue)
        up_concat = tf.concat([tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]]), residue], -1)
        up_conv1 = tf.layers.conv2d(up_concat, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_norm1 = tf.layers.batch_normalization(up_conv1)
        up_conv2 = tf.layers.conv2d(up_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_norm2 = tf.layers.batch_normalization(up_conv2)
        return up_norm2

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('reshape', None, [x]):
            x_stack = cnn.stack_images(x)

        with tf.variable_scope('encoder', None, [x_stack]):
            with tf.variable_scope('downscale_1', None, [x_stack]):
                norm1, pool1 = downscale_block(x_stack, 32)

            with tf.variable_scope('downscale_2', None, [pool1]):
                norm2, pool2 = downscale_block(pool1, 64)

            with tf.variable_scope('downscale_3', None, [pool2]):
                norm3, pool3 = downscale_block(pool2, 128)

            with tf.variable_scope('downscale_4', None, [pool3]):
                norm4, pool4 = downscale_block(pool3, 256)

            with tf.variable_scope('hidden_vector', None, [pool4]):
                h_conv1 = tf.layers.conv2d(pool4, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_norm1 = tf.layers.batch_normalization(h_conv1)
                h_conv2 = tf.layers.conv2d(h_norm1, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_norm2 = tf.layers.batch_normalization(h_conv2)

        with tf.variable_scope('decoder', None, [h_norm2, norm4, norm3, norm2, norm1]):
            with tf.variable_scope('upscale_1', None, [h_norm2, norm4]):
                up4 = upscale_block(h_norm2, norm4, 256)

            with tf.variable_scope('upscale_2', None, [up4, norm3]):
                up3 = upscale_block(up4, norm3, 128)

            with tf.variable_scope('upscale_3', None, [up3, norm2]):
                up2 = upscale_block(up3, norm2, 64)

            with tf.variable_scope('upscale_4', None, [up2, norm1]):
                up1 = upscale_block(up2, norm1, 32)

            return tf.layers.conv2d(up1, 3, 1, activation=tf.nn.sigmoid)
