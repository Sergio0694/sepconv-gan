import tensorflow as tf

def stack_images(x):
    '''Stacks n images over the channels dimension.

    x(tf.tensor<tf.float32>) -- the input [batch, images, h, w, channels] tensor
    '''

    with tf.name_scope('frames'):
        x_t = tf.transpose(x, [0, 2, 3, 1, 4])
        x_shape = tf.shape(x_t, name='batch_shape')
        return tf.reshape(x_t, [x_shape[0], x_shape[1], x_shape[2], 6], name='frames')