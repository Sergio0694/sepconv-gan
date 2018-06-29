import tensorflow as tf
import networks._tf as _tf
from networks.layers.ops import SEPCONV_MODULE

def get_network_v1(x, training):

    def downscale_block(tensor, filters):
        down_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_conv2 = tf.layers.conv2d(down_conv1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_norm = tf.layers.batch_normalization(down_conv2, training=training)
        down_pool = tf.layers.max_pooling2d(down_norm, 2, 2, padding='same')
        return down_conv2, down_pool

    def upscale_block(tensor, residue, filters):
        up_shape = tf.shape(residue)
        up_concat = tf.concat([tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]]), residue], -1)
        up_conv1 = tf.layers.conv2d(up_concat, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_conv2 = tf.layers.conv2d(up_conv1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        return tf.layers.batch_normalization(up_conv2, training=training)

    def subnet(tensor, filters):
        sub_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        sub_conv2 = tf.layers.conv2d(sub_conv1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        sub_norm = tf.layers.batch_normalization(sub_conv2, training=training)
        return tf.layers.conv2d(sub_norm, filters, 3, activation=tf.nn.leaky_relu, padding='same')

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('encoder', None, [x]):
            with tf.variable_scope('downscale_1', None, [x]):
                conv1, pool1 = downscale_block(x, 32)

            with tf.variable_scope('downscale_2', None, [pool1]):
                conv2, pool2 = downscale_block(pool1, 64)

            with tf.variable_scope('downscale_3', None, [pool2]):
                conv3, pool3 = downscale_block(pool2, 128)

            with tf.variable_scope('downscale_4', None, [pool3]):
                conv4, pool4 = downscale_block(pool3, 256)

            with tf.variable_scope('downscale_5', None, [pool4]):
                conv5, pool5 = downscale_block(pool4, 512)

            with tf.variable_scope('hidden_vector', None, [pool5]):
                h_conv1 = tf.layers.conv2d(pool5, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_conv2 = tf.layers.conv2d(h_conv1, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_norm = tf.layers.batch_normalization(h_conv2, training=training)

        with tf.variable_scope('decoder', None, [h_norm, conv5, conv4, conv3, conv2, conv1]):
            with tf.variable_scope('upscale_5', None, [h_norm, conv5]):
                up5 = upscale_block(h_norm, conv5, 512)

            with tf.variable_scope('upscale_4', None, [up5, conv4]):
                up4 = upscale_block(up5, conv4, 256)

            with tf.variable_scope('upscale_3', None, [up4, conv3]):
                up3 = upscale_block(up4, conv3, 128)

            with tf.variable_scope('upscale_2', None, [up3, conv2]):
                up2 = upscale_block(up3, conv2, 64)

            with tf.variable_scope('upscale_1', None, [up2, conv1]):
                up_shape = tf.shape(conv1)
                up_concat = tf.concat([tf.image.resize_bilinear(up2, [up_shape[1], up_shape[2]]), conv1], -1)
        
        with tf.variable_scope('sepconv', None, [up_concat, x]):

            with tf.variable_scope('frame_0', None, [up_concat, x]):
                kv_0 = subnet(up_concat, 31)
                kh_0 = subnet(up_concat, 31)
                frame_0 = SEPCONV_MODULE.sepconv(x[:, :, :, :3], kv_0, kh_0)

            with tf.variable_scope('frame_1', None, [up_concat, x]):
                kv_1 = subnet(up_concat, 31)
                kh_1 = subnet(up_concat, 31)
                frame_1 = SEPCONV_MODULE.sepconv(x[:, :, :, 3:], kv_1, kh_1)
            
            return tf.nn.sigmoid(frame_0 + frame_1)

            