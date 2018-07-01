import tensorflow as tf
import networks._tf as _tf
from networks.layers.ops import SEPCONV_MODULE

def get_network_v1(x, training):

    x_shape = tf.shape(x)

    def block(tensor, filters):
        down_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_conv2 = tf.layers.conv2d(down_conv1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_conv3 = tf.layers.conv2d(down_conv2, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        return down_conv3

    def upsample(tensor, filters, target):
        up_shape = tf.shape(target)
        up = tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]])
        up_conv = tf.layers.conv2d(up, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        return up_conv

    def subnet(tensor):
        sub_conv1 = tf.layers.conv2d(tensor, 64, 3, activation=tf.nn.leaky_relu, padding='same')
        sub_conv2 = tf.layers.conv2d(sub_conv1, 64, 3, activation=tf.nn.leaky_relu, padding='same')
        sub_conv3 = tf.layers.conv2d(sub_conv2, 51, 3, activation=tf.nn.leaky_relu, padding='same')
        sub_up = tf.image.resize_bilinear(sub_conv3, [x_shape[1], x_shape[2]])
        return tf.layers.conv2d(sub_up, 51, 3, activation=tf.nn.leaky_relu, padding='same')

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('encoder', None, [x]):
            with tf.variable_scope('downscale_1', None, [x]):
                conv1 = block(x, 32)
                pool1 = tf.layers.average_pooling2d(conv1, 2, 2, padding='same')

            with tf.variable_scope('downscale_2', None, [pool1]):
                conv2 = block(pool1, 64)
                pool2 = tf.layers.average_pooling2d(conv2, 2, 2, padding='same')

            with tf.variable_scope('downscale_3', None, [pool2]):
                conv3 = block(pool2, 128)
                pool3 = tf.layers.average_pooling2d(conv3, 2, 2, padding='same')

            with tf.variable_scope('downscale_4', None, [pool3]):
                conv4 = block(pool3, 256)
                pool4 = tf.layers.average_pooling2d(conv4, 2, 2, padding='same')

            with tf.variable_scope('downscale_5', None, [pool4]):
                conv5 = block(pool4, 512)
                pool5 = tf.layers.average_pooling2d(conv5, 2, 2, padding='same')

        with tf.variable_scope('decoder', None, [pool5, conv5, conv4, conv3, conv2, conv1]):
            with tf.variable_scope('upscale_5', None, [pool5, conv5]):
                deconv5 = block(pool5, 512)
                up5 = upsample(deconv5, 512, conv5) + conv5

            with tf.variable_scope('upscale_4', None, [up5, conv4]):
                deconv4 = block(up5, 256)
                up4 = upsample(deconv4, 256, conv4) + conv4

            with tf.variable_scope('upscale_3', None, [up4, conv3]):
                deconv3 = block(up4, 128)
                up3 = upsample(deconv3, 128, conv3) + conv3

            with tf.variable_scope('upscale_2', None, [up3, conv2]):
                deconv2 = block(up3, 64)
                up2 = upsample(deconv2, 64, conv2) + conv2
        
        with tf.variable_scope('sepconv', None, [up2, x]):

            with tf.variable_scope('frame_0', None, [up2, x]):
                kv_0 = subnet(up2)
                kh_0 = subnet(up2)
                frame_0 = SEPCONV_MODULE.sepconv(x[:, :, :, :3], kv_0, kh_0)

            with tf.variable_scope('frame_1', None, [up2, x]):
                kv_1 = subnet(up2)
                kh_1 = subnet(up2)
                frame_1 = SEPCONV_MODULE.sepconv(x[:, :, :, 3:], kv_1, kh_1)
            
            return frame_0 + frame_1