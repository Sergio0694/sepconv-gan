import tensorflow as tf

def get_base(x):
    '''Builds the base unet network.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    def downscale_block(tensor, filters):
        down_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_conv2 = tf.layers.conv2d(down_conv1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_pool = tf.layers.max_pooling2d(down_conv2, 2, 2, padding='same')
        return down_conv2, down_pool

    def upscale_block(tensor, residue, filters):
        up_shape = tf.shape(residue)
        up_concat = tf.concat([tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]]), residue], -1)
        up_conv1 = tf.layers.conv2d(up_concat, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_conv2 = tf.layers.conv2d(up_conv1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        return up_conv2

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

        with tf.variable_scope('decoder', None, [h_conv2, conv5, conv4, conv3, conv2, conv1]):
            with tf.variable_scope('upscale_5', None, [h_conv2, conv5]):
                up5 = upscale_block(h_conv2, conv5, 512)

            with tf.variable_scope('upscale_4', None, [up5, conv4]):
                up4 = upscale_block(up5, conv4, 256)

            with tf.variable_scope('upscale_3', None, [up4, conv3]):
                up3 = upscale_block(up4, conv3, 128)

            with tf.variable_scope('upscale_2', None, [up3, conv2]):
                up2 = upscale_block(up3, conv2, 64)

            with tf.variable_scope('upscale_1', None, [up2, conv1]):
                return upscale_block(up2, conv1, 32)

def get_base_bn(x, training):
    '''Builds the base unet network, with batch normalization.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    def downscale_block(tensor, filters):
        down_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_norm1 = tf.layers.batch_normalization(down_conv1, training=training)
        down_conv2 = tf.layers.conv2d(down_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_norm2 = tf.layers.batch_normalization(down_conv2, training=training)
        down_pool = tf.layers.max_pooling2d(down_norm2, 2, 2, padding='same')
        return down_norm2, down_pool

    def upscale_block(tensor, residue, filters):
        up_shape = tf.shape(residue)
        up_concat = tf.concat([tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]]), residue], -1)
        up_conv1 = tf.layers.conv2d(up_concat, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_norm1 = tf.layers.batch_normalization(up_conv1, training=training)
        up_conv2 = tf.layers.conv2d(up_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_norm2 = tf.layers.batch_normalization(up_conv2, training=training)
        return up_norm2

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('encoder', None, [x]):
            with tf.variable_scope('downscale_1', None, [x]):
                norm1, pool1 = downscale_block(x, 32)

            with tf.variable_scope('downscale_2', None, [pool1]):
                norm2, pool2 = downscale_block(pool1, 64)

            with tf.variable_scope('downscale_3', None, [pool2]):
                norm3, pool3 = downscale_block(pool2, 128)

            with tf.variable_scope('downscale_4', None, [pool3]):
                norm4, pool4 = downscale_block(pool3, 256)

            with tf.variable_scope('downscale_5', None, [pool4]):
                norm5, pool5 = downscale_block(pool4, 512)

            with tf.variable_scope('hidden_vector', None, [pool5]):
                h_conv1 = tf.layers.conv2d(pool5, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_norm1 = tf.layers.batch_normalization(h_conv1, training=training)
                h_conv2 = tf.layers.conv2d(h_norm1, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_norm2 = tf.layers.batch_normalization(h_conv2, training=training)

        with tf.variable_scope('decoder', None, [h_norm2, norm5, norm4, norm3, norm2, norm1]):
            with tf.variable_scope('upscale_5', None, [h_norm2, norm5]):
                up5 = upscale_block(h_norm2, norm5, 512)

            with tf.variable_scope('upscale_4', None, [up5, norm4]):
                up4 = upscale_block(up5, norm4, 256)

            with tf.variable_scope('upscale_3', None, [up4, norm3]):
                up3 = upscale_block(up4, norm3, 128)

            with tf.variable_scope('upscale_2', None, [up3, norm2]):
                up2 = upscale_block(up3, norm2, 64)

            with tf.variable_scope('upscale_1', None, [up2, norm1]):
                return upscale_block(up2, norm1, 32)

def get_network_v1(x, training):
    '''Generates a U-Net CNN to perform image interpolation.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    unet = get_base(x)
    return tf.layers.conv2d(unet, 3, 1, activation=tf.nn.sigmoid)

def get_network_v1_with_flow_v1(x, training):
    with tf.variable_scope('flow_unet', None, [x]):
        unet_flow = get_base(x)
        flow = tf.layers.conv2d(unet_flow, 2, 1, activation=tf.nn.sigmoid)
    x_stack = tf.concat([x, flow], -1)
    with tf.variable_scope('rgb_unet', None, [x_stack]): 
        unet_rgb = get_base(x_stack)
        return tf.layers.conv2d(unet_rgb, 3, 1, activation=tf.nn.sigmoid)

def get_network_v1_with_flow_v2(x, training):
    with tf.variable_scope('flow_unet', None, [x]):
        unet_flow = get_base(x)
        flow = tf.layers.conv2d(unet_flow, 4, 1, activation=tf.nn.sigmoid)
    x_stack = tf.concat([x, flow], -1)
    with tf.variable_scope('rgb_unet', None, [x_stack]): 
        unet_rgb = get_base(x_stack)
        return tf.layers.conv2d(unet_rgb, 3, 1, activation=tf.nn.sigmoid)

def get_network_v1_with_flow_v3(x, training):
    def get_flow_unet(frames):
        with tf.variable_scope('flow_unet', None, [frames], reuse=tf.AUTO_REUSE):
            unet_flow = get_base(frames)
            return tf.layers.conv2d(unet_flow, 2, 1, activation=tf.nn.sigmoid)

    with tf.name_scope('flow_forward'):
        unet_flow_f = get_flow_unet(x)
    with tf.name_scope('flow_backwards'):
        x_reversed = tf.concat([x[:, :, :, 3:], x[:, :, :, :3]], -1)
        unet_flow_b = get_flow_unet(x_reversed)
    x_stack = tf.concat([x, unet_flow_f, unet_flow_b], -1)
    with tf.variable_scope('rgb_unet', None, [x_stack]): 
        unet_rgb = get_base(x_stack)
        return tf.layers.conv2d(unet_rgb, 3, 1, activation=tf.nn.sigmoid)

def get_network_v1_with_flow_linear(x, training):
    with tf.variable_scope('flow_unet', None, [x]):
        unet_flow = get_base_bn(x, training)
        flow = tf.layers.conv2d(unet_flow, 2, 1, activation=tf.nn.sigmoid)
    x_stack = tf.concat([x, flow], -1)
    with tf.variable_scope('rgb_unet', None, [x_stack]): 
        unet_rgb = get_base_bn(x_stack, training)
        output = tf.layers.conv2d(unet_rgb, 3, 1, activation=tf.nn.leaky_relu)
        return tf.clip_by_value(output, 0, 255.0) / 255.0

def get_network_v2(x):
    '''An improved version of the network, with Inception-based downscale modules.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    def downscale_block(tensor, filters):
        down_b1_pool = tf.layers.average_pooling2d(tensor, 2, 1, padding='same')
        down_b1_conv = tf.layers.conv2d(down_b1_pool, 16, 1, activation=tf.nn.leaky_relu)
        down_b2_conv1 = tf.layers.conv2d(tensor, 32, 1, activation=tf.nn.leaky_relu)
        down_b2_norm1 = tf.layers.batch_normalization(down_b2_conv1)
        down_b2_conv2 = tf.layers.conv2d(down_b2_norm1, filters // 2, 3, activation=tf.nn.leaky_relu, padding='same')
        down_b3_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_b3_norm1 = tf.layers.batch_normalization(down_b3_conv1)
        down_b3_conv2 = tf.layers.conv2d(down_b3_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_stack = tf.concat([down_b1_conv, down_b2_conv2, down_b3_conv2], -1)
        down_norm = tf.layers.batch_normalization(down_stack)
        down_pool = tf.layers.max_pooling2d(down_stack, 2, 2, padding='same')
        return down_norm, down_pool

    def upscale_block(tensor, residue, filters):
        up_shape = tf.shape(residue)
        up_concat = tf.concat([tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]]), residue], -1)
        up_conv1 = tf.layers.conv2d(up_concat, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_norm1 = tf.layers.batch_normalization(up_conv1)
        up_conv2 = tf.layers.conv2d(up_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_norm2 = tf.layers.batch_normalization(up_conv2)
        return up_norm2

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('encoder', None, [x]):
            with tf.variable_scope('downscale_1', None, [x]):
                norm1, pool1 = downscale_block(x, 48)

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

            return tf.layers.separable_conv2d(up1, 3, 3, activation=tf.nn.sigmoid, padding='same')

def get_network_v2_1(x):
    '''An improved version of the network, with Inception-based downscale/upscale modules.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    def downscale_block(tensor, filters):
        down_b1_pool = tf.layers.average_pooling2d(tensor, 2, 1, padding='same')
        down_b1_conv = tf.layers.conv2d(down_b1_pool, 16, 1, activation=tf.nn.leaky_relu)
        down_b2_conv1 = tf.layers.conv2d(tensor, 32, 1, activation=tf.nn.leaky_relu)
        down_b2_norm1 = tf.layers.batch_normalization(down_b2_conv1)
        down_b2_conv2 = tf.layers.conv2d(down_b2_norm1, filters // 2, 3, activation=tf.nn.leaky_relu, padding='same')
        down_b3_conv1 = tf.layers.conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_b3_norm1 = tf.layers.batch_normalization(down_b3_conv1)
        down_b3_conv2 = tf.layers.conv2d(down_b3_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_stack = tf.concat([down_b1_conv, down_b2_conv2, down_b3_conv2], -1)
        down_norm = tf.layers.batch_normalization(down_stack)
        down_pool = tf.layers.max_pooling2d(down_stack, 2, 2, padding='same')
        return down_norm, down_pool

    def upscale_block(tensor, residue, filters):
        up_shape = tf.shape(residue)
        up_concat = tf.concat([tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]]), residue], -1)
        up_b1_conv = tf.layers.conv2d(up_concat, 16, 1, activation=tf.nn.leaky_relu)
        up_b2_conv1 = tf.layers.conv2d(up_concat, 32, 1, activation=tf.nn.leaky_relu)
        up_b2_norm1 = tf.layers.batch_normalization(up_b2_conv1)
        up_b2_conv2 = tf.layers.conv2d(up_b2_norm1, filters // 4, 3, activation=tf.nn.leaky_relu, padding='same')
        up_b3_conv1 = tf.layers.conv2d(up_concat, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_b3_norm1 = tf.layers.batch_normalization(up_b3_conv1)
        up_b3_conv2 = tf.layers.conv2d(up_b3_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_stack = tf.concat([up_b1_conv, up_b2_conv2, up_b3_conv2], -1)
        up_norm = tf.layers.batch_normalization(up_stack)
        return up_norm

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('encoder', None, [x]):
            with tf.variable_scope('downscale_1', None, [x]):
                norm1, pool1 = downscale_block(x, 32)

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

            return tf.layers.conv2d(up1, 3, 3, activation=tf.nn.sigmoid, padding='same')

def get_network_v3(x):
    '''A variant of the U-Net network, based on separable convolution layers.

    This network takes a [batch, 2, h, w, 3] input tensor, where each input sample is
    made up of two frames, f-1 and f+1.

    x(tf.Tensor<tf.float32>) -- the input frames
    '''

    def downscale_block(tensor, filters):
        down_b1_pool = tf.layers.average_pooling2d(tensor, 2, 1, padding='same')
        down_b1_conv = tf.layers.conv2d(down_b1_pool, filters // 8, 1, activation=tf.nn.leaky_relu)
        down_b1_norm = tf.layers.batch_normalization(down_b1_conv)

        down_b2_conv = tf.layers.conv2d(tensor, filters // 8, 1, activation=tf.nn.leaky_relu)
        down_b2_norm = tf.layers.batch_normalization(down_b2_conv)

        down_b3_conv1 = tf.layers.conv2d(tensor, filters // 8, 1, activation=tf.nn.leaky_relu)
        down_b3_norm1 = tf.layers.batch_normalization(down_b3_conv1)
        down_b3_conv2 = tf.layers.conv2d(down_b3_norm1, filters // 4, 3, 1, activation=tf.nn.leaky_relu, padding='same')
        down_b3_norm2 = tf.layers.batch_normalization(down_b3_conv2)

        down_b4_conv1 = tf.layers.separable_conv2d(tensor, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_b4_norm1 = tf.layers.batch_normalization(down_b4_conv1)
        down_b4_conv2 = tf.layers.separable_conv2d(down_b4_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        down_b4_norm2 = tf.layers.batch_normalization(down_b4_conv2)

        down_stack = tf.concat([down_b1_norm, down_b2_norm, down_b3_norm2, down_b4_norm2], -1)
        down_pool = tf.layers.max_pooling2d(down_stack, 2, 2, padding='same')
        return down_stack, down_pool

    def upscale_block(tensor, residue, filters):
        up_shape = tf.shape(residue)
        up_concat = tf.concat([tf.image.resize_bilinear(tensor, [up_shape[1], up_shape[2]]), residue], -1)

        up_b1_conv = tf.layers.conv2d(up_concat, filters // 16, 1, activation=tf.nn.leaky_relu)
        up_b1_norm = tf.layers.batch_normalization(up_b1_conv)

        up_b2_conv1 = tf.layers.conv2d(up_concat, filters // 16, 1, activation=tf.nn.leaky_relu)
        up_b2_norm1 = tf.layers.batch_normalization(up_b2_conv1)
        up_b2_conv2 = tf.layers.conv2d(up_b2_norm1, filters // 8, 3, 1, activation=tf.nn.leaky_relu, padding='same')
        up_b2_norm2 = tf.layers.batch_normalization(up_b2_conv2)

        up_b3_conv1 = tf.layers.separable_conv2d(up_concat, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_b3_norm1 = tf.layers.batch_normalization(up_b3_conv1)
        up_b3_conv2 = tf.layers.separable_conv2d(up_b3_norm1, filters, 3, activation=tf.nn.leaky_relu, padding='same')
        up_b3_norm2 = tf.layers.batch_normalization(up_b3_conv2)
        
        up_stack = tf.concat([up_b1_norm, up_b2_norm2, up_b3_norm2], -1)
        return up_stack

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('stem', None, [x]):
            stem_conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.leaky_relu, padding='same')
            stem_norm1 = tf.layers.batch_normalization(stem_conv1)
            stem_conv2 = tf.layers.conv2d(stem_norm1, 64, 3, activation=tf.nn.leaky_relu, padding='same')
            stem_norm2 = tf.layers.batch_normalization(stem_conv2)

        with tf.variable_scope('encoder', None, [stem_norm2]):
            with tf.variable_scope('downscale_1', None, [stem_norm2]):
                norm1, pool1 = downscale_block(stem_norm2, 64)

            with tf.variable_scope('downscale_2', None, [pool1]):
                norm2, pool2 = downscale_block(pool1, 128)

            with tf.variable_scope('downscale_3', None, [pool2]):
                norm3, pool3 = downscale_block(pool2, 256)

            with tf.variable_scope('downscale_4', None, [pool3]):
                norm4, pool4 = downscale_block(pool3, 384)

            with tf.variable_scope('hidden_vector', None, [pool4]):
                h_b1_conv = tf.layers.conv2d(pool4, 32, 1, activation=tf.nn.leaky_relu)
                h_b1_norm = tf.layers.batch_normalization(h_b1_conv)

                h_b2_conv1 = tf.layers.conv2d(pool4, 64 // 8, 1, activation=tf.nn.leaky_relu)
                h_b2_norm1 = tf.layers.batch_normalization(h_b2_conv1)
                h_b2_conv2 = tf.layers.conv2d(h_b2_norm1, 128, 3, 1, activation=tf.nn.leaky_relu, padding='same')
                h_b2_norm2 = tf.layers.batch_normalization(h_b2_conv2)

                h_b3_conv1 = tf.layers.separable_conv2d(pool4, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_b3_norm1 = tf.layers.batch_normalization(h_b3_conv1)
                h_b3_conv2 = tf.layers.separable_conv2d(h_b3_norm1, 512, 3, activation=tf.nn.leaky_relu, padding='same')
                h_b3_norm2 = tf.layers.batch_normalization(h_b3_conv2)

                h_stack = tf.concat([h_b1_norm, h_b2_norm2, h_b3_norm2], -1)

        with tf.variable_scope('decoder', None, [h_stack, norm4, norm3, norm2, norm1]):
            with tf.variable_scope('upscale_1', None, [h_stack, norm4]):
                up4 = upscale_block(h_stack, norm4, 256)

            with tf.variable_scope('upscale_2', None, [up4, norm3]):
                up3 = upscale_block(up4, norm3, 128)

            with tf.variable_scope('upscale_3', None, [up3, norm2]):
                up2 = upscale_block(up3, norm2, 64)

            with tf.variable_scope('upscale_4', None, [up2, norm1]):
                up1 = upscale_block(up2, norm1, 32)

        with tf.variable_scope('tail', None, [up1]):
            tail_conv = tf.layers.separable_conv2d(up1, 32, 3, activation=tf.nn.leaky_relu, padding='same')
            tail_norm = tf.layers.batch_normalization(tail_conv)
            return tf.layers.separable_conv2d(tail_norm, 3, 3, activation=tf.nn.sigmoid, padding='same')
