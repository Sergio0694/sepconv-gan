import tensorflow as tf

def get_network(x):
    with tf.variable_scope('discriminator', None, [x], reuse=tf.AUTO_REUSE):
        with tf.variable_scope('stem', None, [x]):

            # [299, ...]
            with tf.variable_scope('downscale_1', None, [x]):
                stem1_conv1 = tf.layers.conv2d(x, 32, 3, 2, activation=tf.nn.leaky_relu, padding='same')
                stem1_conv2 = tf.layers.conv2d(stem1_conv1, 32, 3, activation=tf.nn.leaky_relu, padding='same')
                stem1_conv3 = tf.layers.conv2d(stem1_conv2, 64, 3, activation=tf.nn.leaky_relu, padding='same')
                stem1_pool = tf.layers.max_pooling2d(stem1_conv3, 3, 2, padding='valid')
                stem1_conv4 = tf.layers.conv2d(stem1_conv3, 96, 3, 2, padding='valid')
                stem1_stack = tf.concat([stem1_pool, stem1_conv4], -1)

            # [73, ...]
            with tf.variable_scope('downscale_2', None, [stem1_stack]):
                stem2_b1_conv1 = tf.layers.conv2d(stem1_stack, 64, 1, activation=tf.nn.leaky_relu, padding='same')
                stem2_b1_conv2 = tf.layers.conv2d(stem2_b1_conv1, 96, 3, activation=tf.nn.leaky_relu, padding='valid')
                stem2_b2_conv1 = tf.layers.conv2d(stem1_stack, 64, 1, activation=tf.nn.leaky_relu, padding='same')
                stem2_b2_conv2 = tf.layers.conv2d(stem2_b2_conv1, 64, [7, 1], activation=tf.nn.leaky_relu, padding='same')
                stem2_b2_conv3 = tf.layers.conv2d(stem2_b2_conv2, 64, [1, 7], activation=tf.nn.leaky_relu, padding='same')
                stem2_b2_conv4 = tf.layers.conv2d(stem2_b2_conv3, 96, 3, activation=tf.nn.leaky_relu, padding='valid')
                stem2_stack = tf.concat([stem2_b1_conv2, stem2_b2_conv4], -1)

            # [71, ...]
            with tf.variable_scope('downscale_3', None, [stem2_stack]):
                stem3_b1_conv = tf.layers.conv2d(stem2_stack, 192, 3, 2, padding='valid')
                stem3_b2_pool = tf.layers.max_pooling2d(stem2_stack, 2, 2, padding='valid')
                stem3_stack = tf.concat([stem3_b1_conv, stem3_b2_pool], -1)
        
        with tf.variable_scope('inception', None, [stem3_stack]):

            # [35, ...]
            with tf.variable_scope('block_a', None, [stem3_stack]):
                block_a_out = block_a(stem3_stack)

            with tf.variable_scope('reduction_a', None, [block_a_out]):
                reduction_a_out = reduction_a(block_a_out)

            # [17, ...]
            with tf.variable_scope('reduction_b', None, [reduction_a_out]):
                reduction_b_out = reduction_b(reduction_a_out)
            
            # [8, ...]
            with tf.variable_scope('squeeze', None, [reduction_b_out]):
                reduction_shape = tf.shape(reduction_b_out)
                average_pool = tf.layers.average_pooling2d(reduction_b_out, reduction_shape[1], 1)
                squeeze = tf.squeeze(average_pool)
                dropout_drop_prob = tf.placeholder(tf.bool, 'dropout_drop_prob')
                dropout = tf.layers.dropout(squeeze, dropout_drop_prob)

            # [n, ...]
            with tf.variable_scope('output', None, [dropout]):
                output = tf.layers.dense(dropout, 1, tf.nn.sigmoid)

    return dropout_drop_prob, output

def block_a(tensor, scale=0.8):
    block_a_b1_conv = tf.layers.conv2d(tensor, 32, 1, padding='same')
    block_a_b2_conv1 = tf.layers.conv2d(tensor, 32, 1, padding='same')
    block_a_b2_conv2 = tf.layers.conv2d(block_a_b2_conv1, 32, 3, padding='same')
    block_a_b3_conv1 = tf.layers.conv2d(tensor, 32, 1, padding='same')
    block_a_b3_conv2 = tf.layers.conv2d(block_a_b3_conv1, 32, 3, padding='same')
    block_a_b3_conv3 = tf.layers.conv2d(block_a_b3_conv2, 32, 3, padding='same')
    block_a_stack = tf.concat([block_a_b1_conv, block_a_b2_conv2, block_a_b3_conv3], -1)
    block_a_conv = tf.layers.conv2d(block_a_stack, 256, 1, padding='same')
    return block_a_conv * scale

def reduction_a(tensor):
    reduction_a_b1_pool = tf.layers.max_pooling2d(tensor, 3, 2, padding='valid')
    reduction_a_b2_conv = tf.layers.conv2d(tensor, 192, 3, 2, padding='valid')
    reduction_a_b3_conv1 = tf.layers.conv2d(tensor, 64, 1)
    reduction_a_b3_conv2 = tf.layers.conv2d(reduction_a_b3_conv1, 192, 3, padding='same')
    reduction_a_b3_conv3 = tf.layers.conv2d(reduction_a_b3_conv2, 192, 3, 2, padding='valid')
    reduction_a_stack = tf.concat([reduction_a_b1_pool, reduction_a_b2_conv, reduction_a_b3_conv3], -1)
    return reduction_a_stack

def reduction_b(tensor):
    reduction_b_b1_pool = tf.layers.max_pooling2d(tensor, 3, 2, padding='valid')
    reduction_b_b2_conv1 = tf.layers.conv2d(tensor, 256, 1)
    reduction_b_b2_conv2 = tf.layers.conv2d(reduction_b_b2_conv1, 320, 3, 2, padding='valid')
    reduction_b_b3_conv1 = tf.layers.conv2d(tensor, 256, 1)
    reduction_b_b3_conv2 = tf.layers.conv2d(reduction_b_b3_conv1, 256, 3, 2, padding='valid')
    reduction_b_b4_conv1 = tf.layers.conv2d(tensor, 256, 1)
    reduction_b_b4_conv2 = tf.layers.conv2d(reduction_b_b4_conv1, 256, 3, padding='same')
    reduction_b_b4_conv3 = tf.layers.conv2d(reduction_b_b4_conv2, 256, 3, 2, padding='valid')
    reduction_b_stack = tf.concat([reduction_b_b1_pool, reduction_b_b2_conv2, reduction_b_b3_conv2, reduction_b_b4_conv3], 1)
    return reduction_b_stack
