import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
import helpers.debug_tools as debug_tools
from networks import deep_motion_cnn

def run():

    # initialize the dataset
    debug_tools.LOG('Creating iterator')
    pipeline = data_loader.load(TRAINING_DATASET_PATH, BATCH_SIZE, 1)
    iterator = pipeline.make_initializable_iterator()
    x, yHat = iterator.get_next()

    # change this line to choose the model to train
    debug_tools.LOG('Creating model')
    graph, y = deep_motion_cnn.get_network(x)    

    # setup the loss function
    debug_tools.LOG('Loss setup')
    with graph.as_default():
        with tf.name_scope('loss'):
            loss = 0.5 * tf.reduce_sum((y - yHat) ** 2)
        with tf.name_scope('adam'):
            adam = tf.train.AdamOptimizer().minimize(loss)
        
        # model info (while inside the graph)
        debug_tools.INFO('{} total variable(s)'.format(np.sum([
            np.prod(v.get_shape().as_list()) 
            for v in tf.trainable_variables()
        ])))

    # train the model
    debug_tools.LOG('Training starting...')
    with tf.Session(graph=graph) as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer)
        for i in range(TRAINING_TOTAL_ITERATIONS):
            score, _ = sess.run([loss, adam])
            debug_tools.LOG('#{}:\t{}'.format(i, score))
