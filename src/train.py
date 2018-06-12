import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
import helpers.debug_tools as debug_tools
from networks import deep_motion_cnn

def run():

    graph = tf.Graph()
    with graph.as_default():

        # initialize the dataset
        debug_tools.LOG('Creating iterator')
        pipeline = data_loader.load(TRAINING_DATASET_PATH, BATCH_SIZE, 1)
        iterator = pipeline.make_initializable_iterator()
        x, yHat = iterator.get_next()

        # change this line to choose the model to train
        debug_tools.LOG('Creating model')
        y = deep_motion_cnn.get_network(x)    

        # setup the loss function
        debug_tools.LOG('Loss setup')
        with tf.name_scope('loss'):
            loss = 0.5 * tf.reduce_sum((y - yHat) ** 2)
        with tf.name_scope('adam'):
            adam = tf.train.AdamOptimizer().minimize(loss)
        
        # summaries
        tf.summary.scalar('TRAIN_loss', loss)
        merged_summary = tf.summary.merge_all()

        # model info (while inside the graph)
        debug_tools.INFO('{} total variable(s)'.format(np.sum([
            np.prod(v.get_shape().as_list()) 
            for v in tf.trainable_variables()
        ])))

    # train the model
    debug_tools.LOG('Training starting...')
    model_dir = '{}\\{}'.format(TENSORBOARD_DIR, MODEL_ID)
    with tf.Session(graph=graph) as session:
        with tf.summary.FileWriter(model_dir, session.graph) as writer:

            # initialization
            session.run(iterator.initializer)
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=MAX_MODELS_TO_KEEP)
            
            for i in range(TRAINING_TOTAL_ITERATIONS):
                if i > 0 and i % TENSORBOARD_LOG_INTERVAL == 0:

                    # log to tensorboard
                    _, summary = session.run([adam, merged_summary])
                    writer.add_summary(summary, i)

                    # save the model
                    saver.save(session, model_dir, global_step=i, write_meta_graph=i == TENSORBOARD_LOG_INTERVAL)
                else:
                    _ = session.run(adam)
