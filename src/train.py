from pathlib import Path
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO
from networks import deep_motion_cnn

def run():

    graph = tf.Graph()
    with graph.as_default():

        # initialize the dataset
        LOG('Creating datasets')
        train_dataset = data_loader.load_train(TRAINING_DATASET_PATH, BATCH_SIZE, 1)
        test_dataset = data_loader.load_test(TEST_DATASET_PATH, 1)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        x, yHat = iterator.get_next()
        train_init_op = iterator.make_initializer(train_dataset)
        test_init_op = iterator.make_initializer(test_dataset)

        # change this line to choose the model to train
        LOG('Creating model')
        y = deep_motion_cnn.get_network_v2(x / 255.0) * 255.0

        # setup the loss function
        LOG('Loss setup')
        with tf.name_scope('loss'):
            loss = 0.5 * tf.reduce_sum((y - yHat) ** 2)
        with tf.name_scope('adam'):
            adam = tf.train.AdamOptimizer().minimize(loss)

        # output image
        y_proof = tf.verify_tensor_all_finite(y, 'NaN found :(', 'NaN_check')
        uint8_img = tf.cast(y_proof, tf.uint8, name='yHat')      
        
        # summaries
        tf.summary.scalar('TRAIN_loss', loss)
        merged_summary = tf.summary.merge_all()

        # model info (while inside the graph)
        INFO('{} total variable(s)'.format(np.sum([
            np.prod(v.get_shape().as_list()) 
            for v in tf.trainable_variables()
        ])))

    # train the model
    LOG('Training starting...')
    with tf.Session(graph=graph) as session:
        with tf.summary.FileWriter(TENSORBOARD_RUN_DIR, session.graph) as writer:

            # initialization
            session.run(train_init_op)
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=MAX_MODELS_TO_KEEP)
            
            for i in range(TRAINING_TOTAL_ITERATIONS):
                if i > 0 and i % TENSORBOARD_LOG_INTERVAL == 0:

                    # log to tensorboard
                    _, score, summary = session.run([adam, loss, merged_summary])
                    writer.add_summary(summary, i)
                    INFO('#{}:\t{}'.format(i, score))

                    # save the model
                    saver.save(session, TENSORBOARD_RUN_DIR, global_step=i, write_meta_graph=i == TENSORBOARD_LOG_INTERVAL)

                    # test the model
                    session.run(test_init_op)
                    predictions, ground_truth = session.run([uint8_img, yHat])
                    session.run(train_init_op)

                    # save the generated images to track progress
                    predictions_dir = '{}\\_{}'.format(TENSORBOARD_RUN_DIR, i)
                    Path(predictions_dir).mkdir(exist_ok=True)
                    with open('{}\\yHat[0].txt'.format(predictions_dir), 'w', encoding='utf-8') as test_txt:
                        opt = np.get_printoptions()
                        np.set_printoptions(threshold=np.nan)
                        print(predictions[0], file=test_txt)
                        np.set_printoptions(**opt)
                    for j in range(predictions.shape[0]):
                        cv2.imwrite('{}\\{}_yHat.jpg'.format(predictions_dir, j), predictions[j], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        cv2.imwrite('{}\\{}_gt.jpg'.format(predictions_dir, j), ground_truth[j], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                else:
                    _ = session.run(adam)
