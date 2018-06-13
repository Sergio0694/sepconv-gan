from pathlib import Path
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO, BAR, RESET_LINE
from networks import deep_motion_cnn

os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '6' # Enable all GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'      # See issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# graph setup
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
    y = deep_motion_cnn.get_network_v3(x / 255.0) * 255.0

    # setup the loss function
    LOG('Loss setup')
    with tf.name_scope('loss'):
        loss = 0.5 * tf.reduce_sum((y - yHat) ** 2)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        with tf.name_scope('adam'):
            eta = tf.placeholder(tf.float32)
            adam = tf.train.AdamOptimizer(eta).minimize(loss)

    # output image
    y_proof = tf.verify_tensor_all_finite(y, 'NaN found :(', 'NaN_check')
    uint8_img = tf.cast(y_proof, tf.uint8, name='yHat')      
    
    # summaries
    tf.summary.scalar('TRAIN_loss', loss)
    test_loss = tf.placeholder(tf.float32, name='test_loss')
    tf.summary.scalar('TEST_loss', test_loss, ['_'])
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
        tf.train.Saver().save(session, TENSORBOARD_RUN_DIR) # store the .meta file once
        saver = tf.train.Saver(max_to_keep=MAX_MODELS_TO_KEEP)
        rates = {
            0: 0.00005,
            1: 0.0001,
            2: 0.0002,
            3: 0.0005
        } # to avoid issues with the first iterations exploding
        samples, step, ticks_old = 0, 0, 0

        while samples < TRAINING_TOTAL_SAMPLES:
            lr = rates.get(step, 0.001)
            if samples // TENSORBOARD_LOG_INTERVAL > step:
                step = samples // TENSORBOARD_LOG_INTERVAL

                # log to tensorboard
                _, score, summary = session.run([adam, loss, merged_summary], feed_dict={eta: lr})
                writer.add_summary(summary, samples)
                RESET_LINE()
                LOG('#{}\t{}'.format(step, score))

                # save the model
                saver.save(session, TENSORBOARD_RUN_DIR, global_step=step, write_meta_graph=False)

                # test the model
                session.run(test_init_op)
                test_score, j = 0, 0
                while True:
                    try:
                        score, prediction = session.run([loss, uint8_img])
                        test_score += score

                        # save the generated images to track progress
                        predictions_dir = '{}\\_{}'.format(TENSORBOARD_RUN_DIR, step)
                        Path(predictions_dir).mkdir(exist_ok=True)
                        cv2.imwrite('{}\\{}_yHat.jpg'.format(predictions_dir, j), prediction[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break
                
                # log the test loss and restore the pipeline
                test_summary_tensor = graph.get_tensor_by_name('TEST_loss:0')
                test_summary = session.run(test_summary_tensor, feed_dict={test_loss: test_score})
                writer.add_summary(test_summary, samples)
                session.run(train_init_op)
                INFO('{}'.format(test_score))
            else:
                _ = session.run(adam, feed_dict={eta: lr})

            # training progress
            samples += BATCH_SIZE
            mod = samples % TENSORBOARD_LOG_INTERVAL
            ticks = (mod * TRAINING_PROGRESS_BAR_LENGTH) // TENSORBOARD_LOG_INTERVAL
            if ticks != ticks_old:
                ticks_old = ticks
                BAR(ticks, TRAINING_PROGRESS_BAR_LENGTH)
