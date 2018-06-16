import os
from pathlib import Path
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO, BAR, RESET_LINE
import networks.deep_motion_unet as unet
import networks.discriminators.inception_resnet_mini as inception_mini

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'      # See issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# graph setup
graph = tf.Graph()
with graph.as_default():

    # initialize the dataset
    LOG('Creating datasets')
    with tf.variable_scope('generator_data'):
        train_dataset = data_loader.load_train(TRAINING_DATASET_PATH, BATCH_SIZE, 1)
        test_dataset = data_loader.load_test(TEST_DATASET_PATH, 1)
        gen_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        x_train, y = gen_iterator.get_next()
        train_init_op = gen_iterator.make_initializer(train_dataset)
        test_init_op = gen_iterator.make_initializer(test_dataset)
    
    with tf.variable_scope('discriminator_data'):
        disc_dataset = data_loader.load_discriminator_samples(TRAINING_DATASET_PATH, BATCH_SIZE)
        disc_iterator = disc_dataset.make_one_shot_iterator()
        x_true = disc_iterator.get_next()

    # change this line to choose the model to train
    LOG('Creating model')
    x = tf.placeholder_with_default(x_train, [None, None, None, None, 3], name='x')
    with tf.variable_scope('generator', None, [x]):
        raw_yHat = unet.get_network(x / 255.0)
        yHat = raw_yHat * 255.0

    # discriminator setup
    with tf.variable_scope('discriminator', None, [raw_yHat, x_true]):
        keep_prob, disc_true = inception_mini.get_network(x_true / 255.0)
        _, disc_false = inception_mini.get_network(raw_yHat)

    # setup the loss function
    with tf.variable_scope('optimization', None, [yHat, y, disc_true, disc_false]):
        eta = tf.placeholder(tf.float32)

        with tf.variable_scope('generator_opt', None, [yHat, y, disc_false, eta]):
            with tf.variable_scope('generator_loss', None, [yHat, y, disc_false]):
                gen_own_loss = tf.reduce_mean((yHat - y) ** 2)
                gen_disc_loss = -tf.reduce_mean(tf.log(disc_false))
                gen_loss = gen_own_loss + gen_disc_loss
            with tf.variable_scope('generator_adam', None, [gen_loss, eta]):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
                    gen_adam = tf.train.AdamOptimizer(eta)
                    gen_optimizer = gen_adam.minimize(gen_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))

        with tf.variable_scope('discriminator_opt', None, [disc_true, disc_false, eta]):
            with tf.variable_scope('loss', None, [disc_true, disc_false]):
                disc_loss = -tf.reduce_mean(tf.log(disc_true)) + tf.log(1.0 - disc_false)
            with tf.variable_scope('discriminator_adam', None, [disc_loss, eta, eta]):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
                    disc_adam = tf.train.AdamOptimizer(eta)
                    disc_optimizer = disc_adam.minimize(disc_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'))

    # output image
    with tf.variable_scope('inference', None, [yHat]):
        yHat_proof = tf.verify_tensor_all_finite(yHat, 'NaN found :(', 'NaN_check')
        uint8_img = tf.cast(yHat_proof, tf.uint8, name='uint8_img')
    
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
