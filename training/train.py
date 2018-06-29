from multiprocessing import Process, Queue
import os
from pathlib import Path
from time import time
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO, BAR, RESET_LINE
import networks.discriminators.inception_resnet_mini as inception_mini
import networks._tf as _tf

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'      # See issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1'            # GTX1080 only

def run():

    # cleanup
    LOG('Cleanup')
    leftovers = os.listdir(TENSORBOARD_ROOT_DIR)
    filename = next((x for x in leftovers if x.endswith('.meta')), None)
    if filename is not None:
        cleanup_path = os.path.join(TENSORBOARD_ROOT_DIR, filename[:-5])
        for name in leftovers:
            current_path = os.path.join(TENSORBOARD_ROOT_DIR, name)
            if os.path.isfile(current_path):
                os.rename(current_path, os.path.join(cleanup_path, name))

    # graph setup
    graph = tf.Graph()
    with graph.as_default():

        # initialize the dataset
        LOG('Creating datasets')
        with tf.variable_scope('generator_data'):
            train_dataset = data_loader.load_train(TRAINING_DATASET_PATH, BATCH_SIZE, IMAGES_WINDOW_SIZE)
            test_dataset = data_loader.load_test(TEST_DATASET_PATH, IMAGES_WINDOW_SIZE)
            gen_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            x_train, y = gen_iterator.get_next()
            train_init_op = gen_iterator.make_initializer(train_dataset)
            test_init_op = gen_iterator.make_initializer(test_dataset)
        
        with tf.variable_scope('discriminator_data'):
            disc_dataset = data_loader.load_discriminator_samples(TRAINING_DATASET_PATH, BATCH_SIZE)
            disc_iterator = disc_dataset.make_one_shot_iterator()
            disc_x_next = disc_iterator.get_next()[0]
            x_true = tf.placeholder_with_default(disc_x_next, [None, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3], name='x_true')

        # change this line to choose the model to train
        LOG('Creating model')
        x = tf.placeholder_with_default(
            x_train,
            [None, None, None, INPUT_CHANNELS] if IMAGES_WINDOW_SIZE == 1
            else [None, 3, None, None, INPUT_CHANNELS],
            name='x')
        training = tf.placeholder(tf.bool, name='training_mode')
        with tf.variable_scope('generator', None, [x]):
            raw_yHat = NETWORK_BUILDER(x / 255.0, training)
            yHat = raw_yHat * 255.0

        # discriminator setup
        with tf.variable_scope('discriminator', None, [raw_yHat, x_true], reuse=tf.AUTO_REUSE):
            with tf.name_scope('true', [x_true]):
                disc_true = inception_mini.get_network(x_true / 255.0)
            with tf.name_scope('false', [raw_yHat]):
                raw_yHat.set_shape([None, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3])
                disc_false = inception_mini.get_network(raw_yHat)

        # setup the loss function
        with tf.variable_scope('optimization', None, [yHat, y, disc_true, disc_false]):
            eta = tf.placeholder(tf.float32)

            with tf.variable_scope('generator_opt', None, [yHat, y, disc_false, eta]):
                with tf.variable_scope('generator_loss', None, [yHat, y, disc_false]):
                    gen_own_loss = tf.reduce_mean((yHat - y) ** 2)
                    gen_disc_loss = tf.contrib.gan.losses.wargs.modified_generator_loss(disc_false) # ignored if discriminator is disabled
                    gen_loss = gen_own_loss + gen_disc_loss
                    gen_loss_with_NaN_check = tf.verify_tensor_all_finite(gen_loss if DISCRIMINATOR_ACTIVATION_EPOCH is not None else gen_own_loss, 'NaN found in loss :(', 'NaN_check_output_loss')
                with tf.variable_scope('generator_sgd', None, [gen_loss_with_NaN_check, eta]):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
                        gen_sgd = tf.train.AdamOptimizer(eta) if GENERATOR_ADAM_OPTIMIZER else tf.train.MomentumOptimizer(eta, 0.9, use_nesterov=True)
                        gen_optimizer = gen_sgd.minimize(gen_loss_with_NaN_check, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')) if GENERATOR_GRADIENT_CLIP is None \
                                        else _tf.minimize_with_clipping(gen_sgd, gen_loss_with_NaN_check, GENERATOR_GRADIENT_CLIP, scope='generator')

            with tf.variable_scope('discriminator_opt', None, [disc_true, disc_false, eta]):
                with tf.variable_scope('loss', None, [disc_true, disc_false]):
                    disc_loss = tf.contrib.gan.losses.wargs.modified_discriminator_loss(disc_true, disc_false)
                with tf.variable_scope('discriminator_adam', None, [disc_loss, eta, eta]):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
                        disc_adam = tf.train.AdamOptimizer(DISCRIMINATOR_LR)
                        disc_optimizer = disc_adam.minimize(disc_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')) if DISCRIMINATOR_GRADIENT_CLIP is None \
                                        else _tf.minimize_with_clipping(disc_adam, disc_loss, DISCRIMINATOR_GRADIENT_CLIP, scope='discriminator')

        # output image
        with tf.variable_scope('inference', None, [yHat]):
            yHat_proof = tf.verify_tensor_all_finite(yHat, 'NaN found in output image :(', 'NaN_check_output')
            uint8_img = tf.cast(yHat_proof, tf.uint8, name='uint8_img')
        
        # summaries
        gen_own_loss_summary = tf.summary.scalar('TRAIN_loss', gen_own_loss)
        gen_loss_summary = tf.summary.scalar('TRAIN_full_loss', gen_loss)
        disc_loss_summary = tf.summary.scalar('TRAIN_disc_loss', disc_loss)
        test_loss = tf.placeholder(tf.float32, name='test_loss')
        tf.summary.scalar('TEST_loss', test_loss, ['_'])
        merged_summary_all = tf.summary.merge([gen_own_loss_summary, gen_loss_summary, disc_loss_summary])
        merged_summary_gen = tf.summary.merge([gen_own_loss_summary])

        # model info (while inside the graph)
        INFO('{} generator variable(s)'.format(np.sum([
            np.prod(v.get_shape().as_list()) 
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        ])))
        INFO('{} discriminator variable(s)'.format(np.sum([
            np.prod(v.get_shape().as_list()) 
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        ])))

    LOG('Background queue setup')
    frames_queue = Queue()
    worker = Process(target=save_frame, args=[frames_queue])
    worker.start()

    # train the model
    LOG('Training starting...')
    with tf.Session(graph=graph) as session:
        with tf.summary.FileWriter(TENSORBOARD_RUN_DIR, session.graph) as writer:

            # initialization
            session.run(train_init_op)
            session.run(tf.global_variables_initializer())
            tf.train.Saver().save(session, TENSORBOARD_RUN_DIR) # store the .meta file once
            saver = tf.train.Saver(max_to_keep=MAX_MODELS_TO_KEEP)
            rates = _tf.DecayingRate(INITIAL_GENERATOR_LR, GENERATOR_LR_DECAY_RATE)
            samples, step, ticks_old = 0, 0, 0
            time_start = time()
            lr = rates.get()
            fetches = [gen_optimizer, disc_optimizer] if DISCRIMINATOR_ACTIVATION_EPOCH == 0 else [gen_optimizer]

            while samples < TRAINING_TOTAL_SAMPLES:
                if samples // TENSORBOARD_LOG_INTERVAL > step:
                    step = samples // TENSORBOARD_LOG_INTERVAL

                    # log to tensorboard
                    if disc_optimizer in fetches:
                        _, _, gen_score, gen_full_score, disc_score, summary = session.run(
                            [gen_optimizer, disc_optimizer, gen_own_loss, gen_loss, disc_loss, merged_summary_all],
                            feed_dict={eta: lr, training: True})
                        RESET_LINE(True)
                        LOG('#{}\tgen_own: {:12.04f}, gen_full: {:12.04f}, disc: {:12.04f}'.format(step, gen_score, gen_full_score, disc_score))
                    else:
                        _, gen_score, summary = session.run(
                            [gen_optimizer, gen_own_loss, merged_summary_gen],
                            feed_dict={eta: lr, training: True})
                        RESET_LINE(True)
                        LOG('#{}\tgen_own: {:12.04f}'.format(step, gen_score))
                    writer.add_summary(summary, samples)

                    # save the model
                    saver.save(session, TENSORBOARD_RUN_DIR, global_step=step, write_meta_graph=False)

                    # test the model
                    session.run(test_init_op)
                    test_score, j = 0, 0
                    while True:
                        try:
                            score, prediction = session.run([gen_own_loss, uint8_img], feed_dict={training: False})
                            test_score += score

                            # save the generated images to track progress
                            predictions_dir = os.path.join(TENSORBOARD_RUN_DIR, '_{}'.format(step))
                            Path(predictions_dir).mkdir(exist_ok=True)
                            frames_queue.put((os.path.join(predictions_dir, '{}_yHat.png'.format(j)), prediction[0]))
                            j += 1
                        except tf.errors.OutOfRangeError:
                            break
                    
                    # log the test loss and restore the pipeline
                    test_summary_tensor = graph.get_tensor_by_name('TEST_loss:0')
                    test_summary = session.run(test_summary_tensor, feed_dict={test_loss: test_score})
                    writer.add_summary(test_summary, samples)
                    session.run(train_init_op)

                    # display additional info and progress the learning rate in use
                    INFO('{:12.04f}'.format(test_score))
                    BAR(0, TRAINING_PROGRESS_BAR_LENGTH, ' {:.2f} sample(s)/s'.format(samples / (time() - time_start)))
                    ticks = 0
                    lr = rates.get()
                    if step == DISCRIMINATOR_ACTIVATION_EPOCH:
                        fetches = [gen_optimizer, disc_optimizer]
                else:
                    session.run(fetches, feed_dict={eta: lr, training: True})

                # training progress
                samples += BATCH_SIZE
                mod = samples % TENSORBOARD_LOG_INTERVAL
                ticks = (mod * TRAINING_PROGRESS_BAR_LENGTH) // TENSORBOARD_LOG_INTERVAL
                if ticks > 0 and ticks != ticks_old:
                    ticks_old = ticks
                    BAR(ticks, TRAINING_PROGRESS_BAR_LENGTH, ' {:.2f} sample(s)/s'.format(samples / (time() - time_start)))

    # close queue
    frames_queue.put(None)
    worker.join()
    frames_queue.close()

def save_frame(queue):
    '''Saves a test frame in the background.'''

    while True:
        task = queue.get()
        if task is None:
            break

        # save the new test frame
        cv2.imwrite(task[0], task[1], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

if __name__ == '__main__':
    run()
