from multiprocessing import Process, Queue
import os
from pathlib import Path
from time import time
from shutil import rmtree, copyfile
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO, BAR, RESET_LINE
import networks.discriminators.inception_resnet_mini as inception_mini
import networks.perceptual_loss.vgg19 as vgg19
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
    for subdir in (x for x in leftovers if os.path.isdir(os.path.join(TENSORBOARD_ROOT_DIR, x))):
        if not '_1' in os.listdir(os.path.join(TENSORBOARD_ROOT_DIR, subdir)):
            rmtree(os.path.join(TENSORBOARD_ROOT_DIR, subdir))

    # graph setup
    with tf.Session() as session:

        # initialize the dataset
        LOG('Creating datasets')
        with tf.variable_scope('generator_data'):
            train_dataset = data_loader.load_train(TRAINING_DATASET_PATH, BATCH_SIZE, IMAGES_WINDOW_SIZE)
            test_dataset = data_loader.load_test(TEST_DATASET_PATH, IMAGES_WINDOW_SIZE)
            gen_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            x_train, y = gen_iterator.get_next()
            train_init_op = gen_iterator.make_initializer(train_dataset)
            test_init_op = gen_iterator.make_initializer(test_dataset)

        # generator model
        LOG('Creating model')
        x = tf.placeholder_with_default(
            x_train,
            [None, None, None, INPUT_CHANNELS] if IMAGES_WINDOW_SIZE == 1
            else [None, 3, None, None, INPUT_CHANNELS],
            name='x')
        training = tf.placeholder(tf.bool, name='training_mode')
        with tf.variable_scope('generator', None, [x]):
            yHat = NETWORK_BUILDER(x / 255.0, training)
            clipped_yHat = tf.clip_by_value(yHat, 0.0, 1.0)

        # optional discriminator
        if DISCRIMINATOR_ENABLED:

            # true samples
            with tf.variable_scope('discriminator_data'):
                disc_dataset = data_loader.load_discriminator_samples(TRAINING_DATASET_PATH, BATCH_SIZE)
                disc_iterator = disc_dataset.make_one_shot_iterator()
                disc_x_next = disc_iterator.get_next()[0]
                x_true = tf.placeholder_with_default(disc_x_next, [None, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3], name='x_true')

            # discriminator model
            with tf.variable_scope('discriminator', None, [clipped_yHat, x_true], reuse=tf.AUTO_REUSE):
                with tf.name_scope('true', [x_true]):
                    disc_true = inception_mini.get_network(x_true / 255.0)
                with tf.name_scope('false', [clipped_yHat]):
                    clipped_yHat.set_shape([None, TRAINING_IMAGES_SIZE, TRAINING_IMAGES_SIZE, 3])
                    disc_false = inception_mini.get_network(clipped_yHat)

        # setup the loss function
        generator_loss_inputs = [yHat, y, disc_false] if DISCRIMINATOR_ENABLED else [yHat, y]
        discriminator_loss_inputs = [disc_true, disc_false] if DISCRIMINATOR_ENABLED else []
        with tf.variable_scope('optimization', None, generator_loss_inputs + discriminator_loss_inputs):

            # generator
            with tf.variable_scope('generator_opt', None, generator_loss_inputs):
                with tf.variable_scope('generator_loss', None, generator_loss_inputs):

                    if GENERATOR_LOSS_TYPE == LossType.L1:
                        gen_loss = tf.reduce_mean(tf.abs(yHat * 255.0 - y))
                    elif GENERATOR_LOSS_TYPE == LossType.L2:
                        gen_loss = tf.reduce_mean((yHat * 255.0 - y) ** 2)
                    elif GENERATOR_LOSS_TYPE == LossType.PERCEPTUAL:
                        gen_loss = vgg19.get_loss(yHat * 255.0, y)
                    elif GENERATOR_LOSS_TYPE == LossType.L1_PERCEPTUAL:
                        gen_loss = L_LOSS_FACTOR * tf.reduce_mean(tf.abs(yHat * 255.0 - y)) + PERCEPTUAL_LOSS_FACTOR * vgg19.get_loss(yHat * 255.0, y)
                    elif GENERATOR_LOSS_TYPE == LossType.L2_PERCEPTUAL:
                        gen_loss = L_LOSS_FACTOR * tf.reduce_mean((yHat * 255.0 - y) ** 2) + PERCEPTUAL_LOSS_FACTOR * vgg19.get_loss(yHat * 255.0, y)
                    else:
                        raise ValueError('Invalid loss type')
                    if DISCRIMINATOR_ENABLED:
                        gen_loss = gen_loss + tf.contrib.gan.losses.wargs.modified_generator_loss(disc_false) # ignored if discriminator is disabled
                    gen_loss = tf.verify_tensor_all_finite(gen_loss, 'NaN found in loss :(', 'NaN_check_output_loss')
                
                # optimizer
                eta = tf.placeholder(tf.float32)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
                    gen_sgd = tf.train.AdamOptimizer(eta) if GENERATOR_ADAM_OPTIMIZER else tf.train.MomentumOptimizer(eta, 0.9, use_nesterov=True)
                    gen_optimizer = gen_sgd.minimize(gen_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')) if GENERATOR_GRADIENT_CLIP is None \
                                    else _tf.minimize_with_clipping(gen_sgd, gen_loss, GENERATOR_GRADIENT_CLIP, scope='generator')

            # discriminator, if needed
            if DISCRIMINATOR_ENABLED:
                with tf.variable_scope('discriminator_opt', None, [disc_true, disc_false]):
                    disc_loss = tf.contrib.gan.losses.wargs.modified_discriminator_loss(disc_true, disc_false)
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
                        disc_adam = tf.train.AdamOptimizer(DISCRIMINATOR_LR)
                        disc_optimizer = disc_adam.minimize(disc_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')) if DISCRIMINATOR_GRADIENT_CLIP is None \
                                        else _tf.minimize_with_clipping(disc_adam, disc_loss, DISCRIMINATOR_GRADIENT_CLIP, scope='discriminator')

        # output image
        with tf.variable_scope('inference', None, [clipped_yHat]):
            yHat_proof = tf.verify_tensor_all_finite(clipped_yHat, 'NaN found in output image :(', 'NaN_check_output') * 255.0
            test_clipped_loss = tf.reduce_mean((yHat_proof - y) ** 2)
            uint8_img = tf.cast(yHat_proof, tf.uint8, name='uint8_img')
        
        # summaries
        with tf.name_scope('summaries'):
            gen_loss_summary = tf.summary.scalar('TRAIN_gen_loss', gen_loss)
            if DISCRIMINATOR_ENABLED:
                disc_loss_summary = tf.summary.scalar('TRAIN_disc_loss', disc_loss)
                merged_summary_train = tf.summary.merge([gen_loss_summary, disc_loss_summary])
            else:
                merged_summary_train = tf.summary.merge([gen_loss_summary])
            test_loss = tf.placeholder(tf.float32, name='test_loss')
            test_loss_summary = tf.summary.scalar('TEST_loss', test_loss)            

        # model info (while inside the graph)
        INFO('{} generator variable(s)'.format(np.sum([
            np.prod(v.get_shape().as_list()) 
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        ])))
        if DISCRIMINATOR_ENABLED:
            INFO('{} discriminator variable(s)'.format(np.sum([
                np.prod(v.get_shape().as_list()) 
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
            ])))
        if PERCEPTUAL_LOSS_ENABLED:
            INFO('{} perceptual model variable(s)'.format(np.sum([
                np.prod(v.get_shape().as_list()) 
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimization/generator_opt/generator_loss')
            ])))

        LOG('Background queue setup')
        frames_queue = Queue()
        worker = Process(target=save_frame, args=[frames_queue])
        worker.start()

        # train the model
        LOG('Training initialization')
        with tf.summary.FileWriter(TENSORBOARD_RUN_DIR, session.graph) as writer:

            # initialization
            session.run(train_init_op)
            _tf.initialize_variables(session)
            tf.train.Saver().save(session, TENSORBOARD_RUN_DIR) # store the .meta file once
            copyfile(MACRO_PATH, os.path.join(TENSORBOARD_RUN_DIR, os.path.basename(MACRO_PATH))) # copy the __MACRO__.py file
            saver = tf.train.Saver(max_to_keep=MAX_MODELS_TO_KEEP)
            rates = _tf.DecayingRate(INITIAL_GENERATOR_LR, GENERATOR_LR_DECAY_RATE)
            samples, step, ticks_old = 0, 0, 0
            time_start = time()
            lr = rates.get()
            fetches = [gen_optimizer, disc_optimizer] if DISCRIMINATOR_ENABLED else [gen_optimizer]

            LOG('Training started...') 
            while samples < TRAINING_TOTAL_SAMPLES:
                if samples // TENSORBOARD_LOG_INTERVAL > step:
                    step = samples // TENSORBOARD_LOG_INTERVAL

                    # log to tensorboard
                    _, summary, gen_score = session.run(
                        fetches + [merged_summary_train, gen_loss],
                        feed_dict={eta: lr, training: True})
                    RESET_LINE(True)
                    LOG('#{}\tgen_loss: {:12.04f}'.format(step, gen_score))
                    writer.add_summary(summary, samples)

                    # save the model
                    saver.save(session, TENSORBOARD_RUN_DIR, global_step=step, write_meta_graph=False)

                    # test the model
                    session.run(test_init_op)
                    test_score, j = 0, 0
                    while True:
                        try:
                            score, prediction = session.run([test_clipped_loss, uint8_img], feed_dict={training: False})
                            test_score += score

                            # save the generated images to track progress
                            predictions_dir = os.path.join(TENSORBOARD_RUN_DIR, '_{}'.format(step))
                            Path(predictions_dir).mkdir(exist_ok=True)
                            frames_queue.put((os.path.join(predictions_dir, '{}_yHat.png'.format(j)), prediction[0]))
                            j += 1
                        except tf.errors.OutOfRangeError:
                            break
                    
                    # log the test loss and restore the pipeline
                    test_summary = session.run(test_loss_summary, feed_dict={test_loss: test_score})
                    writer.add_summary(test_summary, samples)
                    session.run(train_init_op)

                    # display additional info and progress the learning rate in use
                    INFO('{:12.04f}'.format(test_score))
                    BAR(0, TRAINING_PROGRESS_BAR_LENGTH, ' {:.2f} sample(s)/s'.format(samples / (time() - time_start)))
                    ticks = 0
                    lr = rates.get()
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
