import os
from multiprocessing import cpu_count
import re
from random import randint
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
from helpers.logger import LOG, INFO
from helpers.debug_tools import calculate_image_difference

def load_train(path, size, window):
    '''Prepares the input pipeline to train the model. Each batch is made up of 
    n frames [-n, ..., -2, -1, +1, +2, ..., +n] and a ground truth frame 
    as the expected value to be generated from the network.

    path(str) -- the directory where the dataset is currently stored
    size(int) -- the batch size for the data pipeline
    window(int) -- the window size
    '''

    assert size > 1     # you don't say?

    groups, _, pipeline = load_core(path, window)
    return pipeline \
        .shuffle(len(groups), reshuffle_each_iteration=True) \
        .map(lambda x, y: tf.py_func(tf_load_images, inp=[x, y, path], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .filter(lambda x, y: tf.py_func(ensure_difference_threshold, inp=[x, y], Tout=[tf.bool])) \
        .map(lambda x, y: tf.py_func(tf_preprocess_train_images, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .repeat() \
        .batch(size) \
        .prefetch(1)
        
def load_test(path, window):
    '''Prepares the input pipeline to test the model. Each sample is made up of 
    n frames [-n, ..., -2, -1, +1, +2, ..., +n] and a ground truth frame 
    as the expected value to be generated from the network.

    path(str) -- the directory where the dataset is currently stored
    window(int) -- the window size
    '''

    groups, labels, pipeline = load_core(path, window)

    if SHOW_TEST_SAMPLES_INFO_ON_LOAD:
        for s in zip(groups, labels):
            seq = s[0][:len(s[0]) // 2] + [s[1]] + s[0][len(s[0]) // 2:]
            errors = [
                int(calculate_image_difference('{}\\{}'.format(path, pair[0]), '{}\\{}'.format(path, pair[1]))[2])
                for pair in zip(seq, seq[1:])
            ]
            INFO('{} ---> {}, e={}'.format(s[0], s[1], errors))

    return pipeline \
        .map(lambda x, y: tf.py_func(tf_load_images, inp=[x, y, path], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .batch(1) \
        .prefetch(1) # only process one sample at a time to avoid OOM issues in inference

def load_inference_samples(path, window):
    '''Loads the inference samples from the input path.

    path(str) -- the directory where the dataset is currently stored
    window(int) -- the window size
    '''

    files = os.listdir(path)
    files.sort(key=lambda name: int(re.findall('([0-9]+)[.]', name)[0]))
    return [
        files[i:i + window * 2]
        for i in range(len(files) - window)
    ]

# ====================
# auxiliary methods
# ====================

def load_core(path, window):
    '''Auxiliary method for the load_train and load_test methods'''

    assert window >= 1      # same here

    # load the available frames, group by the requested window size
    files = os.listdir(path)
    groups, labels = [], []
    for i in range(len(files) - window * 2):
        candidates = files[i: i + window] + files[i + window + 1: i + 2 * window + 1]
        info1, info2 = candidates[0].split('_'), candidates[-1].split('_')
        if info1[0] == info2[0] and info1[1] == info2[1]:
            groups += [candidates]
            labels += [files[i + window]]
    if VERBOSE_MODE:
        LOG('{} total dataset file(s)'.format(len(files)))
        INFO('{} generated sample(s)'.format(len(groups)))

    # create the dataset pipeline
    return groups, labels, tf.data.Dataset.from_tensor_slices((groups, labels))

def tf_load_images(samples, label, directory):
    '''Loads and returns a list of images from the input list of filenames
    
    samples(list<tf.string>) -- a tensor with the list of filenames to load
    label(tf.string) -- a tensor with the filename of the ground truth image
    directory(tf.string) -- the parent directory for the input files
    '''
    
    x = np.array([
        cv2.imread('{}\\{}'.format(str(directory)[2:-1], str(sample)[2:-1])).astype(np.float32)
        for sample in samples
    ], dtype=np.float32, copy=False)
    y = cv2.imread('{}\\{}'.format(str(directory)[2:-1], str(label)[2:-1])).astype(np.float32)

    return x, y

def tf_preprocess_train_images(samples, label):
    '''Clips the sample images and adds random perturbations to augment the dataset and increase variance.
    
    samples(list<tf.string>) -- a tensor with the list of filenames to load
    label(tf.string) -- a tensor with the filename of the ground truth image
    '''
    
    assert label.shape[0] >= TRAINING_IMAGES_SIZE and label.shape[1] >= TRAINING_IMAGES_SIZE

    # setup
    max_flow_x = min(MAX_FLOW, label.shape[1] - TRAINING_IMAGES_SIZE)
    max_flow_y = min(MAX_FLOW, label.shape[0] - TRAINING_IMAGES_SIZE)
    x_offset = randint(max_flow_x, label.shape[1] - TRAINING_IMAGES_SIZE - max_flow_x)
    y_offset = randint(max_flow_y, label.shape[0] - TRAINING_IMAGES_SIZE - max_flow_y)
    x_flow = randint(0, max_flow_x)
    y_flow = randint(0, max_flow_y)

    # clip to square, add random flow
    if samples.shape[0] == 2:
        x = np.array([
            samples[0][
                y_offset - y_flow:y_offset + TRAINING_IMAGES_SIZE - y_flow, \
                x_offset - x_flow:x_offset + TRAINING_IMAGES_SIZE - x_flow, :
            ],
            samples[1][
                y_offset + y_flow:y_offset + TRAINING_IMAGES_SIZE + y_flow, \
                x_offset + x_flow:x_offset + TRAINING_IMAGES_SIZE + x_flow, :
            ]
        ], dtype=np.float32, copy=False)
    else:
        raise NotImplementedError('Unsupported windows size')
    
    # same slicing for the label, regardless of the window size
    y = label[y_offset:y_offset + TRAINING_IMAGES_SIZE, x_offset:x_offset + TRAINING_IMAGES_SIZE, :]

    # randomly reverse the frames order
    if randint(0, 1) == 0:
        x = np.flip(x, 0)

    return x, y

def ensure_difference_threshold(samples, label):
    '''Computes the mean squared error between a series of images and returns whether
    or not all the errors are in the expected interval.

    images(np.array) -- the input images
    threshold(int) -- the maximum squared difference between the first and last image
    '''
    
    # prepare the temporary list of all the sample images
    size = np.prod(samples[0].shape)
    if samples.shape[0] == 2:
        images = [samples[0]] + [label] + [samples[1]]
    else:
        raise NotImplementedError('Unsupported windows size')

    # compute the interval errors
    errors = [
        np.sum((pair[0] - pair[-1]) ** 2, dtype=np.float32) / size
        for pair in zip(images, images[1:])
    ]
    return all([IMAGE_DIFF_MIN_THRESHOLD < error < IMAGE_DIFF_MAX_THRESHOLD for error in errors])
