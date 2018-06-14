import os
from multiprocessing import cpu_count
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
        .filter(lambda x, y: tf.py_func(ensure_difference_threshold, inp=[x], Tout=[tf.bool])) \
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
    return [
        [files[i:i + window * 2 - 1]]
        for i in range(len(files) - window)
    ]

# ====================
# auxiliary methods
# ====================

def calculate_samples_data(path, window):
    '''Calculates the dataset contents for the input path and window size.
    Returns the list of available files, the sample groups and the label paths.

    path(str) -- the directory where the dataset is currently stored
    window(int) -- the window size
    '''

    files = os.listdir(path)
    groups, labels = [], []
    for i in range(len(files) - window * 2):
        candidates = files[i: i + window] + files[i + window + 1: i + 2 * window + 1]
        info1, info2 = candidates[0].split('_'), candidates[-1].split('_')
        if info1[0] == info2[0] and info1[1] == info2[1]:
            groups += [candidates]
            labels += [files[i + window]]
    return files, groups, labels

def load_core(path, window):
    '''Auxiliary method for the load_train and load_test methods'''

    assert window >= 1      # same here

    # load the available frames, group by the requested window size
    files, groups, labels = calculate_samples_data(path, window)
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

def ensure_difference_threshold(images):
    '''Computes the mean squared error between a series of images

    images(np.array) -- the input images
    threshold(int) -- the maximum squared difference between the first and last image
    '''
    
    size = np.prod(images[0].shape)
    error = np.sum((images[0] - images[-1]) ** 2, dtype=np.float32) / size
    return IMAGE_DIFF_MIN_THRESHOLD < error < IMAGE_DIFF_MAX_THRESHOLD
