import os
from multiprocessing import cpu_count
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *

def load_train(path, size, window):
    '''Prepares the input pipeline to train the model. Each batch is made up of 
    n frames [-n, ..., -2, -1, +1, +2, ..., +n] and a ground truth frame 
    as the expected value to be generated from the network.

    path(str) -- the directory where the dataset is currently stored
    size(int) -- the batch size for the data pipeline
    window(int) -- the window size
    '''

    assert size > 1     # you don't say?

    _, pipeline = load_core(path, window)
    return pipeline.batch(size).prefetch(1)
        
def load_test(path, window):
    '''Prepares the input pipeline to test the model. Each batch is made up of 
    n frames [-n, ..., -2, -1, +1, +2, ..., +n] and a ground truth frame 
    as the expected value to be generated from the network.

    path(str) -- the directory where the dataset is currently stored
    size(int) -- the batch size for the data pipeline
    window(int) -- the window size
    '''

    n, pipeline = load_core(path, window)
    return pipeline.batch(n)

# ====================
# auxiliary methods
# ====================

def load_core(path, window):
    '''Auxiliary method for the load_train and load_test methods
    '''

    assert window >= 1      # same here

    # load the available frames, group by 5
    files = os.listdir(path)
    groups, labels = [], []
    for i in range(len(files) - window * 2):
        groups += [files[i: i + window] + files[i + window + 1: i + 2 * window + 1]]
        labels += [files[i + window]]

    # create the dataset pipeline
    return len(groups), \
        tf.data.Dataset.from_tensor_slices((groups, labels)) \
        .shuffle(len(groups), reshuffle_each_iteration=True) \
        .filter(lambda x, y: tf.py_func(tf_ensure_same_video_origin, inp=[x], Tout=[tf.bool])) \
        .map(lambda x, y: tf.py_func(tf_load_images, inp=[x, y, path], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .filter(lambda x, y: tf.py_func(ensure_difference_threshold, inp=[x], Tout=[tf.bool])) \
        .repeat()

def tf_ensure_same_video_origin(paths):
    '''Ensures the input frames all belong to the same video section.
    The default frames export format is v{video index}_s{video section}_f{frame number}.{extension}

    paths(list<tf.string) -- a tensor with the input filenames in the current group
    '''
    
    parts1, parts2 = str(paths[0]).split('_'), str(paths[-1]).split('_')
    return parts1[0] == parts2[0] and parts1[1] == parts2[1]

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
