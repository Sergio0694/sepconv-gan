import os
from functools import reduce
from multiprocessing import cpu_count
import tensorflow as tf
import numpy as np
import cv2

def load(path, threshold, size):
    '''Prepares the input pipeline to train the model. Each batch is made up of 4 frames [-2, -1, +1, +2]
    and a ground truth frame as the expected value to be generated from the network.

    path(str) -- the directory where the dataset is currently stored
    threshold(int) -- the maximum threshold to exclude frames with too much change
    size(int) -- the batch size for the data pipeline
    '''

    # load the available frames, group by 5
    files = os.listdir(path)
    groups = [
        files[i:i + 5]
        for i in range(len(files) - 4)
    ]

    # create the dataset pipeline
    return \
        tf.data.Dataset.from_tensor_slices(groups) \
        .shuffle(len(groups)) \
        .filter(lambda g: tf.py_func(tf_ensure_same_video_origin, inp=[g], Tout=[tf.bool])) \
        .map(lambda g: tf.py_func(tf_load_images, inp=[g, path], Tout=[tf.float32]), num_parallel_calls=cpu_count()) \
        .filter(lambda g: tf.py_func(ensure_difference_threshold, inp=[g, threshold], Tout=[tf.bool])) \
        .batch(size) \
        .prefetch(1)

# ====================
# auxiliary methods
# ====================

def tf_ensure_same_video_origin(paths):
    '''Ensures the input frames all belong to the same video section.
    The default frames export format is v{video index}_s{video section}_f{frame number}.{extension}

    paths(list<tf.string) -- a tensor with the input filenames in the current group
    '''

    parts1, parts2 = str(paths[0]).split('_'), str(paths[-1]).split('_')
    return parts1[0] == parts2[0] and parts1[1] == parts2[1]

def tf_load_images(filenames, directory):
    '''Loads and returns a list of images from the input list of filenames
    
    filenames(list<tf.string>) -- a tensor with the list of filenames to load
    directory(tf.string) -- the parent directory for the input files
    '''
    
    return np.array([
        cv2.imread('{}\\{}'.format(str(directory)[2:-1], str(filename)[2:-1])).astype(np.float32)
        for filename in filenames
    ], dtype=np.float32, copy=False)

def ensure_difference_threshold(images, threshold):
    '''Computes the mean squared error between a pair of images

    image1(np.array) -- the first loaded image
    image2(np.array) -- the second loaded image
    '''

    size = reduce(lambda x, y: x * y, list(images.shape))
    error = np.sum((images[0] - images[-1]) ** 2, dtype=np.float32) / size
    return error < threshold
