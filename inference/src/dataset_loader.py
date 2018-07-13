import os
from multiprocessing import cpu_count
import re
import cv2
import tensorflow as tf
import numpy as np

def load_samples(path, window):
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

def setup_pipeline(path):
    '''Prepares the inference pipeline to be used to process the video frames.
    
    path(str) -- the base path of the filenames that will be fed to the pipeline
    '''

    inference_groups = tf.placeholder(tf.string, name='inference_groups')
    return \
        tf.data.Dataset.from_tensor_slices(inference_groups) \
        .map(lambda x: tf.py_func(tf_load_images, inp=[x, path], Tout=[tf.float32]), num_parallel_calls=cpu_count()) \
        .batch(1) \
        .prefetch(1)

# ====================
# auxiliary methods
# ====================

def tf_load_images(samples, directory):
    '''Loads and returns a list of images from the input list of filenames
    
    samples(list<tf.string>) -- a tensor with the list of filenames to load
    directory(tf.string) -- the parent directory for the input files
    '''
    
    # load the frames
    frames = [
        cv2.imread(os.path.join(str(directory)[2:-1], str(sample)[2:-1])).astype(np.float32)
        for sample in samples
    ]

    # TODO: handle preprocessing here (eg. optical flow)
    return np.concatenate(frames, -1)
