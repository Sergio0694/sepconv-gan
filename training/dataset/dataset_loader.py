import os
from multiprocessing import cpu_count
from random import randint
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import (
    VERBOSE_MODE, SHOW_TEST_SAMPLES_INFO_ON_LOAD,
    TRAINING_IMAGES_SIZE, MAX_FLOW, FLOW_MODE,
    IMAGE_MEAN_VARIANCE, IMAGE_DIFF_MIN_THRESHOLD)
from helpers.logger import LOG, INFO
from helpers.debug_tools import calculate_image_difference
from helpers._cv2 import get_optical_flow_from_rgb, get_bidirectional_prewarped_frames, OpticalFlowEmbeddingType

def load_train(path, size):
    '''Prepares the input pipeline to train the model. Each batch is made up of 
    n frames [-n, ..., -2, -1, +1, +2, ..., +n] and a ground truth frame 
    as the expected value to be generated from the network.

    path(str) -- the directory where the dataset is currently stored
    size(int) -- the batch size for the data pipeline
    '''

    assert size > 1     # you don't say?

    groups, _, pipeline = load_core(path)
    return pipeline \
        .shuffle(len(groups), reshuffle_each_iteration=True) \
        .map(lambda x, y: tf.py_func(tf_load_images, inp=[x, y, path], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .map(lambda x, y: tf.py_func(tf_preprocess_train_images, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .filter(lambda x, y: tf.py_func(tf_ensure_difference_min_threshold, inp=[x, y], Tout=[tf.bool])) \
        .map(lambda x, y: tf.py_func(tf_final_input_transform, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .repeat() \
        .apply(tf.contrib.data.batch_and_drop_remainder(size)) \
        .prefetch(2)
        
def load_test(path):
    '''Prepares the input pipeline to test the model. Each sample is made up of 
    n frames [-n, ..., -2, -1, +1, +2, ..., +n] and a ground truth frame 
    as the expected value to be generated from the network.

    path(str) -- the directory where the dataset is currently stored
    '''

    groups, labels, pipeline = load_core(path)

    if SHOW_TEST_SAMPLES_INFO_ON_LOAD:
        for s in zip(groups, labels):
            seq = s[0][:len(s[0]) // 2] + [s[1]] + s[0][len(s[0]) // 2:]
            errors = [
                int(calculate_image_difference(os.path.join(path, pair[0]), os.path.join(path, pair[1]))[2])
                for pair in zip(seq, seq[1:])
            ]
            INFO('{} ---> {}, e={}'.format(s[0], s[1], errors))

    return pipeline \
        .map(lambda x, y: tf.py_func(tf_load_images, inp=[x, y, path], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .map(lambda x, y: tf.py_func(tf_final_input_transform, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=cpu_count()) \
        .batch(1) \
        .prefetch(1) # only process one sample at a time to avoid OOM issues in inference

def load_single_test(path):
    '''Loads a single test sample from the given test directory.'''

    files = os.listdir(path)
    files.sort()
    frames = files[:3]
    frame0, frame1 = cv2.imread(os.path.join(path, frames[0])), cv2.imread(os.path.join(path, frames[-1]))
    return np.concatenate((frame0, frame1), -1)[np.newaxis, :, :, :]


def load_discriminator_samples(path, size):
    '''Prepares the input pipeline for the discriminator model, with the same resolution of
    the frames used for the generator model.

    path(str) -- the directory where the dataset is currently stored
    size(int) -- the batch size for the data pipeline
    '''

    assert size > 1     # you don't say?

    files = os.listdir(path)
    return \
        tf.data.Dataset.from_tensor_slices(files) \
        .shuffle(len(files), reshuffle_each_iteration=True) \
        .map(lambda x: tf.py_func(tf_load_image, inp=[x, path], Tout=[tf.float32]), num_parallel_calls=cpu_count()) \
        .filter(lambda x: tf.py_func(tf_validate_variance, inp=[x], Tout=[tf.bool])) \
        .repeat() \
        .apply(tf.contrib.data.batch_and_drop_remainder(size)) \
        .prefetch(2)

# ====================
# auxiliary methods
# ====================

def load_core(path):
    '''Auxiliary method for the load_train and load_test methods'''

    # load the available frames
    files = os.listdir(path)
    files.sort()
    groups, labels = [], []
    for i in range(len(files) - 2):
        candidates = files[i: i + 1] + files[i + 2: i + 3]

        # filename format: v0-s0-b1_0.jpg
        info1, info2 = candidates[0].split('_')[0].split('-'), candidates[-1].split('_')[0].split('-')
        if info1[0] == info2[0] and info1[1] == info2[1] and info1[2] == info2[2]:
            groups += [candidates]
            labels += [files[i + 1]]
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
        cv2.imread(os.path.join(str(directory)[2:-1], str(sample)[2:-1])).astype(np.float32)
        for sample in samples
    ], dtype=np.float32, copy=False)
    y = cv2.imread(os.path.join(str(directory)[2:-1], str(label)[2:-1])).astype(np.float32)

    return x, y

def tf_load_image(sample, directory):
    '''Loads an image, clipping them to the current training size set.'''

    image = cv2.imread(os.path.join(str(directory)[2:-1], str(sample)[2:-1])).astype(np.float32)

    # randomly resize the image to cover more view area with a single crop
    y_t = randint(TRAINING_IMAGES_SIZE, image.shape[0] - 2 * MAX_FLOW) + 2 * MAX_FLOW
    x_t = y_t * image.shape[1] // image.shape[0]
    image = cv2.resize(image, (x_t, y_t))

    # setup
    max_flow_x = min(MAX_FLOW, image.shape[1] - TRAINING_IMAGES_SIZE)
    max_flow_y = min(MAX_FLOW, image.shape[0] - TRAINING_IMAGES_SIZE)
    x_offset = randint(max_flow_x, image.shape[1] - TRAINING_IMAGES_SIZE - max_flow_x)
    y_offset = randint(max_flow_y, image.shape[0] - TRAINING_IMAGES_SIZE - max_flow_y)
    
    # same slicing for the image
    return image[y_offset:y_offset + TRAINING_IMAGES_SIZE, x_offset:x_offset + TRAINING_IMAGES_SIZE, :]

def tf_preprocess_train_images(samples, label):
    '''Clips the sample images and adds random perturbations to augment the dataset and increase variance.
    
    samples(list<np.array>) -- a tensor with the list of filenames to load
    label(np.array) -- a tensor with the filename of the ground truth image
    '''
    
    assert label.shape[0] >= TRAINING_IMAGES_SIZE and label.shape[1] >= TRAINING_IMAGES_SIZE

    # randomly resize the images to cover more view area with a single crop
    y_t = randint(TRAINING_IMAGES_SIZE, label.shape[0] - 2 * MAX_FLOW) + 2 * MAX_FLOW
    x_t = y_t * label.shape[1] // label.shape[0]
    mode = cv2.INTER_AREA if randint(0, 1) == 0 else cv2.INTER_LINEAR
    samples = np.array([cv2.resize(sample, (x_t, y_t), interpolation=mode) for sample in samples], dtype=np.float32, copy=False)
    label = cv2.resize(label, (x_t, y_t))

    # setup
    max_flow_x = min(MAX_FLOW, label.shape[1] - TRAINING_IMAGES_SIZE)
    max_flow_y = min(MAX_FLOW, label.shape[0] - TRAINING_IMAGES_SIZE)
    x_offset = randint(max_flow_x, label.shape[1] - TRAINING_IMAGES_SIZE - max_flow_x)
    y_offset = randint(max_flow_y, label.shape[0] - TRAINING_IMAGES_SIZE - max_flow_y)
    x_flow = randint(0, max_flow_x)
    y_flow = randint(0, max_flow_y)

    # clip to square, add random flow
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
    
    # same slicing for the label
    y = label[y_offset:y_offset + TRAINING_IMAGES_SIZE, x_offset:x_offset + TRAINING_IMAGES_SIZE, :]

    # random permutations
    choice = randint(0, 2)
    if choice == 0:
        x = np.flip(x, 1)
        y = np.flip(y, 0) # vertical flip
    elif choice == 1:
        x = np.flip(x, 2)
        y = np.flip(y, 1) # horizontal flip
    else:
        x = np.flip(x, 0) # reverse the frames order

    return x, y

def tf_final_input_transform(samples, label):
    '''Reshapes the inputs as needed, and appends the flow estimation, if requested.'''

    # here the inputs are [2, h, w, 3]
    samples_t = np.transpose(samples, [1, 2, 0, 3])
    samples_r = np.reshape(samples_t, [samples_t.shape[0], samples_t.shape[1], -1])

    if FLOW_MODE == OpticalFlowEmbeddingType.NONE:
        return samples_r, label
    if FLOW_MODE == OpticalFlowEmbeddingType.DIRECTIONAL:
        angle, strength = get_optical_flow_from_rgb(samples[0], samples[1], OpticalFlowEmbeddingType.DIRECTIONAL)
        return np.concatenate([samples_r, angle, strength], -1), label
    if FLOW_MODE == OpticalFlowEmbeddingType.BIDIRECTIONAL:
        raise NotImplementedError('Not supported yet')
    if FLOW_MODE == OpticalFlowEmbeddingType.BIDIRECTIONAL_PREWARPED:
        forward, backward = get_bidirectional_prewarped_frames(samples[0], samples[1])
        return np.concatenate([samples_r, forward, backward], -1), label
    raise ValueError('Invalid flow mode')

def tf_calculate_batch_errors(samples, label):
    '''Shared code for ensure_difference_middle_threshold and ensure_difference_min_threshold'''

    # prepare the temporary list of all the sample images
    size = np.prod(samples[0].shape)
    images = [samples[0]] + [label] + [samples[1]]

    # compute the interval errors
    return [
        np.sum((pair[0] - pair[-1]) ** 2, dtype=np.float32) / size
        for pair in zip(images, images[1:])
    ]

def tf_validate_variance(image):
    _, var = cv2.meanStdDev(image)
    return np.sum(var) / 3 > IMAGE_MEAN_VARIANCE

def tf_ensure_difference_min_threshold(samples, label):
    '''Computes the mean squared error between a series of images and returns whether
    or not all the errors respect just the minimum difference constraint.

    images(np.array) -- the input images
    threshold(int) -- the maximum squared difference between the first and last image
    '''

    return all([
        IMAGE_DIFF_MIN_THRESHOLD < error
        for error in tf_calculate_batch_errors(samples, label)
    ]) and tf_validate_variance(label)
