from functools import reduce
import numpy as np
import cv2
from helpers.logger import LOG, INFO

def display_image_difference(path1, path2):
    '''Computes the mean squared error between a pair of images and shows additional info

    path1(str) -- the first image full path
    path2(str) -- the second image full path
    '''

    LOG('Loading images')
    INFO(path1)
    INFO(path2)
    images = np.array([
        cv2.imread(path).astype(np.float32)
        for path in [path1, path2]
    ], dtype=np.float32, copy=False)

    size = reduce(lambda x, y: x * y, list(images.shape))
    LOG('{}x{} resolution, {} total size'.format(images[0].shape[0], images[0].shape[1], size))

    error = np.sum((images[0] - images[-1]) ** 2, dtype=np.float32) / size
    LOG('Mean squared difference: {}'.format(error))
