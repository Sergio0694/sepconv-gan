import numpy as np
import cv2
from helpers.logger import LOG, INFO

def calculate_image_difference(path1, path2):
    '''Computes the mean squared error between a pair of images and also returns their
    resolution and the total number of color points in each image
    
    path1(str) -- the first image full path
    path2(str) -- the second image full path
    '''

    images = np.array([
        cv2.imread(path).astype(np.float32)
        for path in [path1, path2]
    ], dtype=np.float32, copy=False)

    size = np.prod(images[0].shape)
    error = np.sum((images[0] - images[-1]) ** 2, dtype=np.float32) / size
    return images[0].shape, size, error

def display_image_difference(path1, path2):
    '''Displays the mean squared error between a pair of images and shows additional info

    path1(str) -- the first image full path
    path2(str) -- the second image full path
    '''

    LOG('Loading images')
    INFO(path1)
    INFO(path2)
    resolution, size, error = calculate_image_difference(path1, path2)

    LOG('{}x{} resolution, {} total size'.format(resolution[0], resolution[1], size))
    LOG('Mean squared difference: {}'.format(error))
