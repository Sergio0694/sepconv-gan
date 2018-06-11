import numpy as np
import cv2

def load_images(filenames, directory):
    '''Loads and returns a list of images from the input list of filenames
    
    filenames(list<str>) -- the list of filenames to load
    directory(str) -- the parent directory for the input files
    '''
    
    print('=========== FILES ===========')
    print(filenames)
    print('===========')
    return np.array([
        cv2.imread('{}\\{}'.format(str(directory)[2:-1], str(filename)[2:-1])).astype(np.float32)
        for filename in filenames
    ], dtype=np.float32, copy=False)

def ensure_difference_threshold(images, threshold):
    '''Computes the mean squared error between a pair of images

    image1(np.array) -- the first loaded image
    image2(np.array) -- the second loaded image
    '''

    size = images[0].shape[0] * images[0].shape[1]
    print('shape ---> {}'.format(size))
    error = np.sum((images[0] - images[-1]) ** 2, dtype=np.float32) / size
    print('error ---> {}'.format(error))
    return error < threshold

def diff(path1, path2):
    '''Computes the mean squared error between a pair of images

    image1(np.array) -- the first loaded image
    image2(np.array) -- the second loaded image
    '''

    imgs = np.array([
        cv2.imread(path).astype(np.float32)
        for path in [path1, path2]
    ], dtype=np.float32, copy=False)

    size = imgs[0].shape[0] * imgs[0].shape[1] * 3
    error = np.sum((imgs[0] - imgs[-1]) ** 2, dtype=np.float32) / size
    print('size={}, error={}'.format(size, error))
