import numpy as np
import cv2

def diff(path1, path2):
    '''Computes the mean squared error between a pair of images

    path1(str) -- the path of the first image to read
    path2(str) -- the path of the second image to read
    '''

    image1, image2 = cv2.imread(path1), cv2.imread(path2) # discard warning if present (pylint's fault)
    error = np.sum((image1.astype(np.float) - image2.astype(np.float)) ** 2)
    return error / float(image1.shape[0] * image1.shape[1])
