import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from __MACRO__ import *
import dataset.dataset_loader as data_loader

def show_samples(size):

    # pipeline setup
    train_dataset = data_loader.load_train(TRAINING_DATASET_PATH, size, IMAGES_WINDOW_SIZE)
    gen_iterator = train_dataset.make_one_shot_iterator()
    x_train, y = gen_iterator.get_next()

    with tf.Session() as session:

        # get the dataset items and prepare the results
        samples, labels = session.run([x_train / 255.0, y / 255.0])
        results = []
        for pair in zip(samples, labels):
            results += [pair[0][:, :, :3]]
            results += [pair[1]]
            results += [pair[0][:, :, 3:]]
        array = (np.array(results) * 255.0).astype(np.uint8)
        index, height, width, channels = array.shape

        # show the results
        columns = 9
        rows = index // columns        
        output = array \
            .reshape(rows, columns, height, width, channels) \
            .swapaxes(1, 2) \
            .reshape(height * rows, width * columns, channels)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.show()

if __name__ == '__main__':
    show_samples(18)