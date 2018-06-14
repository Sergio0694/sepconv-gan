from os import listdir
from time import time
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO, BAR, RESET_LINE

MODEL_ROOT_PATH = r'D:\ML\th\trained_models\simple_cnn_v3_520.000'
PROGRESS_BAR_LENGTH = 20

def process_frames(working_path):
    
    # load the inference raw data
    LOG('Preparing samples')
    groups = data_loader.load_inference_samples(working_path, 1)
    previous_idx = len(groups[0]) // 2 - 1
    extension = groups[0][0][-4:] # same image format as the input
    INFO('{} sample(s) to process'.format(len(groups)))

    # restore the model
    LOG('Restoring model')
    meta_file_path = [path for path in listdir(MODEL_ROOT_PATH) if path.endswith('.meta')][0]
    tf.reset_default_graph()
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('{}\\{}'.format(MODEL_ROOT_PATH, meta_file_path))
        saver.restore(session, tf.train.latest_checkpoint(MODEL_ROOT_PATH))
        graph = tf.get_default_graph()

        # initialization
        LOG('Initialization')
        x = graph.get_tensor_by_name('x:0')
        yHat = graph.get_tensor_by_name('uint8_img:0')

        # process the data
        LOG('Processing items')
        BAR(0, PROGRESS_BAR_LENGTH)
        steps = 0
        start_seconds = time()
        for i, group in enumerate(groups):

            # load the current sample
            frames = np.array([[
                cv2.imread('{}\\{}'.format(working_path, sample)).astype(np.float32)
                for sample in group
            ]], dtype=np.float32, copy=False)
            filename = group[previous_idx][:-4]

            # inference
            prediction = session.run(yHat, feed_dict={x: frames})
            cv2.imwrite('{}\\{}_{}'.format(working_path, filename, extension), prediction[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # update the UI
            progress = (i * PROGRESS_BAR_LENGTH) // len(groups)
            if progress > steps:
                steps = progress
                BAR(steps, PROGRESS_BAR_LENGTH, ' | {0:.3f} fps'.format((i + 1) / (time() - start_seconds)))
    RESET_LINE(True)
    LOG('Inference completed')
