from multiprocessing import Process, Queue
from os import listdir
from time import time
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO, BAR, RESET_LINE

PROGRESS_BAR_LENGTH = 20

def save_frame(queue):
    '''Saves a new frame in the background.'''

    while True:
        task = queue.get()
        if task is None:
            break
        cv2.imwrite(task[0], task[1][0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def process_frames(working_path, model_path=None):
    
    # load the inference raw data
    LOG('Preparing samples')
    groups = data_loader.load_inference_samples(working_path, 1)
    previous_idx = len(groups[0]) // 2 - 1
    extension = groups[0][0][-4:] # same image format as the input
    INFO('{} sample(s) to process'.format(len(groups)))

    # setup the background worked
    frames_queue = Queue()
    worker = Process(target=save_frame, args=[frames_queue])
    worker.start()

    # restore the model
    with tf.Session() as session:
        if model_path:
            LOG('Restoring model')
            meta_file_path = [path for path in listdir(model_path) if path.endswith('.meta')][0]
            saver = tf.train.import_meta_graph('{}\\{}'.format(model_path, meta_file_path))
            saver.restore(session, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()

        # initialization
        LOG('Initialization')
        x = graph.get_tensor_by_name('x:0')
        yHat = graph.get_tensor_by_name('inference/uint8_img:0')

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
            frame_path = '{}\\{}_{}'.format(working_path, filename, extension)
            frames_queue.put((frame_path, prediction))

            # update the UI
            progress = (i * PROGRESS_BAR_LENGTH) // len(groups)
            if progress > steps:
                steps = progress
                BAR(steps, PROGRESS_BAR_LENGTH, ' | {0:.3f} fps'.format((i + 1) / (time() - start_seconds)))
    RESET_LINE(True)

    # wait for the background thread
    queue.put(None)
    worker.join()
    queue.close()
    LOG('Inference completed')
