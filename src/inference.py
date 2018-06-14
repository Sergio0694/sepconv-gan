from pathlib import Path
import cv2
import tensorflow as tf
import numpy as np
from __MACRO__ import *
import dataset.dataset_loader as data_loader
from helpers.logger import LOG, INFO, BAR, RESET_LINE

MODEL_ROOT_PATH = 'D:\\ML\\th\\trained_models\\simple_cnn_v3'
META_FILE_PATH = '{}\\{}'.format(MODEL_ROOT_PATH, 'simple_cnn_v3.meta')
PROGRESS_BAR_LENGTH = 20
SOURCE_PATH = 'D:\\ML\\th\\datasets\\test_1080'
OUTPUT_PATH = 'D:\\ML\\th\\datasets\\inference_1080'

# load the inference raw data
LOG('Preparing samples')
_, groups, _ = data_loader.calculate_samples_data(SOURCE_PATH, 1)
previous_idx = len(groups[0]) // 2 - 1
INFO('{} sample(s) to process'.format(len(groups)))

# restore the model
LOG('Restoring model')
tf.reset_default_graph()
with tf.Session() as session:
    saver = tf.train.import_meta_graph(META_FILE_PATH)
    saver.restore(session, tf.train.latest_checkpoint(MODEL_ROOT_PATH))
    graph = tf.get_default_graph()

    # initialization
    LOG('Initialization')
    session.run(tf.global_variables_initializer())
    x = graph.get_tensor_by_name('x:0')
    yHat = graph.get_tensor_by_name('uint8_img:0')

    # process the data
    LOG('Processing items')
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    steps = 0
    for i, group in enumerate(groups):

        # load the current sample
        frames = np.array([[
            cv2.imread('{}\\{}'.format(SOURCE_PATH, sample)).astype(np.float32)
            for sample in group
        ]], dtype=np.float32, copy=False)
        filename = group[previous_idx]

        # inference
        prediction = session.run(yHat, feed_dict={x: frames})
        cv2.imwrite('{}\\{}_.jpg'.format(OUTPUT_PATH, filename), prediction[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # update the UI
        progress = (i * PROGRESS_BAR_LENGTH) // len(groups)
        if progress > steps:
            steps = progress
            BAR(steps, PROGRESS_BAR_LENGTH)
RESET_LINE()
LOG('Inference completed')
