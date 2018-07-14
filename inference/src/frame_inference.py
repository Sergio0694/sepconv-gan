import os
import cv2
import numpy as np
import tensorflow as tf
from src.__MACRO__ import LOG, ERROR
from src.ops.gpu_ops import load_ops

def process_frames(model_path, path_0, path_1):
    '''Generates an intermediate frame from the two in input.

    model_path(str) -- the path of the saved model
    path_0(str) -- the path of the first frame
    path_1(str) -- the path of the second frame
    '''

    LOG('Preparing data')
    frame_0, frame_1 = cv2.imread(path_0), cv2.imread(path_1)
    if frame_0.shape != frame_1.shape:
        ERROR('The two input frames have a different size')

    load_ops()
    with tf.Session() as session:

        # restore the model from the .meta and check point files
        LOG('Restoring model')
        meta_file_path = [path for path in os.listdir(model_path) if path.endswith('.meta')][0]
        saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file_path))
        saver.restore(session, tf.train.latest_checkpoint(model_path))

        # initialization
        LOG('Initialization')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        yHat = graph.get_tensor_by_name('inference/uint8_img:0')

        # process the frames
        LOG('Processing')
        stack = np.concatenate([frame_0, frame_1], axis=-1)
        frames = np.expand_dims(stack, axis=0)
        return session.run(yHat, feed_dict={x: frames})[0]
