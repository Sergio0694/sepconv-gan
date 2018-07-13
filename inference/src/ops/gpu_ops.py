import os
import tensorflow as tf
from tensorflow.python.framework import ops

SEPCONV_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sepconv.so')

def load_ops():
    tf.load_op_library(SEPCONV_SO_PATH)
