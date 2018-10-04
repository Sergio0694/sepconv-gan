import os
import tensorflow as tf
from tensorflow.python.framework import ops

SEPCONV_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sepconv.so')
NEAREST_SHADER_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nearest_shader.so')
SEPCONV_MODULE = tf.load_op_library(SEPCONV_SO_PATH)
NEAREST_SHADER_MODULE = tf.load_op_library(NEAREST_SHADER_SO_PATH)
