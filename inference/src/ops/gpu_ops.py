import os
import tensorflow as tf

SEPCONV_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sepconv.so')
NEAREST_SHADER_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nearest_shader.so')

def load_ops():
    '''Loads the custom GPU ops used in the network.'''

    tf.load_op_library(SEPCONV_SO_PATH)
    tf.load_op_library(NEAREST_SHADER_SO_PATH)
