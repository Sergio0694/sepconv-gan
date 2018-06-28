from os.path import dirname, abspath
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

SEPCONV_SO_PATH = '{}/sepconv.so'.format(dirname(abspath(__file__)))
sepconv_module = tf.load_op_library(SEPCONV_SO_PATH)
 
@ops.RegisterGradient("Sepconv")
def _sepconv_grad(op, grad):
    """
    The gradient for `sepconv` from sepconv.cc.
    
    op -- the TensorFlow operation
    grad -- the output gradient
    """
    
    return sepconv_module.sepconv_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])
