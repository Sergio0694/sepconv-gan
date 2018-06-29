import os
import tensorflow as tf
from tensorflow.python.framework import ops

SEPCONV_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sepconv', 'sepconv.so')
SEPCONV_MODULE = tf.load_op_library(SEPCONV_SO_PATH)
 
@ops.RegisterGradient("Sepconv")
def _sepconv_grad(op, grad):
    """
    The gradient for `sepconv` from sepconv.cc.
    
    op -- the TensorFlow operation
    grad -- the output gradient
    """
    
    kv_grad, kh_grad = SEPCONV_MODULE.sepconv_grad(
        grad,                       # output gradient
        op.inputs[0],               # input image (constant)
        op.inputs[1].shape[-1])     # depth of the separable kernels
    return [None, kv_grad, kh_grad]