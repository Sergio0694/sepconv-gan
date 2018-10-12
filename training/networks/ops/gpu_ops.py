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
        op.inputs[1],
        op.inputs[2])               # kernels used in the forward pass
    return [None, kv_grad, kh_grad]

NEAREST_SHADER_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nearest_shader', 'nearest_shader.so')
NEAREST_SHADER_LIB = tf.load_op_library(NEAREST_SHADER_SO_PATH)

KERNEL_SIZE = 10.0
KERNEL_MEAN = 0.0
KERNEL_STD_DEV = 10.0

def NEAREST_SHADER_MODULE(x, frame0, frame1):

    # get the distance map
    distance_raw = (tf.abs(frame0[:, :, :, 0] - frame1[:, :, :, 0]) + \
                    tf.abs(frame0[:, :, :, 1] - frame1[:, :, :, 1]) + \
                    tf.abs(frame0[:, :, :, 2] - frame1[:, :, :, 2])) / 3.0
    distance_scaled = tf.clip_by_value(distance_raw, 0.0, 32.0) * 8.0
    distance_clip = tf.clip_by_value(distance_scaled, 0.0, 255.0) / 255.0

    # pad with reflection to avoid artifacts on the image edges
    image_pad = tf.pad(distance_clip[:, :, :], [[0, 0], [10, 10], [10, 10]], mode='SYMMETRIC')
    image_4d = image_pad[:, :, :, tf.newaxis]

    # get the gaussian kernel
    normal = tf.distributions.Normal(KERNEL_MEAN, KERNEL_STD_DEV)
    samples = normal.prob(tf.range(-KERNEL_SIZE, KERNEL_SIZE + 1, dtype=tf.float32))
    kernel_raw = tf.einsum('i,j->ij', samples, samples)
    kernel_norm = kernel_raw / tf.reduce_sum(kernel_raw)
    kernel_4d = kernel_norm[:, :, tf.newaxis, tf.newaxis]

    # process the blur effect
    blurred_4d = tf.nn.conv2d(image_4d, kernel_4d, [1, 1, 1, 1], 'VALID')
    blurred_3d = blurred_4d[:, :, :, 0]
    blend_map = tf.clip_by_value(blurred_3d * 255.0, 0.0, 255.0)

    # execute the pixel shader
    return NEAREST_SHADER_LIB.nearestshader(x, frame0, frame1, blend_map)