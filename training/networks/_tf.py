from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

def prelu(x, name):
    '''Parametric ReLU activation function.'''

    alpha = tf.get_variable(name, shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

def minimize_with_clipping(optimizer, loss, clip=5.0, scope=None):
    '''Returns an operation that minimizes the input loss function, 
    while clipping the gradients by the specified local norm.

    optimizer(tf.python.training.optimizer.Optimizer) -- the optimizer to use to minimize the loss
    loss(tf.Tensor) -- the loss function to minimize
    clip(float) -- the amount of local norm to use when clipping the gradients
    scope(str) -- the optional scope for the variables of the optimizer'''

    gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)))
    with tf.name_scope('clipping'):
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, clip)
            for gradient in gradients
        ]
    return optimizer.apply_gradients(zip(gradients, variables))

def initialize_variables(session):
    '''Initializes all uninitialized variables in the current session.

    session(tf.Session) -- the current executing session
    '''

    global_vars = tf.global_variables()
    is_not_initialized = session.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    
    if len(not_initialized_vars):
        session.run(tf.variables_initializer(not_initialized_vars))

def get_parent_by_match(tensor, tokens):
    '''Retrieves the parent tensor with the specified name parts.

    tensor(tf.Tensor) -- the child tensor
    tokens(list<str>) -- the name parts to look for
    '''

    for parent in tensor.op.inputs:
        if all((token in parent.name for token in tokens)):
            return parent
        return get_parent_by_match(parent, tokens)
    return None

def luminance_loss(t1, t2):
    '''Gets a loss based on the L2 error on the luminance difference between the two input tensors.
    The input images must have values in the [0, 1] range.

    t1(tf.Tensor) -- a tensor which represents a batch of BGR images [b, h, w, 3]
    t2(tf.Tensor) -- a tensor with the same shape as the other
    '''

    assert len(t1.shape) == 4
    assert t1.shape[-1] == 3

    def get_luminance(image):
        b, g, r = tf.unstack(image, axis=-1)
        return 0.0722 * b + 0.7152 * g + 0.2126 * r   
    
    return tf.reduce_mean((get_luminance(t1) - get_luminance(t2)) ** 2)

class DynamicRate(object):
    '''A class that produces learning rates as specified from the 
    initial mapping of custom rates to training epochs.
    '''

    def __init__(self, rates):
        self.rates = rates
        self.keys = sorted(rates)[::-1]

    def get(self, step):
        '''Returns the appropriate learning rate for the current training step.'''

        for key in self.keys:
            if step >= key:
                return self.rates[key]
        raise RuntimeError('wut?')

class DecayingRate(object):
    '''A class that returns an exponentially decaying learning rate.'''

    def __init__(self, rate, decay=0.94, sustain=0):
        self.rate = rate
        self.decay = decay
        self.sustain = sustain

    def get(self):
        '''Returns the appropriate learning rate for the current training step.'''
        
        if self.sustain > 0:
            self.sustain -= 1
            return self.rate
        lr, self.rate = self.rate, self.rate ** (1.0 / self.decay)
        return lr

class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.

    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
