from __MACRO__ import *
import tensorflow as tf

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

    def __init__(self, rate, decay=0.94):
        self.rate = rate
        self.decay = decay

    def get(self):
        '''Returns the appropriate learning rate for the current training step.'''
        
        lr, self.rate = self.rate, self.rate ** (1.0 / self.decay)
        return lr
