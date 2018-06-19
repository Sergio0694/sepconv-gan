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
    gradients = [
        None if gradient is None else tf.clip_by_norm(gradient, clip)
        for gradient in gradients
    ]
    return optimizer.apply_gradients(zip(gradients, variables))

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
