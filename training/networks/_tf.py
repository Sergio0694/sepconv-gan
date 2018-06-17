import tensorflow as tf

def stack_images(x):
    '''Stacks n images over the channels dimension.

    x(tf.tensor<tf.float32>) -- the input [batch, images, h, w, channels] tensor
    '''

    with tf.name_scope('frames'):
        x_t = tf.transpose(x, [0, 2, 3, 1, 4])
        x_shape = tf.shape(x_t, name='batch_shape')
        return tf.reshape(x_t, [x_shape[0], x_shape[1], x_shape[2], 6], name='frames')

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
