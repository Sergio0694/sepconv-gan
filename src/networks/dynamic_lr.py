class DynamicRate(object):

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

    def __init__(self, rate, decay=0.94):
        self.rate = rate
        self.decay = decay

    def get(self):
        '''Returns the appropriate learning rate for the current training step.'''
        
        lr, self.rate = self.rate, self.rate ** (1.0 / self.decay)
        return lr