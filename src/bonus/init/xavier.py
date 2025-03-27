import numpy as np

def xavier(shape, xmethod):
    if (xmethod == 'uniform'):
        x = (6/(shape[0] + shape[1]))**0.5
        return np.random.uniform(-x, x, shape)
    elif (xmethod == 'normal'):
        std = (2/(shape[0] + shape[1]))**0.5
        return np.random.normal(0, std, shape)
        