import numpy as np

def he(shape):
    return np.random.normal(0, (2/shape[1])**0.5, shape)
        