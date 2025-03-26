import numpy as np

def activation(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def derivative(x):
    raise NotImplementedError("Use softmax + cross-entropy simplification in loss backward")