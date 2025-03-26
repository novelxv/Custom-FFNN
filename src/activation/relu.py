import numpy as np

def activation(x):
    return np.maximum(0, x)

def derivative(x):
    return (x > 0).astype(float)