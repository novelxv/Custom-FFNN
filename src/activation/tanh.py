import numpy as np

def activation(x):
    return np.tanh(x)

def derivative(x):
    return 1 - np.tanh(x)**2