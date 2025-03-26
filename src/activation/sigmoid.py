import numpy as np

def activation(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    sig = activation(x)
    return sig * (1 - sig)