import numpy as np

def activation(x):
    return x / (1 + np.exp(-x))

def derivative(x):
    exp = np.exp(-x)
    return (1 + exp + x*exp)/((1 + exp)**2)