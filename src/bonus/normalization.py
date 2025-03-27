import numpy as np

def RMSnorm(x, gamma, epsilon):
    rms = (np.mean(x**2, axis=0, keepdims=True) + epsilon)**0.5
    return (x / rms) * gamma