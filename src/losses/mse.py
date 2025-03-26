import numpy as np

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size