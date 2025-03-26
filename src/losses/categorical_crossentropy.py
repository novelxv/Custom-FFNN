import numpy as np

def loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

def derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[1]