from bonus.init.xavier import xavier
from bonus.init.he import he
import numpy as np

def initialize_weights(method, shape, **kwargs):
    seed = kwargs.get("seed", None)
    if seed is not None:
        np.random.seed(seed)
    if method == "zero":
        return np.zeros(shape)
    elif method == "uniform":
        low = kwargs.get("lower", -1.0)
        high = kwargs.get("upper", 1.0)
        return np.random.uniform(low, high, shape)
    elif method == "normal":
        mean = kwargs.get("mean", 0.0)
        var = kwargs.get("variance", 1.0)
        std = np.sqrt(var)
        return np.random.normal(mean, std, shape)
    elif method == "xavier":
        xmethod = kwargs.get("xmethod", "uniform")
        return xavier(shape, xmethod)
    elif method == "he":
        return he(shape)
    else:
        raise ValueError(f"Unknown initialization method: {method}")