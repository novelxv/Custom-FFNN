import numpy as np
from src.utils.initializer import initialize_weights

class Layer:
    def __init__(self, input_size, output_size, init_method="uniform", init_params={}):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = initialize_weights(init_method, (output_size, input_size), **init_params)
        if (init_method == "he"):
            self.bias = 0
        else:
            self.bias = initialize_weights(init_method, (output_size, 1), **init_params)

        self.z = None
        self.a = None

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        self.z = np.dot(self.weights, input) + self.bias
        return self.z