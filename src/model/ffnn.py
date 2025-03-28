import pickle
import numpy as np
from layers.layer import Layer
from activation import relu, sigmoid, tanh, linear, softmax
from bonus.activation import gelu, swish
from bonus.normalization import RMSnorm

activation_map = {
    "relu": (relu.activation, relu.derivative),
    "sigmoid": (sigmoid.activation, sigmoid.derivative),
    "tanh": (tanh.activation, tanh.derivative),
    "linear": (linear.activation, linear.derivative),
    "softmax": (softmax.activation, softmax.derivative),
    "gelu": (gelu.activation, gelu.derivative),
    "swish": (swish.activation, swish.derivative),
}

class FFNN:
    def __init__(self, layer_sizes, activations, learning_rate, regularization="none", rmsnorm=False, epsilon=1e-7, init_method="uniform", init_params={}):
        assert len(layer_sizes) - 1 == len(activations), "Number of layers and activations should match"
        
        self.layers = []
        self.activations = []
        self.input_cache = None

        # init for regularization
        self.regularization = regularization
        self.l = learning_rate
        

        for i in range(len(layer_sizes) - 1):
            layer = Layer(input_size=layer_sizes[i], output_size=layer_sizes[i+1], init_method=init_method, init_params=init_params)
            self.layers.append(layer)

            act_func, act_deriv = activation_map[activations[i]]
            self.activations.append((act_func, act_deriv))
        
        # init for normalization
        self.rmsnorm = rmsnorm
        self.epsilon = epsilon
        self.gammas = []
        if self.rmsnorm:
            self.gammas = [np.ones((layer_sizes[i+1], 1)) for i in range(len(layer_sizes) - 1)]

    def forward(self, X):
        """Forward propagation"""
        self.input_cache = X
        a = X
        i = 0
        for layer, (act_func, _) in zip(self.layers, self.activations):
            z = layer.forward(a)
            a = act_func(z)

            if self.rmsnorm:
                a = RMSnorm(a, self.gammas[i], self.epsilon)

            layer.z = z
            layer.a = a
            i += 1
        return a

    def backward(self, y_true, loss_fn, loss_deriv):
        batch_size = y_true.shape[1]
        last_idx = len(self.layers) - 1
        last_layer = self.layers[last_idx]

        if self.activations[last_idx][0].__module__.endswith("softmax") and loss_fn.__module__.endswith("categorical_crossentropy"):
            dz = last_layer.a - y_true
        else:
            last_activation_deriv = self.activations[last_idx][1]
            dz = loss_deriv(y_true, last_layer.a) * last_activation_deriv(last_layer.z)

        da = dz

        prev_a = self.layers[last_idx - 1].a if last_idx > 0 else self.input_cache
        last_layer.grad_weights = np.dot(da, prev_a.T) / batch_size
        last_layer.grad_bias = np.sum(da, axis=1, keepdims=True) / batch_size

        for i in reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            activation_deriv = self.activations[i][1]
            dz = np.dot(next_layer.weights.T, da) * activation_deriv(current_layer.z)
            da = dz

            input_activation = self.layers[i - 1].a if i > 0 else self.input_cache
            current_layer.grad_weights = np.dot(dz, input_activation.T) / batch_size
            current_layer.grad_bias = np.sum(dz, axis=1, keepdims=True) / batch_size

    def update_weights(self, lr):
        """Update weights and biases on each layer"""
        for layer in self.layers:
            if self.regularization == "l1":
                layer.weights -= lr * (self.l * np.sign(layer.weights))
            elif self.regularization == "l2":
                layer.weights -= lr * (self.l * 2 * layer.weights)

            layer.weights -= lr * layer.grad_weights
            layer.bias -= lr * layer.grad_bias

    def save(self, filename):
        """
        Save model to pickle file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load model from pickle file
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)