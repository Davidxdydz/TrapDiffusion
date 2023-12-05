import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np


class CPUDense:
    @staticmethod
    def from_keras(layer: keras.layers.Dense):
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        activation = layer.activation.__name__
        return CPUDense(weights, biases, activation)

    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def __call__(self, inputs):
        x = inputs @ self.weights + self.biases
        if self.activation == "relu":
            x[x < 0] = 0
        elif self.activation == "linear":
            ...
        elif self.activation == "tanh":
            x = np.tanh(x)
        elif self.activation == "leaky_relu":
            x[x < 0] = 0.3 * x[x < 0]
        elif self.activation == "sigmoid":
            x = 1 / (1 + np.exp(-x))
        else:
            raise NotImplementedError(f"Activation {self.activation} not implemented")
        return x
