import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
from models.pinn import Normalizer


class CPUNormalizer:
    def __call__(self, inputs):
        sums = np.sum(inputs, axis=1, keepdims=True)
        return inputs / sums


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


class CPUSequential:
    def get_from(self, pinn: keras.Sequential):
        layers = []
        for layer in pinn.layers:
            if isinstance(layer, keras.layers.Dense):
                layers.append(CPUDense.from_keras(layer))
            elif isinstance(layer, Normalizer):
                layers.append(CPUNormalizer())
            else:
                raise NotImplementedError(f"Layer {layer} not implemented")
        self.layers = layers

    def __init__(self, model: keras.Sequential):
        self.get_from(model)
        self.input_shape = model.input_shape
        self.output_shape = model.output_shape
        self.name = model.name

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
