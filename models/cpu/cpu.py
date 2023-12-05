import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
from models.pinn import Normalizer
from models.cpu.layers import CPUDense, CPUNormalizer


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
