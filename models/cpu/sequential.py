import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
from models.cpu.layers.layer import CPULayer


class CPUSequential:
    def get_from(self, pinn: keras.Sequential):
        layers = []
        for layer in pinn.layers:
            layers.append(CPULayer.from_keras(layer))
        self.layers = layers

    def __init__(self, model: keras.Sequential):
        self.get_from(model)
        self.input_shape = model.input_shape
        self.output_shape = model.output_shape
        self.name = model.name

    def predict(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x
