import os

os.environ["KERAS_BACKEND"] = "torch"
import keras

from models.cpu.layers import CPUDense, CPUNormalizer
from keras.layers import Dense
from models.pinn import Normalizer

cpu_translation = {Dense: CPUDense, Normalizer: CPUNormalizer}


class CPULayer:
    def __new__(cls, layer: keras.layers.Layer):
        if type(layer) in cpu_translation:
            return cpu_translation[type(layer)](layer)
        else:
            raise NotImplementedError(f"Layer {layer} not implemented")

    def __call__(self, inputs):
        raise NotImplementedError()
