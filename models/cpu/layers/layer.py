import keras
from typing import Dict


class CPULayer:
    cpu_translation: Dict[keras.layers.Layer, "CPULayer"] = {}

    @staticmethod
    def from_keras(layer: keras.layers.Layer) -> "CPULayer":
        if type(layer) in CPULayer.cpu_translation:
            return CPULayer.cpu_translation[type(layer)].from_keras(layer)
        else:
            print(CPULayer.cpu_translation)
            raise NotImplementedError(f"Layer {type(layer)} not implemented")

    def __call__(self, inputs):
        raise NotImplementedError()
