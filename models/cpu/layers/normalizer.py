import numpy as np
from models.cpu.layers.layer import CPULayer
from models.pinn import Normalizer


class CPUNormalizer(CPULayer):
    @staticmethod
    def from_keras(layer):
        return CPUNormalizer()

    def __call__(self, inputs):
        sums = np.sum(inputs, axis=1, keepdims=True)
        return inputs / sums


CPULayer.cpu_translation[Normalizer] = CPUNormalizer
