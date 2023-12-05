import numpy as np


class CPUNormalizer:
    def __call__(self, inputs):
        sums = np.sum(inputs, axis=1, keepdims=True)
        return inputs / sums
