import numpy as np
from typing import List

def predict(weights: List[np.ndarray], biases: List[np.ndarray], input: np.ndarray) -> np.ndarray:
    """
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    input: np.ndarray
        must be (batch_size, input_size)
    """

def affine_batched(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    A: np.ndarray
        must be (n,n)
    x: np.ndarray
        must be (batch_size, n)
    b: np.ndarray
        must be (n,)
    """

def relu(x: np.ndarray) -> np.ndarray:
    """
    x: np.ndarray
    """