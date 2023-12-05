from models.cpu.layers.layer import CPULayer
from models.cpu.layers.dense import CPUDense
from models.cpu.layers.normalizer import CPUNormalizer
import os

os.environ["KERAS_BACKEND"] = "torch"
