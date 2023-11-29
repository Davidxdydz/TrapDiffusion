from models.analytical import SingleOccupationSingleIsotope, MultiOccupationMultiIsotope
from training.datasets import load_dataset
import numpy as np
import os
from typing import Tuple, Callable

os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras


class ModelBuilder:
    def __init__(self, dataset_name, dataset_dir="datasets"):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

    def build_model(
        self,
        intput_channels,
        output_channels,
        layer_sizes,
        activations,
        output_activation,
        name,
    ):
        model = keras.Sequential(name=name)
        model.add(keras.layers.Input(shape=(intput_channels,)))
        for size, activation in zip(layer_sizes, activations):
            model.add(keras.layers.Dense(size, activation=activation))
        model.add(keras.layers.Dense(output_channels, activation=output_activation))
        return model

    def info(self):
        return dict(
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
        )

    def prepare(self, layer_sizes, activations, physics_weight, output_activation):
        x, y, c, info = load_dataset(self.dataset_name, self.dataset_dir)

        # append the correction factors to the labels, to be used in the loss function
        y_with_corrections = np.append(y, c, axis=1)

        model = self.build_model(
            intput_channels=info["input_dim"],
            output_channels=info["output_dim"],
            layer_sizes=layer_sizes,
            activations=activations,
            output_activation=output_activation,
            name=f"{self.dataset_name}, physics_weight={physics_weight}",
        )

        def physics_loss(y_true, y_pred):
            """
            Loss function is mae + physics_weight * physics_loss
            """
            ys = y_true[:, : -info["output_dim"]]
            corrections = y_true[:, -info["output_dim"] :]
            mae_loss = keras.ops.mean(keras.ops.abs(ys - y_pred), axis=1)
            total_mass = keras.ops.sum(y_pred * corrections, axis=1, keepdims=True)
            physics_loss = keras.ops.abs(1 - total_mass)
            return (1 - physics_weight) * mae_loss + physics_weight * physics_loss

        return model, physics_loss, (x, y_with_corrections), (x, y)
