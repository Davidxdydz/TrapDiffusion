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

    def prepare(
        self, layer_sizes, activations, physics_weight, output_activation
    ) -> Tuple[
        keras.Sequential,
        Callable[[keras.KerasTensor, keras.KerasTensor], keras.KerasTensor],
        Tuple[np.ndarray, np.ndarray],
    ]:
        raise NotImplementedError()


class SOSIFixed(ModelBuilder):
    def __init__(self):
        ModelBuilder.__init__(self, "Single-Occupation, Single Isotope, fixed matrix")

    def prepare(self, layer_sizes, activations, physics_weight, output_activation):
        x, y, info = load_dataset(self.dataset_name, self.dataset_dir)

        # recover the correction factors for the physics loss
        np.random.seed(info["seed"])
        analytical = SingleOccupationSingleIsotope()
        corrections = keras.ops.convert_to_tensor(analytical.correction_factors())

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
            return keras.ops.mean(
                keras.ops.abs(y_true - y_pred), axis=0
            ) + physics_weight * keras.ops.abs(1 - y_pred * corrections)

        return model, physics_loss, (x, y)


class SOSIRandom(ModelBuilder):
    def __init__(self):
        ModelBuilder.__init__(self, "Single-Occupation, Single Isotope, random matrix")

    def prepare(self, layer_sizes, activations, physics_weight, output_activation):
        x, y, info = load_dataset(self.dataset_name, self.dataset_dir)

        # append the correction factors to the labels, to be used in the loss function
        y = np.append(y, x[:, 4:7], axis=1)

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
            ys = y_true[:, :-3]
            corrections = y_true[:, -3:]
            return keras.ops.mean(
                keras.ops.abs(ys - y_pred), axis=0
            ) + physics_weight * keras.ops.abs(1 - y_pred * corrections)

        return model, physics_loss, (x, y)


class MOMIFixed(ModelBuilder):
    def __init__(self):
        ModelBuilder.__init__(self, "Multi-Occupation, Multi Isotope, fixed matrix")

    def prepare(self, layer_sizes, activations, physics_weight, output_activation):
        x, y, info = load_dataset(self.dataset_name, self.dataset_dir)

        # recover the correction factors for the physics loss
        np.random.seed(info["seed"])
        analytical = MultiOccupationMultiIsotope()
        corrections = keras.ops.convert_to_tensor(analytical.correction_factors())

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
            return keras.ops.mean(
                keras.ops.abs(y_true - y_pred), axis=0
            ) + physics_weight * keras.ops.abs(1 - y_pred * corrections)

        return model, physics_loss, (x, y)
