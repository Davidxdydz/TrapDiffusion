from training.datasets import load_dataset
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras


class Normalizer(keras.layers.Layer):
    def call(self, inputs):
        sums = keras.ops.sum(inputs, axis=1, keepdims=True)
        return inputs / sums


class PhysicsLoss:
    def __init__(self, physics_weight):
        self.physics_weight = physics_weight

    def __call__(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor):
        """
        Loss function is mae + physics_weight * physics_loss
        """
        # the y_true also store the correction factors in the second half
        corrections_start = y_pred.shape[-1]
        y_true, corrections = (
            y_true[:, :-corrections_start],
            y_true[:, -corrections_start:],
        )
        mae_loss = keras.losses.mean_absolute_error(y_true, y_pred)
        total_mass = keras.ops.sum(y_pred * corrections, axis=-1, keepdims=True)
        physics_loss = keras.ops.abs(1 - total_mass)
        return (1 - self.physics_weight) * mae_loss + self.physics_weight * physics_loss

    def get_config(self):
        return dict(physics_weight=self.physics_weight)


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
        normalizer=False,
    ):
        model = keras.Sequential(name=name)
        model.add(keras.layers.Input(shape=(intput_channels,)))
        for size, activation in zip(layer_sizes, activations):
            model.add(keras.layers.Dense(size, activation=activation))
        model.add(keras.layers.Dense(output_channels, activation=output_activation))
        if normalizer:
            model.add(Normalizer())
        return model

    def info(self):
        return dict(
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
        )

    def prepare(self, layer_sizes, activations, physics_weight, output_activation):
        x, y, c, info = load_dataset(self.dataset_name, self.dataset_dir)

        x_train = x
        x_val = x
        y_train = None
        y_val = None
        pre_normalized = info.get("pre_normalized", False)
        if not pre_normalized:
            # append the correction factors to the labels, to be used in the loss function
            y_train = np.append(y, c, axis=1)
            y_val = y
        else:
            y_train = y
            y_val = y

        model = self.build_model(
            intput_channels=info["input_dim"],
            output_channels=info["output_dim"],
            layer_sizes=layer_sizes,
            activations=activations,
            output_activation=output_activation,
            name=f"{self.dataset_name}, {f'physics_weight={physics_weight}' if not pre_normalized else 'pre_normalized'}",
            normalizer=pre_normalized,
        )

        loss = (
            PhysicsLoss(physics_weight)
            if not pre_normalized
            else keras.losses.mean_absolute_error
        )

        return (
            model,
            loss,
            (x_train, y_train),
            (x_val, y_val),
        )
