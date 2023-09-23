import pathlib
import yaml
from models.analytical import SingleOccupationSingleIsotope, MultiOccupationMultiIsotope
from models.datasets import load_dataset
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras


def load_dataset_info(dataset_name, dataset_dir):
    """
    Load the dataset info from the dataset directory.
    """
    dataset_dir = pathlib.Path(dataset_dir)
    dataset_info_file = dataset_dir / dataset_name / "info.yaml"
    with open(dataset_info_file, "r") as f:
        info = yaml.safe_load(f)
    return info


def prepare_SOSI_fixed(
    physics_weight=0,
    dataset_name="Single-Occupation, Single Isotope, fixed matrix",
    dataset_dir="datasets",
):
    x, y, info = load_dataset(dataset_name, dataset_dir)

    # recover the correction factors for the physics loss
    np.random.seed(info["seed"])
    analytical = SingleOccupationSingleIsotope()
    corrections = keras.ops.convert_to_tensor(analytical.correction_factors())

    input_shape = (info["input_dim"],)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(3, activation="relu"),
        ],
        name=f"{dataset_name}, physics_weight={physics_weight}",
    )

    def physics_loss(y_true, y_pred):
        """
        Loss function is mae + physics_weight * physics_loss
        """
        return keras.ops.mean(
            keras.ops.abs(y_true - y_pred), axis=0
        ) + physics_weight * keras.ops.abs(1 - y_pred * corrections)

    return model, physics_loss, (x, y)


def prepare_SOSI_random(
    physics_weight=0,
    dataset_name="Single-Occupation, Single Isotope, random matrix",
    dataset_dir="datasets",
):
    x, y, info = load_dataset(dataset_name, dataset_dir)

    # append the correction factors to the labels, to be used in the loss function
    y = np.append(y, x[:, 4:7], axis=1)

    input_shape = (info["input_dim"],)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(3, activation="linear"),
        ],
        name=f"{dataset_name}, physics_weight={physics_weight}",
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


def prepare_MOMI_fixed(
    physics_weight=0,
    dataset_name="Multi-Occupation, Multi Isotope, fixed matrix",
    dataset_dir="datasets",
):
    x, y, info = load_dataset(dataset_name, dataset_dir)

    input_shape = (info["input_dim"],)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(12, activation="linear"),
        ],
        name=f"{dataset_name}, physics_weight={physics_weight}",
    )
    corrections = keras.ops.convert_to_tensor(
        MultiOccupationMultiIsotope().correction_factors()
    )

    def physics_loss(y_true, y_pred):
        """
        Loss function is mae + physics_weight * physics_loss
        """
        return keras.ops.mean(
            keras.ops.abs(y_true - y_pred), axis=0
        ) + physics_weight * keras.ops.abs(1 - y_pred * corrections)

    return model, physics_loss, (x, y)
