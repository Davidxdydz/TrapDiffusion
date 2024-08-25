import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras_tuner as kt
from training.utils import estimate_cpu_time
from keras import ops
from models.gpu.layers import Normalizer, IsotopeNormalizer
from models.gpu.metrics import (
    MaxAbsoluteError,
    MeanAbsoluteMassLoss,
    cut_labels,
)
from training.utils import CustomTensorboard


class PhysicsLoss(keras.losses.Loss):
    def __init__(self, physics_weight, **kwargs):
        super(PhysicsLoss, self).__init__(**kwargs)
        self.physics_weight = physics_weight

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor):
        """
        Loss function is mae + physics_weight * physics_loss
        """
        # the y_true also store the correction factors in the second half
        channels = y_pred.shape[-1]
        y_true, corrections = (
            y_true[:, :channels],
            y_true[:, channels:],
        )
        mae_loss = keras.losses.mean_absolute_error(y_true, y_pred)
        total_mass = ops.sum(y_pred * corrections, axis=-1, keepdims=True)
        physics_loss = ops.abs(1 - total_mass)
        return (1 - self.physics_weight) * mae_loss + self.physics_weight * physics_loss

    def get_config(self):
        return dict(physics_weight=self.physics_weight)


class HyperModel(kt.HyperModel):
    def __init__(
        self,
        input_channels,
        output_channels,
        normalizer=False,
        retries=50,
        dataset_name=None,
        dataset_dir="datasets",
        **kwargs,
    ):
        super(HyperModel, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.normalizer = normalizer
        self.retries = retries
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

    def create_model(self, hp: kt.HyperParameters):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.input_channels,)))
        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                keras.layers.Dense(
                    units=hp.Int(f"layer_size_{i}", 32, 1024, 32),
                    activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
                )
            )
        model.add(
            keras.layers.Dense(
                units=self.output_channels,
                activation=hp.Choice("output_activation", ["leaky_relu"]),
            )
        )
        if self.normalizer:
            norm_loss_weight = hp.Float("norm_loss_weight", 0, 1)
            model.add(Normalizer(norm_loss_weight=norm_loss_weight))
            loss_function = "mae"
        else:
            physics_weight = hp.Float("physics_weight", 0, 1)
            loss_function = PhysicsLoss(physics_weight=physics_weight)

        learning_rate = hp.Float("learning_rate", 1e-4, 1e-3, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_function,
            metrics=[
                cut_labels(MaxAbsoluteError()),
                MeanAbsoluteMassLoss(),
                cut_labels(keras.metrics.MeanAbsoluteError()),
            ],
        )
        return model

    def build(self, hp: kt.HyperParameters):
        return self.create_model(hp)

    def fit(self, hp: kt.HyperParameters, model, *args, **kwargs):
        kwargs.get("callbacks", []).append(
            keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=1e-4)
        )
        return model.fit(
            *args,
            shuffle=True,
            batch_size=2**12,
            **kwargs,
        )


class MIMOHyperModel(kt.HyperModel):
    n = 0

    def __init__(
        self,
        input_channels,
        output_channels,
        normalizer=False,
        retries=50,
        dataset_name=None,
        dataset_dir="datasets",
        **kwargs,
    ):
        super(kt.HyperModel, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.retries = retries
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

    def create_model(self, hp: kt.HyperParameters):
        mlp = keras.Sequential(name="unnormed")
        mlp.add(keras.layers.Input(shape=(self.input_channels,)))
        activations = {
            0: ["tanh", "leaky_relu"],
            1: ["relu", "leaky_relu"],
            2: ["leaky_relu", "sigmoid"],
            3: ["tanh", "leaky_relu"],
            4: ["relu", "sigmoid", "leaky_relu"],
            5: ["sigmoid"],
        }
        for i in range(hp.Int("num_layers", 5, 5)):
            mlp.add(
                keras.layers.Dense(
                    units=hp.Int(f"layer_size_{i}", 32, 900, 32),
                    activation=hp.Choice(f"activation_{i}", activations[i]),
                )
            )
        mlp.add(
            keras.layers.Dense(
                units=self.output_channels,
                activation=hp.Choice("output_activation", ["leaky_relu", "sigmoid"]),
            )
        )
        normed_loss_weight = hp.Float("normed_weight", 0, 0.5)
        unnormed_loss_weight = 1 - normed_loss_weight
        ratio_normalizer = IsotopeNormalizer(name="normed")
        inputs = keras.layers.Input(shape=(self.input_channels,))
        c = ops.abs(ops.transpose(inputs[..., 1:-9]))
        matrix = ops.array(
            # [c_H, c_D, c_T_00, c_T_10, c_T_01, c_T_20, c_T_11, c_T_02, c_T_30, c_T_21, c_T_12, c_T_03]
            [
                [1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0],
                [0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            ],
            dtype=float,
        )
        tmp = matrix @ c
        ratios = tmp[0] / tmp[1]
        unnormed = mlp(inputs)
        normed = ratio_normalizer(unnormed, ratios)
        model = keras.Model(
            inputs=inputs,
            outputs={"normed": normed, "unnormed": unnormed},
            name=f"trial_{MIMOHyperModel.n}",
        )
        MIMOHyperModel.n += 1
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float("learning_rate", 1e-3, 1e-3, sampling="log")
            ),
            loss={
                "unnormed": hp.Choice("unnormed_loss", ["mse", "mae"]),
                "normed": hp.Choice("normed_loss", ["mae"]),
            },
            metrics={
                "unnormed": MaxAbsoluteError(),
                "normed": MaxAbsoluteError(),
            },
            loss_weights={
                "unnormed": unnormed_loss_weight,
                "normed": normed_loss_weight,
            },
        )
        return model

    def build(self, hp: kt.HyperParameters):
        return self.create_model(hp)

    def fit(self, hp: kt.HyperParameters, model, *args, **kwargs):
        # kwargs.get("callbacks", []).append(
        #     keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=1e-4)
        # )
        kwargs.get("callbacks", []).append(
            CustomTensorboard(log_dir="logs/momi3", name=model.name)
        )
        return model.fit(
            *args,
            shuffle=True,
            batch_size=2**14,
            **kwargs,
        )
