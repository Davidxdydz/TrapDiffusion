import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras_tuner as kt
from training.utils import estimate_cpu_time
from keras import ops
from models.gpu.layers import Normalizer
from models.gpu.metrics import (
    MaxAbsoluteError,
    MeanAbsoluteMassLoss,
    cut_labels,
)


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
        max_cpu_time=50e-6,
        retries=50,
        dataset_name=None,
        dataset_dir="datasets",
        **kwargs,
    ):
        super(HyperModel, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.normalizer = normalizer
        self.max_cpu_time = max_cpu_time
        self.retries = retries
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

    def create_model(self, hp: kt.HyperParameters):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.input_channels,)))
        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                keras.layers.Dense(
                    units=hp.Int(f"layer_size_{i}", 32, 512, 32),
                    activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
                )
            )
        model.add(keras.layers.Dense(self.output_channels, activation="leaky_relu"))
        if self.normalizer:
            norm_loss_weight = hp.Float("norm_loss_weight", 0, 1)
            model.add(Normalizer(norm_loss_weight=norm_loss_weight))
            loss_function = "mae"
        else:
            physics_weight = hp.Float("physics_weight", 0, 1)
            loss_function = PhysicsLoss(physics_weight=physics_weight)

        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
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
        tries = 0
        while tries <= self.retries:
            tries += 1
            model = self.create_model(hp)
            cpu_time = estimate_cpu_time(model)
            if cpu_time < self.max_cpu_time:
                return model
            hp.values.clear()
        raise ValueError("Could not find a model that meets the time constraint")

    def fit(self, hp: kt.HyperParameters, model, *args, **kwargs):
        return model.fit(
            *args,
            #  epochs=hp.Int("epochs", 20, 40),
            shuffle=hp.Boolean("shuffle"),
            batch_size=hp.Int("batch_size", 2**11, 2**14, sampling="log"),
            **kwargs,
        )
