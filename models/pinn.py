import os

os.environ["KERAS_BACKEND"] = "torch"
import keras


class Normalizer(keras.layers.Layer):
    def __init__(self, norm_loss_weight=0, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.norm_loss_weight = norm_loss_weight

    def call(self, inputs):
        sums = keras.ops.sum(inputs, axis=1, keepdims=True)
        self.add_loss(keras.ops.abs(1 - sums) * self.norm_loss_weight)
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

    def from_config(config):
        return PhysicsLoss(**config)


def build_model(
    intput_channels,
    output_channels,
    layer_sizes,
    activations,
    output_activation,
    name,
    optimizer,
    loss_function,
    normalizer=False,
):
    model = keras.Sequential(name=name)
    model.add(keras.layers.Input(shape=(intput_channels,)))
    for size, activation in zip(layer_sizes, activations):
        model.add(keras.layers.Dense(size, activation=activation))
    model.add(keras.layers.Dense(output_channels, activation=output_activation))
    if normalizer:
        model.add(Normalizer())
    model.compile(optimizer=optimizer, loss=loss_function)
    return model
