import keras
from keras import ops


class Normalizer(keras.layers.Layer):
    def __init__(self, norm_loss_weight=0, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.norm_loss_weight = norm_loss_weight

    def call(self, inputs):
        sums = ops.sum(ops.abs(inputs), axis=-1, keepdims=True)
        self.add_loss(ops.mean(ops.abs(1 - sums)) * self.norm_loss_weight)
        result = inputs / sums
        return result

    def get_config(self):
        return dict(norm_loss_weight=self.norm_loss_weight)
