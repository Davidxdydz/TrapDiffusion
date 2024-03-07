import keras
from keras import ops
import functools

keras.metrics.MeanAbsoluteError


class MaxAbsoluteError(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "max_ae"
        self.max_ae = -1

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        if sample_weight is not None:
            raise NotImplementedError("sample_weight not supported")
        ae = ops.abs(y_true - y_pred)
        new_max = ops.max(ae)
        if new_max > self.max_ae:
            self.max_ae = new_max.item()

    def result(self):
        return self.max_ae

    def reset_state(self):
        self.max_ae = -1


class MeanAbsoluteMassLoss(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "maml"
        self.mass_loss = 0
        self.count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        if sample_weight is not None:
            raise NotImplementedError("sample_weight not supported")
        output_channels = y_pred.shape[-1]
        pre_normalized = y_true.shape[-1] == output_channels
        if not pre_normalized:
            y_true, corrections = (
                y_true[..., :output_channels],
                y_true[..., output_channels:],
            )
            mass_loss = 1 - ops.sum(y_pred * corrections, axis=-1)
        else:
            mass_loss = 1 - ops.sum(y_pred, axis=-1)
        self.mass_loss += ops.sum(ops.abs(mass_loss))
        self.count += len(mass_loss)

    def result(self):
        return self.mass_loss / self.count

    def reset_state(self):
        self.mass_loss = 0
        self.count = 0


def cut_labels(metric):
    original_update_state = metric.update_state

    @functools.wraps(original_update_state)
    def wrapper(y_true, y_pred, sample_weight=None):
        output_channels = y_pred.shape[-1]
        y_true = y_true[..., :output_channels]
        return original_update_state(y_true, y_pred, sample_weight)

    metric.update_state = wrapper
    return metric
