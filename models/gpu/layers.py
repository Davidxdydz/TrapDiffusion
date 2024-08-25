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


class ConcentrationRatios(keras.layers.Layer):
    def call(self, inputs):
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
        return ratios


class IsotopeNormalizer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IsotopeNormalizer, self).__init__(**kwargs)

    def call(self, inputs, ratio):
        matrix = ops.array(
            # [c_H, c_D, c_T_00, c_T_10, c_T_01, c_T_20, c_T_11, c_T_02, c_T_30, c_T_21, c_T_12, c_T_03]
            #  0    1    2       3       4       5       6       7       8       9       10      11
            [
                [1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0],
                [0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            ],
            dtype=float,
        )

        c = inputs.T

        # renormalization
        c = ops.abs(c)
        new_ics = matrix @ c
        new_ratio = new_ics[0] / new_ics[1]
        mask = new_ratio < ratio
        if ops.any(mask):
            delta_H = (ratio[mask] * new_ics[1, mask] - new_ics[0, mask]) / 7
            # c[[0, 3, 5, 8], mask] += delta_H
            c[0, mask] += delta_H
            c[3, mask] += delta_H
            c[5, mask] += delta_H
            c[8, mask] += delta_H

        if ops.any(~mask):
            delta_D = (new_ics[0, ~mask] / ratio[~mask] - new_ics[1, ~mask]) / 7
            # c[[1, 4, 7, 11], ~mask] += delta_D
            c[1, ~mask] += delta_D
            c[4, ~mask] += delta_D
            c[7, ~mask] += delta_D
            c[11, ~mask] += delta_D
        final_ics = matrix @ c
        final_total = ops.sum(final_ics, axis=0)
        c /= final_total
        return c.T

    def get_config(self):
        return dict()
