from typing import overload
import numpy as np
from models.analytical.trap_diffusion import TrapDiffusion, hadamard
from training.parameter_range import ParameterRange
import matplotlib.pyplot as plt
from saveable import saveable


class MultiOccupationMultiIsotope(TrapDiffusion):
    f_range = ParameterRange((1e11, 1e14))
    E_range = ParameterRange((0.3, 1.9))
    T_range = ParameterRange((400.0, 1000))
    min_r = f_range.choices[0] * np.exp(
        -E_range.choices[1] * 11604 / T_range.choices[0]
    )
    max_r = f_range.choices[1] * np.exp(
        -E_range.choices[0] * 11604 / T_range.choices[1]
    )

    def set_params(self, random=False):
        if not random:
            self.T_to_S = np.array(
                [
                    3022.8272984145406,  # T10
                    3022.8272984145406,  # T01
                    347725598.62135780,  # T20
                    173862799.31067890,  # T11
                    347725598.62135780,  # T02
                    20133797667.950905,  # T30
                    13422531778.633938,  # T21
                    6711265889.3169689,  # T12
                    20133797667.950905,  # T03
                ]
            )
        else:
            # total: 1e3 - 1e11
            n = 9

            f = np.array(MultiOccupationMultiIsotope.f_range.random(n))
            E = np.array(
                sorted(MultiOccupationMultiIsotope.E_range.random(n), reverse=True)
            )
            T = MultiOccupationMultiIsotope.T_range.random()
            self.T_to_S = f * np.exp(-(E * 11604) / T)

            # f_0 = 1e12
            # E_0 = 0.29
            # r_0i = f_0 * np.exp(-E_0 * 11604 / T)
        self.e = 536918980.06994247
        self.d = 379659051.75522107

    def __init__(
        self,
        fixed=False,
        normalized_rs=None,
    ):
        TrapDiffusion.__init__(
            self,
            "Multi-Occupation, Multi-Isotope Model",
            True,
            1 if fixed else 1e7,
            fixed=fixed,
        )
        self.cS = 1
        self.set_params(random=not fixed)
        if normalized_rs is not None:
            self.T_to_S = MultiOccupationMultiIsotope.log_un_normalize(
                normalized_rs,
                MultiOccupationMultiIsotope.min_r,
                MultiOccupationMultiIsotope.max_r,
            )
        g = 1
        f = 1

        TS = {
            "10": self.T_to_S[0],
            "01": self.T_to_S[1],
            "20": self.T_to_S[2],
            "11": self.T_to_S[3],
            "02": self.T_to_S[4],
            "30": self.T_to_S[5],
            "21": self.T_to_S[6],
            "12": self.T_to_S[7],
            "03": self.T_to_S[8],
        }

        E = np.identity(12)
        E[0, 0] = 0
        E[1, 1] = 0

        E[0, 3] = -1 / self.cS
        E[0, 5] = -2 / self.cS
        E[0, 6] = -1 / self.cS
        E[0, 8] = -3 / self.cS
        E[0, 9] = -2 / self.cS
        E[0, 10] = -1 / self.cS

        E[1, 4] = -1 / self.cS
        E[1, 6] = -1 / self.cS
        E[1, 7] = -2 / self.cS
        E[1, 9] = -1 / self.cS
        E[1, 10] = -2 / self.cS
        E[1, 11] = -3 / self.cS

        A = np.zeros((2, 12, 12))
        A[0, 2, 3] = TS["10"]
        A[0, 3, 3] = -TS["10"]
        A[0, 3, 5] = TS["20"]
        A[0, 5, 5] = -TS["20"]
        A[0, 4, 6] = TS["11"]
        A[0, 6, 6] = -TS["11"]
        A[0, 5, 8] = TS["30"]
        A[0, 8, 8] = -TS["30"]
        A[0, 6, 9] = TS["21"]
        A[0, 9, 9] = -TS["21"]
        A[0, 7, 10] = TS["12"]
        A[0, 10, 10] = -TS["12"]

        A[1, 2, 4] = TS["01"]
        A[1, 4, 4] = -TS["01"]
        A[1, 3, 6] = TS["11"]
        A[1, 6, 6] = -TS["11"]
        A[1, 4, 7] = TS["02"]
        A[1, 7, 7] = -TS["02"]
        A[1, 5, 9] = TS["21"]
        A[1, 9, 9] = -TS["21"]
        A[1, 6, 10] = TS["12"]
        A[1, 10, 10] = -TS["12"]
        A[1, 7, 11] = TS["03"]
        A[1, 11, 11] = -TS["03"]

        H = np.zeros((2, 12, 12))
        H[0, 2, 2] = -1
        H[0, 3, 3] = -1
        H[0, 4, 4] = -1
        H[0, 5, 5] = -1
        H[0, 6, 6] = -1
        H[0, 7, 7] = -1

        H[0, 3, 2] = 1
        H[0, 5, 3] = 1
        H[0, 6, 4] = 1
        H[0, 8, 5] = 1
        H[0, 9, 6] = 1
        H[0, 10, 7] = 1

        H[0, 8, 9] = self.e
        H[0, 9, 9] = -self.e
        H[0, 10, 10] = -self.e
        H[0, 11, 11] = -self.e

        H[0, 8, 9] = self.e
        H[0, 9, 10] = self.e
        H[0, 10, 11] = self.e

        H[1, 2, 2] = -1
        H[1, 3, 3] = -1
        H[1, 4, 4] = -1
        H[1, 5, 5] = -1
        H[1, 6, 6] = -1
        H[1, 7, 7] = -1

        H[1, 4, 2] = 1
        H[1, 6, 3] = 1
        H[1, 7, 4] = 1
        H[1, 9, 5] = 1
        H[1, 10, 6] = 1
        H[1, 11, 7] = 1

        H[1, 8, 8] = -self.d
        H[1, 9, 9] = -self.d
        H[1, 10, 10] = -self.d

        H[1, 9, 8] = self.d
        H[1, 10, 9] = self.d
        H[1, 11, 10] = self.d

        M = np.zeros((2, 12, 12))
        M[0, 2:, 0] = g
        M[1, 2:, 1] = f

        self.E = E
        self.A = A
        self.H = H
        self.M = M

    # TODO move this method
    @staticmethod
    def log_normalize(val, min_val, max_val):
        min_log = np.log(min_val)
        max_log = np.log(max_val)
        val_log = np.log(val)
        return (val_log - min_log) / (max_log - min_log)

    @staticmethod
    def log_un_normalize(val, min_val, max_val):
        min_log = np.log(min_val)
        max_log = np.log(max_val)
        return np.exp(val * (max_log - min_log) + min_log)

    def rhs(self, t, c):
        return self.E @ (
            self.A[0] @ c
            + self.A[1] @ c
            + hadamard(self.H[0] @ c, self.M[0] @ c)
            + hadamard(self.H[1] @ c, self.M[1] @ c)
        )

    def jacobian(self, t, c):
        return self.E @ (
            self.A[0]
            + self.A[1]
            + hadamard(self.H[0], self.M[0] @ c)
            + hadamard(self.H[1], self.M[1] @ c)
            + hadamard(self.H[0] @ c, self.M[0])
            + hadamard(self.H[1] @ c, self.M[1])
        )

    def correction_factors(self):
        return np.array([1, 1, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    def get_relevant_params(self):
        return self.log_normalize(
            self.T_to_S,
            MultiOccupationMultiIsotope.min_r,
            MultiOccupationMultiIsotope.max_r,
        )

    def isotope_concentration_matrix(self):
        """
        M * c0 = (c_H_total, c_D_total)
        """
        return np.array(
            # [c_H, c_D, c_T_00, c_T_10, c_T_01, c_T_20, c_T_11, c_T_02, c_T_30, c_T_21, c_T_12, c_T_03]
            [
                [1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0],
                [0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            ]
        )

    def initial_values(self):
        """
        Return random plausible initial values for the concentration vector.
        """
        c = np.random.uniform(0, 1, 12)
        c /= np.sum(c * self.correction_factors())
        return c

    # @saveable(default_dir="report/figures/model_evaluation")
    def evaluate(
        self,
        model,
        include_params=False,
        n_eval=None,
        legend=True,
        initial_values=None,
        log_t_eval=False,
        plot_error=True,
        faulty=False,
        plot_masses=False,
    ):
        """
        Evaluate the model with the given prediction function.
        """
        inputs, targets = self.training_data(
            n_eval=n_eval,
            include_params=include_params,
            initial_values=initial_values,
            log_t_eval=log_t_eval,
            faulty=faulty,
        )
        predictions = model.predict(inputs)
        predictions = self.targets_reverse_transform(predictions)
        targets = self.targets_reverse_transform(targets)
        inputs = self.inputs_reverse_transform(inputs)

        delta = np.abs(targets - predictions)
        ts = inputs[:, 0]

        import matplotlib.gridspec as gridspec

        if plot_error:
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            plt.subplot(gs[0])
            main_axis = plt.gca()
            plt.subplot(gs[1], sharex=main_axis)
            error_axis = plt.gca()
        else:
            main_axis = plt.gca()
        for index, description in self.vector_description.items():
            p = main_axis.plot(
                ts,
                targets[:, index],
                # label=f"{description} - analytical",
                linewidth=2,
            )
            color = p[0].get_color()
            main_axis.scatter(
                ts,
                predictions[:, index],
                # label=f"{description} - PINN",
                color=color,
                marker="x",
            )
            if plot_error:
                error_axis.plot(
                    ts,
                    delta[:, index],
                    # label=f"$|Error|$ of {description}",
                    color=color,
                    # linewidth=0.5,
                    # linestyle="--",
                )

        plt.sca(main_axis)
        if plot_masses:
            self.plot_details(ts, predictions.T)
        plt.scatter([], [], color="black", label="Predicted Concentration", marker="x")
        plt.plot([], [], color="black", label="Numerical Solution")
        plt.legend()
        if plot_error:
            plt.setp(main_axis.get_xticklabels(), visible=False)
        else:
            plt.xlabel(self.xlabel)
        plt.ylabel(f"Concentration")

        plt.xscale(self.xscale)
        if legend:
            plt.legend(loc="center right")
        plt.grid()
        plt.tight_layout()

        if plot_error:
            plt.sca(error_axis)
            plt.grid()
            plt.xlabel(self.xlabel)
            plt.xscale(self.xscale)
            if plot_error:
                plt.ylabel(f"Absolute Error")
            if legend:
                plt.legend(loc="center left")
        plt.tight_layout()

    @property
    def vector_description(self):
        # c0 = [c_H, c_D, c_T_00, c_T_10, c_T_01, c_T_20, c_T_11, c_T_02, c_T_30, c_T_21, c_T_12, c_T_03]
        desc = {
            0: "$c^H$",
            1: "$c^D$",
            2: "$c^T_{00}$",
            3: "$c^T_{10}$",
            4: "$c^T_{01}$",
            5: "$c^T_{20}$",
            6: "$c^T_{11}$",
            7: "$c^T_{02}$",
            8: "$c^T_{30}$",
            9: "$c^T_{21}$",
            10: "$c^T_{12}$",
            11: "$c^T_{03}$",
        }
        return desc

    def plot_details(self, t, y):
        # self.plot_total(t, y, correct=False, label="total")
        isotope_concentrations = self.isotope_concentration_matrix() @ y
        plt.plot(t, isotope_concentrations[0], label="$c_{H,total}$", linestyle="--")
        plt.plot(t, isotope_concentrations[1], label="$c_{D,total}$", linestyle="--")

    def plot(self, *args, **kwargs):
        if "log_t_eval" not in kwargs:
            kwargs["log_t_eval"] = True
        if "pre_normalized" not in kwargs:
            kwargs["pre_normalized"] = True
        return TrapDiffusion.plot(self, *args, **kwargs)

    @property
    def y_unit(self):
        return "$\\left[\\frac{trap-sites}{lattice-sites}\\right]$"

    @property
    def xscale(self):
        return "log"

    def inputs_transform(self, inputs):
        # 13 is empirical
        inputs[:, 0] = MultiOccupationMultiIsotope.log_normalize(
            inputs[:, 0], 1e-13, self.t_final
        )
        return inputs

    def inputs_reverse_transform(self, inputs):
        inputs[:, 0] = MultiOccupationMultiIsotope.log_un_normalize(
            inputs[:, 0], 1e-13, self.t_final
        )
        return inputs
