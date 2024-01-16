import numpy as np
from models.analytical.trap_diffusion import TrapDiffusion, hadamard
from training.utils import ParameterRange


class MultiOccupationMultiIsotope(TrapDiffusion):
    T_to_S_ranges = [
        ParameterRange(1e3, 1e4),  # T10
        ParameterRange(1e3, 1e4),  # T01
        ParameterRange(100e6, 100e7),  # T20
        ParameterRange(100e6, 100e7),  # T11
        ParameterRange(100e6, 100e7),  # T02
        ParameterRange(10e9, 10e10),  # T30
        ParameterRange(10e9, 10e10),  # T21
        ParameterRange(1e9, 1e10),  # T12
        ParameterRange(10e9, 10e10),  # T03
    ]
    e_range = ParameterRange(100e6, 100e7)
    d_range = ParameterRange(100e6, 100e7)

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
            self.e = 536918980.06994247
            self.d = 379659051.75522107
        else:
            # total: 1e3 - 1e11
            # norm: (log10(x) - 3)/ 8

            self.T_to_S = np.array(
                [r.random() for r in MultiOccupationMultiIsotope.T_to_S_ranges]
            )
            self.e = MultiOccupationMultiIsotope.e_range.random()
            self.d = MultiOccupationMultiIsotope.d_range.random()

    def __init__(self, fixed=False):
        TrapDiffusion.__init__(
            self, "Multi-Occupation, Multi-Isotope Model", True, 0.2, fixed=fixed
        )
        self.cS = 1
        self.set_params(random=not fixed)
        g = 1

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
        M[1, 2:, 1] = g

        self.E = E
        self.A = A
        self.H = H
        self.M = M

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
        relevant_entries = []

        for param, param_range in zip(
            self.T_to_S, MultiOccupationMultiIsotope.T_to_S_ranges
        ):
            relevant_entries.append(param_range.log_normalize(param))

        relevant_entries.append(
            MultiOccupationMultiIsotope.e_range.log_normalize(self.e)
        )
        relevant_entries.append(
            MultiOccupationMultiIsotope.d_range.log_normalize(self.d)
        )
        return np.array(relevant_entries)

    def initial_values(self):
        """
        Return random plausible initial values for the concentration vector.
        """
        c = np.random.uniform(0, 1, 12)
        c /= np.sum(c * self.correction_factors())
        return c

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
        self.plot_total(t, y, correct=True, label="total")

    @property
    def y_unit(self):
        return "$\\left[\\frac{trap-sites}{lattice-sites}\\right]$"

    @property
    def xscale(self):
        return "log"

    def inputs_transform(self, inputs):
        # 13 is empirical
        inputs[:, 0] = (np.log10(inputs[:, 0]) + 13) / 13
        return inputs

    def inputs_reverse_transform(self, inputs):
        inputs[:, 0] = 10 ** (inputs[:, 0] * 13 - 13)
        return inputs
