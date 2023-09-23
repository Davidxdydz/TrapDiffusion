import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def hadamard(A, B):
    n_dim_a = len(A.shape)
    n_dim_b = len(B.shape)
    if n_dim_a > n_dim_b:
        return B[:, None] * A
    elif n_dim_a < n_dim_b:
        return A[:, None] * B
    else:
        return A * B


class TrapDiffusion:
    def __init__(self, name, use_jacobian=False, t_final=2):
        self.t_final = t_final
        self.sol = None
        self.name = name
        self.use_jacobian = use_jacobian

    def rhs(self, t, y):
        raise NotImplementedError("Subclass must implement abstract method")

    def jacobian(self, t, y):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_relevant_params(self):
        """
        Return an array containing all non redundant matrix entries.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def training_data(self, include_params=False, n_eval=None, initial_values=None):
        """
        Generate training data from existing parameters but with random initial conditions.
        The format will be two arrays of
        [[t,c0_0,c0_1,..., <optional params>],] and [[ct_0,ct_1,...]]
        which correspond to the training data and the labels.
        """
        if initial_values is None:
            initial_values = self.initial_values()
        t, solutions = self.solve(initial_values, n_eval)
        tiled_initial_values = np.tile(initial_values, (len(t), 1))
        if include_params:
            relevant_params = self.get_relevant_params()
            tiled_relevant_params = np.tile(relevant_params, (len(t), 1))
            x = np.hstack([t[:, None], tiled_initial_values, tiled_relevant_params])
        else:
            x = np.hstack([t[:, None], tiled_initial_values])
        y = solutions.T
        x = self.inputs_transform(x)
        y = self.targets_transform(y)
        return x, y

    def inputs_transform(self, inputs):
        return inputs

    def targets_transform(self, targets):
        return targets

    def inputs_reverse_transform(self, inputs):
        return inputs

    def targets_reverse_transform(self, targets):
        return targets

    def solve(self, y0, n_eval=None):
        t_eval = None
        if n_eval is not None:
            t_eval = np.linspace(0, self.t_final, n_eval)

        if self.use_jacobian:
            sol = solve_ivp(
                fun=self.rhs,
                y0=y0,
                t_span=(0, self.t_final),
                t_eval=t_eval,
                jac=self.jacobian,
                method="BDF",
            )
        else:
            sol = solve_ivp(
                fun=self.rhs,
                y0=y0,
                t_span=(0, self.t_final),
                t_eval=t_eval,
            )
        # t = 0 is the initial value, so we can drop it
        return sol.t[1:], sol.y[:, 1:]

    def plot_details(self, t, y):
        ...

    def initial_values(self):
        raise NotImplementedError("Subclass must implement abstract method")

    @property
    def vector_description(self):
        raise NotImplementedError("Subclass must implement abstract method")

    @property
    def y_unit(self):
        """
        The unit of the y axis, like $[s]$
        """
        return ""

    @property
    def xlabel(self):
        return "Time $[s]$"

    @property
    def xscale(self):
        return "linear"

    def correction_factors(self):
        """
        Return an array size of the solution vector that contains the correction factors for each entry.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def plot(self, t=None, y=None):
        if y is None or t is None:
            initial_values = self.initial_values()
            t, y = self.solve(initial_values)

        plt.figure()
        correction_factors = self.correction_factors()
        for key, value in self.vector_description.items():
            # to get the concentration in H/lattice site we have to multiply with the trap/solute concentrations c_S_T
            # otherwise this would be h per solute-site/ trap-site
            plt.plot(t, y[key] * correction_factors[key], label=value)

        self.plot_details(t, y)
        plt.legend(loc="center right")
        plt.ylabel(f"Concentration {self.y_unit}")
        plt.xlabel(self.xlabel)
        plt.xscale(self.xscale)
        plt.title(self.name)
        plt.grid()
        plt.tight_layout()

    def plot_total(self, t, y, correct=False, label="total concentration"):
        if correct:
            y *= self.correction_factors()[:, None]

        # total h per lattice site has to stay constant
        total = np.sum(y, axis=0)
        plt.plot(t, total, label=label, color="red", linewidth=2, linestyle=":")

    def evaluate(self, model, include_params=False, n_eval=None, legend=True):
        """
        Evaluate the model with the given prediction function.
        """
        inputs, targets = self.training_data(
            n_eval=n_eval, include_params=include_params
        )
        predictions = model.predict(inputs)
        predictions = self.targets_reverse_transform(predictions)
        targets = self.targets_reverse_transform(targets)
        inputs = self.inputs_reverse_transform(inputs)

        corrections = self.correction_factors()
        predictions *= corrections
        targets *= corrections
        delta = np.abs(targets - predictions)
        ts = inputs[:, 0]
        main_axis = plt.gca()
        error_axis = main_axis.twinx()
        for key, value in self.vector_description.items():
            p = main_axis.plot(
                ts,
                targets[:, key],
                label=f"{value} - analytical",
                linestyle="--",
                linewidth=2,
            )
            color = p[0].get_color()
            main_axis.plot(
                ts,
                predictions[:, key],
                label=f"{value} - PINN",
                color=color,
                marker="x",
            )
            error_axis.plot(
                ts,
                delta[:, key],
                label=f"$|Error|$ of {value}",
                color=color,
                linewidth=0.5,
            )
        plt.sca(main_axis)
        plt.ylabel(f"Concentration {self.y_unit}")
        plt.xlabel(self.xlabel)
        plt.xscale(self.xscale)
        plt.title(f"{model.name}")
        self.plot_total(ts, predictions.T)
        if legend:
            plt.legend(loc="center right")
        plt.grid()
        plt.tight_layout()

        plt.sca(error_axis)
        plt.xscale(self.xscale)
        plt.ylabel(f"Absolute Error {self.y_unit}")
        if legend:
            plt.legend(loc="center left")
        plt.tight_layout()


class SingleOccupationSingleIsotope(TrapDiffusion):
    def __init__(self, n_traps=2, t_final=2):
        TrapDiffusion.__init__(
            self, "Single-Occupation, Single-Isotope Model", False, t_final
        )
        self.n_traps = n_traps
        # concentraion of trap sites and solute sites.
        # c_S_T = [c_S, c_T_1, c_T_2, ...]

        self.c_S_T = np.random.random(n_traps + 1)
        self.c_S_T = self.c_S_T / np.sum(self.c_S_T)
        c_S = self.c_S_T[0]

        # site concentration matrix
        self.C = np.diag(self.c_S_T)

        # max trap conentration, has to be greater than current concentration
        self.c_Max = np.random.random(n_traps)
        self.c_Max = self.c_Max * 0.5

        # capture cross-section of trap-site
        self.sigma = 1

        # base transition rates
        self.a = np.random.random((n_traps + 1, n_traps + 1))

        # transition rate matrix
        # trap to trap terms on the diagonal, (0,0) will be overwritten
        self.A_tilde = np.diag(-self.a[0, :])

        # solute's losses (0,0)
        # sigma is constant for all traps at the moment, so can be factored out
        self.A_tilde[0, 0] = -self.sigma * np.sum(self.a[1:, 0] * self.c_S_T[1:])

        # trap's gains from solute (first column)
        self.A_tilde[1:, 0] = self.a[1:, 0] * self.c_S_T[1:] * self.sigma

        # solutes gain from traps (first row)
        self.A_tilde[0, 1:] = self.a[0, 1:]

        self.A = self.A_tilde @ self.C

        self.B = np.zeros((n_traps + 1, n_traps + 1))

        # first row
        self.B[0, 1:] = self.a[1:, 0] * self.c_S_T[1:] * c_S * self.sigma / self.c_Max
        # fill in anty-symmetric part
        self.B[:, 0] = -self.B[0, :]

    @property
    def y_unit(self):
        return "$\\left[\\frac{H-atoms}{trap-sites}\\right]$"

    def get_relevant_params(self):
        relevant_entries = []

        # trap concentrations
        relevant_entries.extend(self.c_S_T)

        # first column and row of A, diagonal is redundant
        relevant_entries.extend(self.A[0, :])
        relevant_entries.extend(self.A[1:, 0])

        # first row of B, first column is redundant as B is antisymmetric
        relevant_entries.extend(self.B[0, 1:])
        return np.array(relevant_entries)

    def initial_values(self):
        """
        Return random plausible initial values for the concentration vector.
        """
        c = np.random.random(self.n_traps + 1)
        c[1:] *= self.c_Max
        c[0] = (1 - np.sum(c[1:] * self.c_S_T[1:])) / self.c_S_T[0]
        return c

    def correction_factors(self):
        return self.c_S_T

    def rhs(self, t, c):
        return hadamard(1 / self.c_S_T, self.A @ c + hadamard(c, self.B @ c))

    def jacobian(self, t, c):
        return hadamard(1 / self.c_S_T, self.A + self.B @ c + hadamard(c, self.B))

    @property
    def vector_description(self):
        desc = {
            0: "$c_Sc^S$",
        }
        for i in range(1, self.n_traps + 1):
            desc[i] = f"$c_{{T_{i-1}}}c^T_{i}$"
        return desc

    def plot_details(self, t, y):
        for i, (cm, ct) in enumerate(zip(self.c_Max, self.c_S_T[1:])):
            # cm : max concentration for trap i in H/trap-site
            # ct : concentration of trapsites i in H/lattice-site
            # to get the total concentration of trap i in H/lattice-site we have to multiply cm * ct
            plt.hlines([cm * ct], 0, t[-1], linestyles="dashed", color="black")
            plt.text(t[-1], cm * ct, f"$c^{{Max}}_{{T_{i+1}}}$")

        # sol.y contains c_s, c_t_1, c_t_2, ...
        # c_s is h per solute-site
        # c_t_i is h per trap-site
        # again, to get concentration in H/lattice site we have to multiply with the trap/solute concentrations c_S_T
        self.plot_total(
            t, y, correct=True, label="$c_sc^S+\\sum_{i}c_{T_i}\\cdot c_i^T$"
        )


class MultiOccupationMultiIsotope(TrapDiffusion):
    def __init__(self):
        TrapDiffusion.__init__(self, "Multi-Occupation, Multi-Isotope Model", True, 0.2)
        self.cS = 1
        T_to_S = np.array(
            [
                3022.8272984145406,  # T10
                3022.8272984145406,  # T01
                347725598.62135780,  # T20
                173862799.31067890,  # T11
                347725598.62135780,  # T02
                20133797667.950905,  # T30
                13422531778.633938,  # T21
                6711265889.3169689,  # T12
                20133797667.95090,  # T03
            ]
        )
        e = 536918980.06994247
        d = 379659051.75522107
        g = 1

        TS = {
            "10": T_to_S[0],
            "01": T_to_S[1],
            "20": T_to_S[2],
            "11": T_to_S[3],
            "02": T_to_S[4],
            "30": T_to_S[5],
            "21": T_to_S[6],
            "12": T_to_S[7],
            "03": T_to_S[8],
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

        H[0, 8, 9] = e
        H[0, 9, 9] = -e
        H[0, 10, 10] = -e
        H[0, 11, 11] = -e

        H[0, 8, 9] = e
        H[0, 9, 10] = e
        H[0, 10, 11] = e

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

        H[1, 8, 8] = -d
        H[1, 9, 9] = -d
        H[1, 10, 10] = -d

        H[1, 9, 8] = d
        H[1, 10, 9] = d
        H[1, 11, 10] = d

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
