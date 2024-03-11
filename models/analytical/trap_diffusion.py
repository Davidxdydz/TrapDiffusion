import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


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
    def __init__(self, name, use_jacobian=False, t_final=2, fixed=False):
        self.t_final = t_final
        self.sol = None
        self.name = name
        self.use_jacobian = use_jacobian
        self.fixed = fixed

    def rhs(self, t, y):
        raise NotImplementedError("Subclass must implement abstract method")

    def jacobian(self, t, y):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_relevant_params(self):
        """
        Return an array containing all non redundant matrix entries.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def training_data(self, include_params=False, n_eval=None, initial_values=None, log_t_eval = False):
        """
        Generate training data from existing parameters but with random initial conditions.
        The format will be two arrays of
        [[t,c0_0,c0_1,..., <optional params>],] and [[ct_0,ct_1,...]]
        which correspond to the training data and the labels.
        """
        if initial_values is None:
            initial_values = self.initial_values()
        t, solutions = self.solve(initial_values, n_eval, log_t_eval)
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

    def solve(self, y0, n_eval, log_t_eval = False):
        t_eval = None
        if log_t_eval:
            t_eval = np.geomspace(1e-13,self.t_final,n_eval)
        else:
            t_eval = np.linspace(0, self.t_final, n_eval)


        if self.use_jacobian:
            sol = solve_ivp(
                fun=self.rhs,
                y0=y0,
                t_span=(0, self.t_final),
                t_eval=t_eval,
                jac=self.jacobian,
                method="Radau",
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

    def evaluate(
        self,
        model,
        include_params=False,
        n_eval=None,
        legend=True,
        pre_normalized=False,
        initial_values = None,
        log_t_eval = False
    ):
        """
        Evaluate the model with the given prediction function.
        """
        inputs, targets = self.training_data(
            n_eval=n_eval,
            include_params=include_params,
            initial_values = initial_values,
            log_t_eval = log_t_eval
        )
        predictions = model.predict(inputs)
        predictions = self.targets_reverse_transform(predictions)
        targets = self.targets_reverse_transform(targets)
        inputs = self.inputs_reverse_transform(inputs)

        corrections = self.correction_factors()
        if not pre_normalized:
            predictions *= corrections
        targets *= corrections
        delta = np.abs(targets - predictions)
        ts = inputs[:, 0]
        plt.figure(figsize=(20,10))
        main_axis = plt.gca()
        error_axis = main_axis.twinx()
        for index, description in self.vector_description.items():
            p = main_axis.plot(
                ts,
                targets[:, index],
                label=f"{description} - analytical",
                linewidth=2,
            )
            color = p[0].get_color()
            main_axis.scatter(
                ts,
                predictions[:, index],
                label=f"{description} - PINN",
                color=color,
                marker="x",
            )
            error_axis.plot(
                ts,
                delta[:, index],
                label=f"$|Error|$ of {description}",
                color=color,
                linewidth=0.5,
                linestyle="--",
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
