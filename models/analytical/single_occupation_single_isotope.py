import numpy as np
from models.analytical.trap_diffusion import TrapDiffusion, hadamard
import matplotlib.pyplot as plt
from training.parameter_range import ParameterRange


class SingleOccupationSingleIsotope(TrapDiffusion):
    def __init__(self, n_traps=2, t_final=2, fixed=False):
        TrapDiffusion.__init__(
            self, "Single-Occupation, Single-Isotope Model", False, t_final, fixed=fixed
        )
        if fixed:
            np.random.seed(1)
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

    def inputs_transform(self, inputs):
        # normalize time to be from 0 to 1
        inputs[:, 0] = inputs[:, 0] / self.t_final
        return inputs

    def inputs_reverse_transform(self, inputs):
        inputs[:, 0] = inputs[:, 0] * self.t_final
        return inputs
