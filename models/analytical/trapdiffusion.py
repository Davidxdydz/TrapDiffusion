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
        return A*B


class TrapDiffusion:
    def __init__(self, n_traps, name):
        self.t_final = 2
        self.sol = None
        self.n_traps = n_traps
        self.name = name

    def rhs(self, t, y):
        raise NotImplementedError("Subclass must implement abstract method")

    def jacobian(self, t, y):
        raise NotImplementedError("Subclass must implement abstract method")

    def solve(self, y0, n_eval=None):
        t_eval = None
        if n_eval is not None:
            t_eval = np.linspace(0, self.t_final, n_eval)
        self.sol = solve_ivp(fun=self.rhs, y0=y0, t_span=(
            0, self.t_final), t_eval=t_eval)  # , jac = self.jacobian
        return self.sol.t, self.sol.y

    def plot_details(self):
        ...

    @property
    def vector_description(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def plot(self):
        if self.sol is None:
            self.solve(self.c)

        plt.figure()
        for key, value in self.vector_description.items():
            # to get the concentration in H/lattice site we have to multiply with the trap/solute concentrations c_S_T
            # otherwise this would be h per solute-site/ trap-site
            plt.plot(self.sol.t, self.sol.y[key] * self.c_S_T[key], label=value)

        self.plot_details()
        plt.legend()
        plt.ylabel("Concentration [$\\frac{H-atoms}{lattice-sites}$]")
        plt.xlabel("Time [$s$]")
        plt.title(self.name)
        plt.grid()


class SingleOccupationSingleIsotope(TrapDiffusion):
    def __init__(self, n_traps=2, t_final=None):
        TrapDiffusion.__init__(
            self, n_traps, "Single-Occupation, Single-Isotope Model")
        if t_final is not None:
            self.t_final = t_final
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
        self.a = np.random.random((n_traps+1, n_traps+1))

        # transition rate matrix
        # trap to trap terms on the diagonal, (0,0) will be overwritten
        self.A_tilde = np.diag(-self.a[0, :])

        # solute's losses (0,0)
        # sigma is constant for all traps at the moment, so can be factored out
        self.A_tilde[0, 0] = -self.sigma * \
            np.sum(self.a[1:, 0] * self.c_S_T[1:])

        # trap's gains from solute (first column)
        self.A_tilde[1:, 0] = self.a[1:, 0] * self.c_S_T[1:] * self.sigma

        # solutes gain from traps (first row)
        self.A_tilde[0, 1:] = self.a[0, 1:]

        self.A = self.A_tilde @ self.C

        self.B = np.zeros((n_traps+1, n_traps+1))

        # first row
        self.B[0, 1:] = self.a[1:, 0] * \
            self.c_S_T[1:] * c_S * self.sigma / self.c_Max
        # fill in anty-symmetric part
        self.B[:, 0] = -self.B[0, :]

        # random start concetrations
        self.c = self.initial_values()

    def get_relevant_params(self):
        """
        Return a list containing all non redundant matrix entries.
        """
        relevant_entries = []

        # trap concentrations
        relevant_entries.extend(self.c_S_T)

        # first column and row of A, diagonal is redundant
        relevant_entries.extend(self.A[0, :])
        relevant_entries.extend(self.A[1:, 0])

        # first row of B, first column is redundant as B is antisymmetric
        relevant_entries.extend(self.B[0, 1:])
        return relevant_entries

    def initial_values(self):
        """
        Return random plausible initial values for the concentration vector.
        """
        c = np.random.random(self.n_traps+1)
        c[1:] *= self.c_Max
        c[0] = (1 - np.sum(c[1:]* self.c_S_T[1:])) / self.c_S_T[0]
        return c

    def rhs(self, t, c):
        return hadamard(1/self.c_S_T, self.A@c + hadamard(c, self.B@c))

    def jacobian(self, t, c):
        return hadamard(1/self.c_S_T, self.A + self.B@c + hadamard(c, self.B))

    @property
    def vector_description(self):
        desc = {
            0: "$c_Sc_S$",
        }
        for i in range(1, len(self.c)):
            desc[i] = f"$c_{{T_{i-1}}}c^T_{i}$"
        return desc

    def plot_details(self):
        for i, (cm, ct) in enumerate(zip(self.c_Max, self.c_S_T[1:])):
            # cm : max concentration for trap i in H/trap-site
            # ct : concentration of trapsites i in H/lattice-site
            # to get the total concentration of trap i in H/lattice-site we have to multiply cm * ct
            plt.hlines([cm * ct], 0, self.t_final,
                       linestyles="dashed", color="black")
            plt.text(self.t_final, cm * ct, f"$c^{{Max}}_{{T_{i+1}}}$")
        
        # sol.y contains c_s, c_t_1, c_t_2, ...
        # c_s is h per solute-site
        # c_t_i is h per trap-site
        # again, to get concentration in H/lattice site we have to multiply with the trap/solute concentrations c_S_T
        corrected = self.sol.y
        corrected *= self.c_S_T[:,None]

        # total h per lattice site has to stay constant
        total = np.sum(corrected, axis=0)
        plt.plot(self.sol.t, total, label="$c_sc^S+\\sum_{i}c_{T_i}\\cdot c_i^T$")
