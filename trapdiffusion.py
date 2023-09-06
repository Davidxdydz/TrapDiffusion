import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def hadamard(A,B):
    n_dim_a = len(A.shape)
    n_dim_b = len(B.shape)
    if n_dim_a > n_dim_b:
        return B[:,None]* A
    elif n_dim_a < n_dim_b:
        return A[:,None]* B
    else:
        return A*B

class TrapDiffusion:
    def __init__(self, name):
        self.t_final = 2
        self.sol = None
        self.name = name
    
    def rhs(self,t,y):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def jacobian(self,t,y):
        raise NotImplementedError("Subclass must implement abstract method")

    def solve(self,y0):
        self.sol = solve_ivp(fun = self.rhs, y0 = y0, t_span = (0, self.t_final), jac = self.jacobian)


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
            plt.plot(self.sol.t,self.sol.y[key], label = value)
        
        self.plot_details()
        plt.legend()
        plt.ylabel("Concentration")
        plt.xlabel("Time")
        plt.title(self.name)
        plt.grid()