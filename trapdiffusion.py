import numpy as np
from scipy.integrate import odeint
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
        self.ts = np.linspace(0, 2, 1000)
        self.sol = None
        self.name = name
    
    def rhs(self,y,t):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def jacobian(self,y,t):
        raise NotImplementedError("Subclass must implement abstract method")

    def solve(self,y0):
        self.sol = odeint(self.rhs, y0, self.ts, Dfun=self.jacobian)

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
            plt.plot(self.ts,self.sol[:,key], label = value)
        
        self.plot_details()
        plt.legend()
        plt.ylabel("Concentration")
        plt.xlabel("Time")
        plt.title(self.name)
        plt.grid()