import numpy as np

class HankelSystem:
    """
    Hankel system is a system of the form:
    x_dot = A*x + B*u
    y = C*x + D*u
    """
    
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    
    def SteadyState(self, u0):
        return np.linalg.solve(self.A - self.B*self.C, self.B*u0)
    