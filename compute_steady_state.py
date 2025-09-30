from scipy.optimize import fsolve
import numpy as np

# Compute steady state
def compute_steady_state(f, x0, u): 

    if x0.shape[0] == 4:
        x0 = np.concatenate([x0, np.zeros(2)]) 

    def f_steady(x0,u):
        return f(0, x0, u)

    xs = fsolve(f_steady, x0, args=(u))

    return xs
