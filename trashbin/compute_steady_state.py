from scipy.optimize import fsolve

# Compute steady state
def compute_steady_state(f, x0, u, d, p):  

    def f_steady(x0,u,d,p):
        return f(0, x0, u, d, p)

    xs = fsolve(f_steady, x0, args=(u,d,p))

    return xs

