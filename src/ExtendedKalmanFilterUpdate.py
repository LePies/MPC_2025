import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def ExtendedKalmanFilterUpdate(xt, yt, f_func, sigma, h_func, F_jacobian_func, H_jacobian_func, P, Rv, Re):


    A = F_jacobian_func(xt)
    
    

    # Predict state
    xt = sp.linalg.solve_continuous_lyapunov(A,Q)

    # Update Step 
    S = H @ Pt_pred @ H.T + Re
    Kt = Pt_pred @ H.T / S 
    y_pred = h_func(xt_pred).flatten()
    residual = (yt - y_pred)
    xt_upd = xt_pred + Kt @ residual
    Pt_upd = (I - Kt @ H) @ Pt_pred

    return xt_upd.flatten(), Pt_upd, Kt















