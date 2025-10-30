import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def solve_riccati(A:np.ndarray,R1:np.ndarray,R2:np.ndarray,C:np.ndarray) -> np.ndarray:

    A_in = A.copy()
    R1_in = R1.copy()
    R2_in = R2.copy()
    C_in = C.copy()

    # Changing to scipy notation: 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html
    A = A_in.T; B = A_in.T@C_in.T; S = R1_in@C_in.T; R = C_in@R1_in@C_in.T+R2_in; Q = R1_in

    Ps = sp.linalg.solve_discrete_are(A, B, Q, R, s=S)

    return Ps

def KalmanFilterUpdate(xt:float,ut:float,yt:float,wt:float,A:np.ndarray,B:np.ndarray,C:np.ndarray,P:np.ndarray,Q:np.ndarray,R:np.ndarray,stationary:bool=True) -> float:

    kappa = P@C.T@np.linalg.inv(C@P@C.T + R)

    if stationary:
        P = solve_riccati(A,Q,R,C)
    else:
        P = A@(P-kappa@C@P)+Q

    xtp1 = A@(xt+kappa@(yt-C@xt))+B@ut

    return xtp1,P







