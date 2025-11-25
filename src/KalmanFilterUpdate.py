import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def solve_riccati(A,C,R,Q) -> np.ndarray:

    # Changing to scipy notation: 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html
    
    A_scipy = A.T
    B_scipy = C.T

    Ps = sp.linalg.solve_discrete_are(A_scipy, B_scipy, Q, R)

    return Ps

def KalmanFilterUpdate_old(
    xt:float,
    ut:float,
    yt:float,
    A:np.ndarray,
    B:np.ndarray,
    C:np.ndarray,
    P:np.ndarray,
    Q:np.ndarray,
    R:np.ndarray,
    stationary:bool=True
) -> float:

    kappa = P@C.T@np.linalg.inv(C@P@C.T + R)

    if stationary:
        try:
            P = solve_riccati(A,C,R,Q)
        except np.linalg.LinAlgError:
            # Fall back to non-stationary update if DARE solver fails
            # (e.g., when system has eigenvalues on unit circle)
            P = A@(P-kappa@C@P)@A.T+Q
    else:
        P = A@(P-kappa@C@P)@A.T+Q

    xtp1 = A@(xt+kappa@(yt-C@xt))+B@ut #+E@dt

    return xtp1, P


def KalmanFilterUpdate(
    xt:float,
    ut:float,
    yt:float,
    A:np.ndarray,
    B:np.ndarray,
    C:np.ndarray,
    P:np.ndarray,
    Q:np.ndarray,
    R:np.ndarray,
    stationary:bool=True
) -> float:

    kappa = P@C.T@np.linalg.inv(C@P@C.T + R)

    if stationary:
        try:
            P = solve_riccati(A,C,R,Q)
        except np.linalg.LinAlgError:
            # Fall back to non-stationary update if DARE solver fails
            # (e.g., when system has eigenvalues on unit circle)
            P = A@(P-kappa@C@P)@A.T+Q
    else:
        P = A@(P-kappa@C@P)@A.T+Q

    xtp1 = A@xt+B@ut # predict 
    xtp1 = xtp1 + kappa@(yt - C@xtp1) # update

    return xtp1, P





