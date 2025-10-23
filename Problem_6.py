import numpy as np
from src.KalmanFilterUpdate import KalmanFilterUpdate
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem

data = np.load(r"Results\Problem5\Problem_5_estimates.npz")
A_est = data["A"]
B_est = data["B"]
C_est = data["C"]

x0, u, d, p , R_s, R_d, delta_t = initialize()
Model_Stochastic = FourTankSystem(R_s*0, R_d*0, p, delta_t)

P = np.eye(A_est.shape[0])  # Initial estimate error covariance


for t_idx,_ in enumerate(t):
    yt = Model_Stochastic.StateSensor(xt)
    zt = yt[:2]
    xt, P = KalmanFilterUpdate(xt, ut, yt, A_est, B_est, C_est, P, R_s, R_d)
    U[t_idx, :] = ut
    Y[t_idx, :] = yt        
    X[t_idx, :] = xt[:-2]
    D[t_idx,:] = xt[-2:]        
    # Save true state before simulating
    X_true[t_idx, :] = xt

    