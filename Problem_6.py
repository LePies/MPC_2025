import numpy as np
from src.KalmanFilterUpdate import KalmanFilterUpdate
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.load(r"Results\Problem5\Problem_5_estimates.npz")
A_est = data["A"]
B_est = data["B"]
C_est = data["C"]
Q = data["Q"]

x0, ut, d, p , R, R_d, delta_t = initialize()
Model_Stochastic = FourTankSystem(R_s=R*2, R_d=R_d*0, p=p, delta_t=delta_t)

P = np.eye(A_est.shape[0])  # Initial estimate error covariance 

Tf = 100 
N = int(Tf/delta_t)
t = np.arange(0, Tf, delta_t)
xt = np.concatenate((x0, d))  # Initial state with disturbances

X = np.zeros([N,4])
Y = np.zeros([N,4])
D = np.zeros([N,2])
U = np.zeros([N,2])
W = np.random.multivariate_normal(mean=np.zeros(A_est.shape[0]), cov=Q, size=N)
X_true = np.zeros([N, 4])  

for t_idx,_ in enumerate(t):

    # Save true state before simulating 
    X_true[t_idx, :] = xt[:-2]

    yt = Model_Stochastic.StateSensor(xt[:-2])
    zt = yt[:2]
    xt_hat, P = KalmanFilterUpdate(xt, ut, zt, W[t_idx], A_est, B_est, C_est, P, Q, R[:2,:2], stationary=False)
    
    U[t_idx, :] = ut
    Y[t_idx, :] = yt        
    X[t_idx, :] = xt_hat[:-2]
    D[t_idx, :] = xt_hat[-2:]
    
    # Simulate next true state  
    xt = Model_Stochastic.FullEquation(_, xt, ut)

fig, ax = plt.subplots(4, 1, figsize=(12, 12))  # 4 states
for i in range(2):
    ax[i].plot(t, X_true[:, i],'--', label='True State', color="black")
    ax[i].plot(t, X[:, i], '*-', label='Kalman Estimate', color="red")
    ax[i].set_title(f'$x_{{{i+1},t}}$')
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xticklabels([])  # Hide x-axis tick labels for all but the last subplot
    ax[i].set_xlabel('')       # Optionally also remove x-axis label if set elsewhere


ax[i+1].plot(t,U[:,0],label="control input $u_1$")
ax[i+1].plot(t,U[:,1],label="control input $u_2$")
ax[i+1].set_title(f'Control input')
ax[i+1].legend()
ax[i+1].grid(True)
ax[i+1].set_xlabel('Time')
fig.suptitle("Open loop and state estimation using Kalman filter", fontsize=16)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

plt.figure()
plt.plot(t,Y[:,0],label="measured output $y_1$")
plt.show()