import numpy as np
from src.KalmanFilterUpdate import KalmanFilterUpdate
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def discrete_state_update(A,B,xt,ut,wt):
    return A@xt + B@ut + wt

def discrete_output_update(C,xt,vt):
    return C@xt + vt

np.random.seed(42)
data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
data_prob4 = np.load(r"Results\Problem4\Problem_4_estimates.npz")
A_est = data_prob5["A"]
B_est = data_prob5["B"]
C_est = data_prob5["C"]
Q = data_prob5["Q"]

x0, ut, d, p , R, R_d, delta_t = initialize()
Model_Stochastic = FourTankSystem(R_s=R*0.01, R_d=R_d*0, p=p, delta_t=delta_t)

P = 10*np.eye(A_est.shape[0])  # Initial estimate error covariance 

Tf = 100
N = int(Tf/delta_t)
t = np.arange(0, Tf, delta_t)
xt = np.concatenate((x0, d))  # Initial state with disturbances

X = np.zeros([N-1,4])
Z = np.zeros([N-1,2])
D = np.zeros([N-1,2])
U = np.zeros([N-1,2])
W = np.random.multivariate_normal(mean=np.zeros(A_est.shape[0]), cov=Q, size=N)
V = np.random.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R, size=N)
X_true = np.zeros([N, 4])  

linear = False
static = True

for t_idx,_ in enumerate(t[:-1]):

    # Save true state before simulating 
    X_true[t_idx, :] = xt[:-2]

    if linear:
        zt = discrete_output_update(C_est, xt, V[t_idx][:-2])
    else:
        yt = Model_Stochastic.StateSensor(xt[:-2])
        zt = yt[:2]

    xt_hat, P = KalmanFilterUpdate(xt, ut, zt, W[t_idx], A_est, B_est, C_est, P, Q, R[:2,:2], stationary=static)
    
    U[t_idx, :] = ut
    Z[t_idx, :] = zt        
    X[t_idx, :] = xt_hat[:-2]
    D[t_idx, :] = xt_hat[-2:]
    
    # Simulate next true state  
    if linear:
        xt = discrete_state_update(A_est, B_est, xt, ut, W[t_idx])
    else:
        f = Model_Stochastic.FullEquation
        xt = f(_, xt, ut)
        sol = solve_ivp(f, (t[t_idx], t[t_idx+1]), xt, method='RK45',args = (ut,))
        xt = sol.y[:,-1]

fig, ax = plt.subplots(4, 1, figsize=(12, 12))  
for i in range(4):
    ax[i].plot(t[:-1], X_true[:-1, i],'--', label='True State', color="black")
    ax[i].plot(t[:-1], X[:, i], '*-', label='Kalman Estimate', color="red")
    ax[i].set_title(f'$x_{{{i+1},t}}$')
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xticklabels([]) 
    ax[i].set_xlabel('') 
ax[i].legend()
ax[i].grid(True)
ax[i].set_xlabel('Time')     
fig.suptitle("Open loop of the discrete time system\n State estimation using Kalman filter", fontsize=16)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

fig, ax = plt.subplots(2, 1, figsize=(12, 12)) 
ax[0].plot(t[:-1],d[0]*np.ones(len(t[:-1])), label='$d_1$')
ax[0].plot(t[:-1],d[1]*np.ones(len(t[:-1])), label='$d_2$')
for i,VEC, label, title in zip(range(2),[D,U],["d","u"],["Disturbance Estimates","Control inputs"]):
    ax[i].plot(t[:-1],VEC[:,0],label=f"{label}$_1$")
    ax[i].plot(t[:-1],VEC[:,1],label=f"{label}$_2$")
    ax[i].set_title(title)
    ax[i].legend()
    ax[i].grid(True)
ax[i].set_xlabel('Time')
fig.suptitle("Control input and estimate of disturbance input", fontsize=16)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

plt.show()

