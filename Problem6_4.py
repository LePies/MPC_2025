import numpy as np
from src.KalmanFilterUpdate import KalmanFilterUpdate
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.FourTankSystem import FourTankSystem

def discrete_state_update(A,B,E,xt,ut,dt,wt):
    return A@xt + B@ut + E@dt + wt

def discrete_output_update(C,xt,vt):
    return C@xt + vt

np.random.seed(42)
data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
data_prob4 = np.load(r"Results\Problem4\Problem_4_estimates.npz")

prob = 4  
if prob == 4:
    data = data_prob4
elif prob == 5: 
    data = data_prob5

A_est = data["A"]
B_est = data["B"]
C_est = data["C"]
Q = data_prob5["Q"]

if prob == 4:
    Q1=Q[:2,:2]
    Q2 = Q[4:,:2]
    Q3 = Q[:2,4:]
    Q4 =Q[4:,4:]
    Q = np.block([
    [Q1, Q2],
    [Q3, Q4]
])
   
Ts = 1

def F3_func(t):
    if t < 250:
        return 100
    else:
        return 50
    return 100
    
def F4_func(t):
    if t < 250:
        return 120
    else:
        return 50
    return 120

x0, us, ds, p , R, R_d, delta_t = initialize()
Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t,F3=F3_func,F4=F4_func)

# Discrete Kalman filter parameters 
xs = Model_Stochastic.GetSteadyState(x0, us, ds)

P = 5*np.eye(A_est.shape[0])  # Initial estimate error covariance 

Tf = 500
N = int(Tf/delta_t)
t = np.arange(0, Tf, delta_t)
xt = x0.copy()-xs.copy()
xt_hat = x0.copy()-xs.copy()

# Find ud af hvordan vi estimerer d når vi har hankel matricen!!!!!!!!

X = np.zeros([N-1,4])
Z = np.zeros([N-1,2])
D = np.zeros([N-1,2])
U = np.zeros([N-1,2])
Z_est = np.zeros([N-1,2])
W = np.random.multivariate_normal(mean=np.zeros(A_est.shape[0]), cov=Q, size=N)
V = np.random.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R, size=N)
X_true = np.zeros([N, 4])  

linear = 1
static = 1
Hankel = 1
disturbance_change = 0

if disturbance_change:
    d = np.ones([len(t),2])*ds
    d[len(t)//2:] = np.array([50,50])
else:
    d = np.ones([len(t),2])*ds

A_use = A_est
B_use = B_est
C_use = C_est

for t_idx,t_val in enumerate(t[:-1]): 

    # Save true state before simulating     
    X_true[t_idx, :] = xt + xs

    if linear:
        zt = discrete_output_update(C_use, xt, V[t_idx][:-2]) 
        
    else:
        yt = Model_Stochastic.StateSensor(xt[:4])
        zt = yt[:2]

    # Designed with linear system 
    xt_hat, P = KalmanFilterUpdate(xt_hat, d[t_idx]-ds, us*0, zt, A_use, B_use, C_use, P, Q, R[:2,:2], stationary=static)

    U[t_idx, :] = us
    Z[t_idx, :] = zt + xs[:2]   
    X[t_idx, :] = xt_hat + xs
    D[t_idx, :] = xt_hat[-2:] + xs[4:]

    # Estimate output based on estimated state 
    if linear:
        yt_est = discrete_output_update(C_use, xt_hat, V[t_idx][:-2])
        Z_est[t_idx, :] = yt_est[:2] + xs[:2]
    else:
        yt_est = Model_Stochastic.StateSensor(xt_hat[:4])
        Z_est[t_idx, :] = yt_est[:2] + xs[:2]

    # Simulate next true state  
    if linear:
        xt = discrete_state_update(A_use, B_use, E_use, xt, us-us, d[t_idx]-ds, W[t_idx])

    else:
        f = Model_Stochastic.FullEquation
        sol = solve_ivp(f, (t_val, t_val+delta_t), xt+xs, method='RK45',args = (us,))
        xt = sol.y[:,-1]-xs

fig, ax = plt.subplots(4, 1, figsize=(12, 12))  
for i in range(4): 
    ax[i].plot(t[:-1], X_true[:-1, i],'--', label='True State', color="black")
    ax[i].plot(t[:-1], X[:, i], '*-', label='Kalman Estimate', color="red")
    ax[i].set_title(f'$x_{{{i+1},t}}$')
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xlabel('') 
ax[i].legend()
ax[i].grid(True)
ax[i].set_xlabel('Time')     
fig.suptitle("Open loop of the linear discrete time system system\n State estimation using discrete time Kalman filter", fontsize=16)
plt.tight_layout(pad=2)
plt.savefig(f"Figures/Problem6/Problem6_4/KF_5_D_constant_states_linear.png")
plt.close()
fig, ax = plt.subplots(2, 1, figsize=(12, 12))  
for i in range(2):
    ax[i].plot(t[:-1], Z[:, i]/100,'--', label='True state', color="black")
    ax[i].plot(t[:-1], Z_est[:, i]/100, '*-', label='Kalman estimate', color="red")
    ax[i].set_title(f'$z_{{{i+1},t}}$')
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xticklabels([]) 
    ax[i].set_xlabel('') 
ax[i].legend()
ax[i].grid(True)
ax[i].set_xlabel('Time')     
fig.suptitle("Open loop of the linear discrete time system  system\n State estimation using discrete time Kalman filter", fontsize=16)
plt.tight_layout(pad=2)
plt.savefig(f"Figures/Problem6/Problem6_4/KF_5_D_constant_output_linear.png")
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(12, 12)) 
ax[0].plot(t[:-1],d[:-1,0]*np.ones(len(t[:-1])), label='True $d_1$')
ax[0].plot(t[:-1],d[:-1,1]*np.ones(len(t[:-1])), label='True $d_2 $')
for i,VEC, label, title in zip(range(2),[D,U],["d","u"],["Disturbance Estimates","Control inputs"]):
    if "d" in label:
        ax[i].plot(t[:-1],VEC[:,0],'--',label=f"{label}$_1$ Kalman estimate")
        ax[i].plot(t[:-1],VEC[:,1],'--',label=f"{label}$_2$ Kalman estimate")
    else:
        ax[i].plot(t[:-1],VEC[:,0],'--',label=f"{label}$_1$")
        ax[i].plot(t[:-1],VEC[:,1],'--',label=f"{label}$_2$")
    ax[i].set_title(title)
    ax[i].legend()
    ax[i].grid(True)
ax[i].set_xlabel('Time')
fig.suptitle("Control input and disturbance estimated using discrete time Kalman filter with the linear discrete time system", fontsize=16)
plt.tight_layout(pad=2)

plt.savefig(f"Figures/Problem6/Problem6_4/KF_5_D_constant_input_linear.png")
plt.close()
