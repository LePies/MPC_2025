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

def augment_system(A,B,C,E,Q):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    A_aug = np.block([[A, E],
                      [np.zeros((2,n)), np.eye(2)]])
    B_aug = np.block([[B],
                      [np.zeros((2,m))]])
    C_aug = np.block([C, np.zeros((p,2))])
    Q_aug = np.block([[Q, np.zeros((n,2))],
                      [np.zeros((2,n)), 0.01*np.eye(2)]])
    return A_aug, B_aug, C_aug, Q_aug

np.random.seed(42)
#data_prob5 = np.load(r"Results\Problem4\Problem_4_estimates.npz")#np.load(r"Results\Problem5\Problem_5_estimates.npz")
data_probQ = np.load(r"Results\Problem5\Problem_5_estimates.npz")
Q = data_probQ["Q"]


Ts = 1

def F3_func(t):
    # if t < 250:
    #     return 100
    # else:
    #     return 50
    return 100
    
def F4_func(t):
    # if t < 250:
    #     return 120
    # else:
    #     return 50
    return 120

x0, us, ds, p , R, R_d, delta_t = initialize()
Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t,F3=F3_func,F4=F4_func)

# Discrete Kalman filter parameters 
x0 = np.concatenate((x0, ds))  
xs_true = Model_Stochastic.GetSteadyState(x0, us)
zs_true = Model_Stochastic.StateSensor(xs_true[:4])[:2]
print(xs_true)
Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs_true, ds, Ts)
Tf = 500
N = int(Tf/delta_t)
t = np.arange(0, Tf, delta_t)
linear = 1
static = 1
Hankel = 0
disturbance_change = 0

if disturbance_change:
    d = np.ones([len(t),2])*ds
    d[len(t)//2:] = np.array([50,50])
else:
    d = np.ones([len(t),2])*ds

A_use = Ad
B_use = Bd
C_use = Cz
E_use = Ed

A_use, B_use, C_use, Q_use = augment_system(A_use, B_use, C_use, E_use, Q)

P = 5*np.eye(A_use.shape[0])  # Initial estimate error covariance 

xt = xs_true.copy()-xs_true.copy()
x0 = np.concatenate((x0, ds))
xs = np.concatenate((xs_true, ds))
xt_hat = xs.copy()-xs.copy()

X = np.zeros([N-1,4]) 
Z = np.zeros([N-1,2])
D = np.zeros([N-1,2])
U = np.zeros([N-1,2])
Z_est = np.zeros([N-1,2])
W = np.random.multivariate_normal(mean=np.zeros(Ad.shape[0]), cov=Q, size=N)
V = np.random.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R, size=N)
X_true = np.zeros([N, 4])  

for t_idx,t_val in enumerate(t[:-1]): 

    # Save true state before simulating     
    X_true[t_idx, :] = xt[:4] + xs[:4] 

    if linear:
        zt = discrete_output_update(Cz, xt, V[t_idx][:-2]) 
        
    else:
        yt = Model_Stochastic.StateSensor(xt[:4])
        zt = yt[:2]

    # Designed with linear system 
    xt_hat, P = KalmanFilterUpdate(xt_hat, us*0, zt, A_use, B_use, C_use, P, Q_use, R[:2,:2], stationary=static)

    U[t_idx, :] = us
    Z[t_idx, :] = zt + zs_true 
    X[t_idx, :] = xt_hat[:4] + xs_true[:4]
    D[t_idx, :] = xt_hat[4:-2] + xs_true[4:]

    # Estimate output based on estimated state 
    if linear:
        yt_est = discrete_output_update(C_use, xt_hat, V[t_idx][:-2])
        Z_est[t_idx, :] = yt_est[:2] + zs_true
    else:
        yt_est = Model_Stochastic.StateSensor(xt_hat[:4])
        Z_est[t_idx, :] = yt_est[:2] + zs_true

    # Simulate next true state  
    if linear:
        xt = discrete_state_update(Ad, Bd, Ed, xt, us-us, d[t_idx]-ds, W[t_idx])

    else:
        f = Model_Stochastic.FullEquation
        sol = solve_ivp(f, (t_val, t_val+delta_t), xt+xs_true, method='RK45',args = (us,))
        xt = sol.y[:,-1]-xs_true

fig, ax = plt.subplots(4, 1, figsize=(12, 12))  
for i in range(4): 
    ax[i].plot(t[:-1], X_true[:-1, i],'--', label='True State', color="black")
    ax[i].plot(t[:-1], X[:, i], '*-', label='Kalman Estimate', color="red")
    ax[i].set_title(f'$x_{{{i+1},t}}$')
    ax[i].legend()
    ax[i].grid(True)
    #ax[i].set_xticklabels([]) 
    ax[i].set_xlabel('') 
ax[i].legend()
ax[i].grid(True)
ax[i].set_xlabel('Time')     
fig.suptitle("Open loop of the linear system in problem 5\n State estimation using discrete time linear Kalman filter", fontsize=16)
plt.tight_layout(pad=2)
plt.savefig(f"Figures/Problem6/KF_5_states_linear.png")
plt.close()
fig, ax = plt.subplots(2, 1, figsize=(12, 12)) 
ax[0].plot(t[:-1], Z[:, 0],'--', label='True height of Tank 1', color="black")
ax[0].plot(t[:-1], Z_est[:, 0], '*-', label='Kalman estimate of height of Tank 1', color="dodgerblue") 
ax[1].plot(t[:-1], Z[:, 1],'--', label='True height of Tank 2', color="black")
ax[1].plot(t[:-1], Z_est[:, 1], '*-', label='Kalman estimate of height of Tank 2', color="tomato")
for i in range(2):
    ax[i].legend(fontsize=18)
    ax[i].grid(True)
    #ax[i].set_xticklabels([]) 
    ax[i].set_ylabel('Height [m]',fontsize=16)
    ax[i].set_xlabel('') 
ax[i].set_xlabel('Time [min]',fontsize=16)     
fig.suptitle("Open loop of the linear system of problem 4\n State estimation using discrete time Kalman filter", fontsize=20)
plt.tight_layout(pad=2)
plt.savefig(f"Figures/Problem6/KF_4_output_linear.png")
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
fig.suptitle("Control input and disturbance estimated using discrete time Kalman filter with linear system", fontsize=16)
plt.tight_layout(pad=2)

plt.savefig(f"Figures/Problem6/KF_5_input_linear.png")
plt.close()
