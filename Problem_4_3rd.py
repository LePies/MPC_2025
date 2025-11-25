from cProfile import label
from src.HankelSystem import HankelSystem
import numpy as np
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import matplotlib.pyplot as plt
from src.KalmanFilterUpdate import KalmanFilterUpdate
import sys
import scipy as sp

p = para.parameters()
a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p
R_s = np.eye(4)*0
R_d = np.eye(2)*0

F3 = 250
F4 = 325

delta_t = 1

Model = FourTankSystem(R_s, R_d, p, delta_t, F3 = F3, F4 = F4)

data = np.load("Results/Problem4/Problem_4_estimates_d.npz")
data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")

# Load discrete-time matrices (system was identified from discrete-time Markov parameters)
A = data["A"]  # Discrete-time A matrix
B = data["B"]  # Discrete-time B matrix
C = data["C"]  # C matrix (same for discrete and continuous)
D = data["D"]  # D matrix (same for discrete and continuous)
E = data["E"]
Dd = data["Dd"]
Q = data["Q"]


Hankel = HankelSystem(A, B, C, D)



# Operating point for linearization
u_op = np.array([250, 325])  # Operating point inputs
d_op = np.array([100, 120])  # Operating point disturbances

# Steady state at operating point
xs_op = Model.GetSteadyState(np.array([0, 0, 0, 0]), u_op, d_op)
hs_op = xs_op / (rho * np.array([A1, A2, A3, A4]))  # Operating point heights

# Create oscillating input
# Oscillating around operating point: u(t) = u_op + amplitude * sin(2*pi*frequency*t) 
amplitude = np.array([200, 150])  # Oscillation amplitude for each input
frequency = 0.01  # Frequency in Hz (oscillation period = 1/frequency seconds)

# For discrete-time simulation, we need to create time-varying input
t_span = (0, 30*60)
t_array = np.arange(t_span[0], t_span[1] + delta_t, delta_t)
n_steps = len(t_array)

# Create oscillating input array (absolute values)
u_new_array = np.zeros((2, n_steps + 1))
for k, t in enumerate(t_array):
    u_new_array[:, k] = u_op #+ amplitude * np.sin(2 * np.pi * frequency * t)

# Input deviations from operating point (for Hankel system)
u_dev_array = u_new_array - u_op[:, None]

# Initial condition: deviation from operating point (0 if starting at steady state)
x0_Hankel = np.zeros(A.shape[0])  # Start at operating point steady state

# Simulate Hankel system with oscillating deviations using discrete-time solver 
# The system is discrete-time (identified from discrete-time Markov parameters) 
# Pass time-varying input array
t_Hankel, y_Hankel, x_Hankel = Hankel.Simulation(x0_Hankel, u_dev_array, t_span, delta_t, discrete_time=True)


print("Eigenvalues of Hankel system:", Hankel.GetEigenvalues())
# OpenLoop expects u to be 2D array (n_inputs, n_time_steps)
# Use the oscillating input array (already created above)
u_array = u_new_array[:, :n_steps]  # Match the number of steps OpenLoop expects 
d_array_original = np.tile(d_op.reshape(-1, 1), (1, n_steps))


fig, axes = plt.subplots(3, 2 , figsize=(12, 12), sharex=True)
for i, step in enumerate([0.25, 0.0]):
    idx_mod = len(t_array)//3
    d_array = np.copy(d_array_original)
    d_array[0, idx_mod:] = d_array[0, idx_mod:]*(1 + step)

    t, x, u, d, h = Model.OpenLoop((t_span[0], t_span[1]), xs_op, u_array, d_array)

    # Calculate output deviations from operating point for FourTankSystem 
    # h are heights (4 values), we need deviations and then apply StateOutput 
    h_dev = h - hs_op[:, None]  # Deviations from operating point
    y_model = Model.StateOutput(h_dev)  # Get 2 outputs (tanks 1 and 2) 


    xt_hat = np.zeros(A.shape[0]) 
    x_hankel = np.zeros(A.shape[0])
    P = np.eye(A.shape[0])
    ds = d_op
    static = True
    us = u_op
    xs = xs_op
    X_kalman = np.zeros((len(t), A.shape[0]))
    X_hankel = np.zeros((len(t), A.shape[0]))

    yt = np.zeros(2)

    Y_est = np.zeros((len(t), 2))

    Y_est[0, :] = C@xt_hat + D@u_dev_array[:, 0]
    X_hankel[0, :] = x_hankel
    X_kalman[0, :] = xt_hat

    R = np.eye(2)*1

    P_est = np.zeros((len(t), A.shape[0], A.shape[0]))

    abs_step = d_op[0]*step

    for (i_t, t_val) in enumerate(t[:-1]):
        zt = y_model[:, i_t]  # Use the 2-output measurement (tanks 1 and 2) from StateOutput  
        u = u_dev_array[:, i_t]
        xt_hat, P = KalmanFilterUpdate(xt = xt_hat, ut = u, yt = zt, A = A, B = B, C = C, P = P, Q = Q, R = R, stationary=static)
        X_kalman[i_t, :] = xt_hat

        P_est[i_t, :, :] = P
        if i_t == idx_mod:
            x_hankel[-2] = x_hankel[-2] + abs_step
        x_hankel = A@x_hankel + B@u
        X_hankel[i_t+1, :] = x_hankel

        yt = C@x_hankel + D@u
        Y_est[i_t+1, :] = C@xt_hat + D@u

    # Compute output covariance for each time step 
    Py_est = np.zeros((len(t), C.shape[0], C.shape[0]))
    for i_t in range(len(t)):
        Py_est[i_t, :, :] = C @ P_est[i_t, :, :] @ C.T

    colors = ['dodgerblue', 'tomato', 'limegreen', 'orange', 'purple', 'brown', 'gray', 'pink']
    for xj in range(X_kalman.shape[1] - 2):
        axes[0, i].plot(t[:-1]/60, X_kalman[:-1, xj], 'x', label=f"Kalman Estimate (Hankel {xj+1})", color=colors[xj], markersize=2)
        axes[0, i].fill_between(t[:-1]/60, X_kalman[:-1, xj] - 2*np.sqrt(P_est[:-1, xj, xj]), X_kalman[:-1, xj] + 2*np.sqrt(P_est[:-1, xj, xj]), color=colors[xj], alpha=0.2)
        axes[0, i].plot(t[:-1]/60, X_hankel[:-1, xj], label=f"Hankel {xj+1}", ls='--', color=colors[xj])

    axes[0, i].legend(fontsize=5)
    axes[0, i].grid(True, alpha=0.2)
    axes[0, i].set_title('Kalman and Hankel', fontsize=10)

    axes[2, i].plot(t[:-1]/60, Y_est[:-1, 0] + hs_op[0], 'x', label="Output 1 (Kalman)", color='dodgerblue', markersize=2)
    axes[2, i].plot(t[:-1]/60, Y_est[:-1, 1] + hs_op[1], 'x', label="Output 2 (Kalman)", color='tomato', markersize=2)

    axes[2, i].plot(t[:-1]/60, y_model[0, :-1] + hs_op[0], label="Output 1 (Model)", color='dodgerblue')
    axes[2, i].plot(t[:-1]/60, y_model[1, :-1] + hs_op[1], label="Output 2 (Model)", color='tomato')

    axes[2, i].fill_between(t[:-1]/60, hs_op[0] + Y_est[:-1, 0] - 2*np.sqrt(Py_est[:-1, 0, 0]), hs_op[0] + Y_est[:-1, 0] + 2*np.sqrt(Py_est[:-1, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[2, i].fill_between(t[:-1]/60, hs_op[1] + Y_est[:-1, 1] - 2*np.sqrt(Py_est[:-1, 1, 1]), hs_op[1] + Y_est[:-1, 1] + 2*np.sqrt(Py_est[:-1, 1, 1]), color='tomato', alpha=0.2)

    axes[2, i].legend()
    axes[2, i].grid(True, alpha=0.5)
    axes[2, i].set_xlabel('Time [min]')
    axes[2, i].set_ylabel('Height [m]')
    axes[2, i].set_title('Output')

    axes[1, i].plot(t/60, X_kalman[:, -2] + d_op[0], 'x', label="Disturbance estimate 1", color='gray', markersize=2)
    axes[1, i].plot(t/60, X_kalman[:, -1] + d_op[1], 'x', label="Disturbance estimate 2", color='magenta', markersize=2)

    axes[1, i].fill_between(t/60, X_kalman[:, -2] + d_op[0] - 2*np.sqrt(P_est[:, -2, -2]), X_kalman[:, -2] + d_op[0] + 2*np.sqrt(P_est[:, -2, -2]), color='gray', alpha=0.2)
    axes[1, i].fill_between(t/60, X_kalman[:, -1] + d_op[1] - 2*np.sqrt(P_est[:, -1, -1]), X_kalman[:, -1] + d_op[1] + 2*np.sqrt(P_est[:, -1, -1]), color='magenta', alpha=0.2)

    axes[1, i].plot(t/60, d_array[0, :n_steps-1], label="Disturbance 1", color='gray')
    axes[1, i].plot(t/60, d_array[1, :n_steps-1], label="Disturbance 2", color='magenta')

    axes[1, i].plot(t/60, X_hankel[:, -2] + d_op[0], label="Hankel 5", color='gray', ls='--')
    axes[1, i].plot(t/60, X_hankel[:, -1] + d_op[1], label="Hankel 6", color='magenta', ls='--')
    axes[1, i].grid(True, alpha=0.5)
    axes[1, i].set_ylabel('Flow [m³/s]')
    axes[1, i].set_title('Disturbance')
    axes[1, i].legend()

axes[0, 0].set_title(r'd'+' Step of 0.1\nHankel states')
axes[0, 1].set_title(r'd'+' Step of 0.0\nHankel states')
axes[1, 1].set_ylabel('')
axes[2, 1].set_ylabel('')

plt.savefig('figures/Problem4/Problem_4_3rd_kalman.png')
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

axes[0].plot(t_Hankel/60, y_Hankel[0, :] + hs_op[0], label="Hankel Tank 1", color='dodgerblue', ls='--')
axes[0].plot(t_Hankel/60, y_Hankel[1, :] + hs_op[1], label="Hankel Tank 2", color='tomato', ls='--')

axes[0].plot(t/60, y_model[0, :] + hs_op[0], label="Model Tank 1", color='dodgerblue')
axes[0].plot(t/60, y_model[1, :] + hs_op[1], label="Model Tank 2", color='tomato')

axes[1].plot(t/60, u_array[0, :-1], label="U 1", color='dodgerblue')
axes[1].plot(t/60, u_array[1, :-1], label="U 2", color='tomato')

for i in range(2):
    axes[i].legend()
    axes[i].grid(True, alpha=0.5)
    axes[i].set_xlabel('Time [min]')
axes[0].set_ylabel('Height [m]')
axes[1].set_ylabel('Flow [m³/s]')

axes[0].set_title('Tank 1 and 2 Heights')
axes[1].set_title('Control Inputs')

plt.savefig('figures/Problem4/Problem_4_3rd.png')
plt.close()