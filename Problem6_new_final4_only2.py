from cProfile import label
from src.HankelSystem import HankelSystem
import numpy as np
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import matplotlib.pyplot as plt
from src.KalmanFilterUpdate import KalmanFilterUpdate
from params.initialize import initialize

# ----------------------------
# Setup
# ----------------------------
p = para.parameters()
a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p

R_s = np.eye(4) * 0
R_d = np.eye(2) * 0

F3 = 250
F4 = 325
delta_t = 1

Model = FourTankSystem(R_s, R_d, p, delta_t, F3=F3, F4=F4)

data = np.load("Results/Problem4/Problem_4_estimates_d.npz")
# data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")  # not used below

A = data["A"]
B = data["B"]
C = data["C"]
D = data["D"]
Q = data["Q"]

Hankel = HankelSystem(A, B, C, D)

# Load measurement noise covariance etc. from initialize()
# (We use only the 2x2 for the two outputs)
x0, us, ds, p_init, R, R_d_init, delta_t_init = initialize()
R_y = R[:2, :2].copy()  # output measurement noise covariance (2x2)

# ----------------------------
# Operating point
# ----------------------------
u_op = np.array([250, 325])
d_op = np.array([100, 120])

xs_op = Model.GetSteadyState(np.array([0, 0, 0, 0]), u_op, d_op)
hs_op = xs_op / (rho * np.array([A1, A2, A3, A4]))

# ----------------------------
# Time + inputs
# ----------------------------
t_span = (0, 30 * 60)
t_array = np.arange(t_span[0], t_span[1] + delta_t, delta_t)
n_steps = len(t_array)

# Absolute input (constant around operating point)
u_new_array = np.zeros((2, n_steps + 1))
for k, tval in enumerate(t_array):
    u_new_array[:, k] = u_op  # (you can add sinusoids if you want)

u_dev_array = u_new_array - u_op[:, None]  # deviation input for Hankel/KF

# Hankel simulation (optional plot later)
x0_Hankel = np.zeros(A.shape[0])
t_Hankel, y_Hankel, x_Hankel = Hankel.Simulation(
    x0_Hankel, u_dev_array, t_span, delta_t, discrete_time=True
)

print("Eigenvalues of Hankel system:", Hankel.GetEigenvalues())

# OpenLoop expects u as (n_inputs, n_time_steps)
u_array = u_new_array[:, :n_steps]
d_array_original = np.tile(d_op.reshape(-1, 1), (1, n_steps))

# ----------------------------
# Plot containers
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

# ----------------------------
# Run two cases: step and no step
# ----------------------------
for col, step in enumerate([0.25, 0.0]):
    idx_mod = len(t_array) // 3

    # Disturbance profile for the nonlinear plant
    d_array = np.copy(d_array_original)
    d_array[0, idx_mod:] = d_array[0, idx_mod:] * (1 + step)

    # Simulate nonlinear plant
    t, x, u_abs, d_abs, h = Model.OpenLoop((t_span[0], t_span[1]), xs_op, u_array, d_array)

    # Output deviation from operating point:
    # h is height (absolute). Convert to deviation and apply output map (2 outputs).
    h_dev = h - hs_op[:, None]
    y_true_dev = Model.StateOutput(h_dev)  # shape (2, N)

    # Add measurement noise to outputs (this is the "sensor measurement")
    # One noise sample per time step
    v = np.random.multivariate_normal(mean=np.zeros(2), cov=R_y, size=y_true_dev.shape[1]).T
    y_meas_dev = y_true_dev + v  # shape (2, N)

    # KF states (Hankel state vector)
    xt_hat = np.zeros(A.shape[0])       # KF estimate (deviation)
    x_hankel = np.zeros(A.shape[0])     # "true" Hankel internal state for comparison
    P = 10.0 * np.eye(A.shape[0])

    X_kalman = np.zeros((len(t), A.shape[0]))
    X_hankel = np.zeros((len(t), A.shape[0]))
    Y_est = np.zeros((len(t), 2))

    P_est = np.zeros((len(t), A.shape[0], A.shape[0]))

    # Initial predicted output estimate
    Y_est[0, :] = (C @ xt_hat + D @ u_dev_array[:, 0])

    X_kalman[0, :] = xt_hat
    X_hankel[0, :] = x_hankel

    abs_step = d_op[0] * step

    for k in range(len(t) - 1):
        # Measurement (deviation)
        zt = y_meas_dev[:, k]
        u_dev = u_dev_array[:, k]

        # KF update (works on deviation coordinates)
        xt_hat, P = KalmanFilterUpdate(
            xt=xt_hat,
            ut=u_dev,
            yt=zt,
            A=A, B=B, C=C,
            P=P, Q=Q, R=R_y,
            stationary=False
        )

        X_kalman[k, :] = xt_hat
        P_est[k, :, :] = P

        # "True" Hankel state propagation for comparison
        if k == idx_mod:
            x_hankel[-2] = x_hankel[-2] + abs_step  # your manual disturbance step (as before)
        x_hankel = A @ x_hankel + B @ u_dev
        X_hankel[k + 1, :] = x_hankel

        # Output estimate from KF
        Y_est[k + 1, :] = (C @ xt_hat + D @ u_dev)

    # Output covariance over time from P
    Py_est = np.zeros((len(t), C.shape[0], C.shape[0]))
    for k in range(len(t)):
        Py_est[k, :, :] = C @ P_est[k, :, :] @ C.T + R[:2, :2]

    # ----------------------------
    # PLOTTING (match screenshot)
    # ----------------------------
    tt = t[:-1] / 60.0  # minutes on x-axis, but label says seconds in screenshot; screenshot uses "Time [s]" though.
                        # If you want exactly "Time [s]" with 0..30, keep minutes and label as seconds.
                        # We follow your screenshot: 0..30 with label "Time [s]" -> use minutes but label seconds.

    # State stds
    sx = np.sqrt(np.maximum(0.0, np.diagonal(P_est[:-1, :, :], axis1=1, axis2=2)))  # (N-1, nx)
    # Output stds
    sy = np.sqrt(np.maximum(0.0, np.diagonal(Py_est[:-1, :, :], axis1=1, axis2=2)))  # (N-1, 2)

    # # ----- TOP: states (first 4) -----
    # state_colors = ["dodgerblue", "tomato", "limegreen", "orange"]
    # for j in range(4):#+ xs_op[j]
    #     axes[0, col].plot(tt, X_hankel[:-1, j] , "-", color=state_colors[j], lw=2,label = rf"$True\ x_{{{j+1}}}$")
    #     axes[0, col].plot(tt, X_kalman[:-1, j] , "--", color=state_colors[j], lw=2, label=f"KF $x_{{{j+1}}}$")
    # axes[0, col].grid(True, alpha=0.25)
    # axes[0, col].legend(fontsize=8, ncols=2)

    # ----- MIDDLE: disturbances -----
    d1_true = d_array[0, :len(tt)]
    d2_true = d_array[1, :len(tt)]

    d1_hat = X_kalman[:-1, -2] + d_op[0]
    d2_hat = X_kalman[:-1, -1] + d_op[1]

    sd1 = 2.0 * np.sqrt(np.maximum(0.0, P_est[:-1, -2, -2]))
    sd2 = 2.0 * np.sqrt(np.maximum(0.0, P_est[:-1, -1, -1]))

    axes[0, col].plot(tt, d1_true, ".", color="black", lw=1.8, label="True $d_1$")
    axes[0, col].plot(tt, d1_hat, "--", color="purple", lw=2.0, label="KF $d_1$")
    axes[0, col].fill_between(tt, d1_hat - sd1, d1_hat + sd1, color="purple", alpha=0.18)

    axes[0, col].plot(tt, d2_true, ".", color="black", lw=1.8, label="True $d_2$")
    axes[0, col].plot(tt, d2_hat, "--", color="magenta", lw=2.0, label="KF $d_2$")
    axes[0, col].fill_between(tt, d2_hat - sd2, d2_hat + sd2, color="magenta", alpha=0.18)

    axes[0, col].set_ylabel("Flow [m³/s]",fontsize=16)
    axes[0, col].grid(True, alpha=0.25)
    axes[0, col].legend(fontsize=18)

    # ----- BOTTOM: outputs -----
    y1_true_abs = y_meas_dev[0, :-1] + hs_op[0]   # noisy "true" measurement, matches screenshot dots
    y2_true_abs = y_meas_dev[1, :-1] + hs_op[1]

    y1_hat_abs = Y_est[:-1, 0] + hs_op[0]
    y2_hat_abs = Y_est[:-1, 1] + hs_op[1]

    axes[1, col].plot(tt, y1_true_abs, ".", color="black", ms=2, alpha=0.9, label="True $z_1$")
    axes[1, col].plot(tt, y1_hat_abs, "--", color="dodgerblue", lw=2.0, label="KF $z_1$")
    axes[1, col].fill_between(tt, y1_hat_abs - 2 * sy[:, 0], y1_hat_abs + 2 * sy[:, 0], color="dodgerblue", alpha=0.15)

    axes[1, col].plot(tt, y2_true_abs, ".", color="black", ms=2, alpha=0.9, label="True $z_2$")
    axes[1, col].plot(tt, y2_hat_abs, "--", color="tomato", lw=2.0, label="KF $z_2$")
    axes[1, col].fill_between(tt, y2_hat_abs - 2 * sy[:, 1], y2_hat_abs + 2 * sy[:, 1], color="tomato", alpha=0.15)

    axes[1, col].set_ylabel("Height [m]", fontsize=16)
    axes[1, col].grid(True, alpha=0.25)
    axes[1, col].legend(fontsize=18)

    # Titles like screenshot
    # if col == 0:
    #     axes[0, col].set_title("States with step change in disturbance")
    # else:
    #     axes[0, col].set_title("States with no step change in disturbance")
    axes[0, col].set_title("Disturbance input",fontsize=18)
    axes[1, col].set_title("Output",fontsize=18)

    # Remove duplicate y-labels on right column
    if col == 1:
        axes[0, col].set_ylabel("")
        axes[1, col].set_ylabel("")

# x labels (screenshot says Time [s] and shows 0..30)
axes[1, 0].set_xlabel("Time [min]",fontsize=16)
axes[1, 1].set_xlabel("Time [min]",fontsize=16)

plt.tight_layout()
plt.savefig("figures/Problem6/Problem4_KF_dynamic.png", dpi=200)
plt.close()

