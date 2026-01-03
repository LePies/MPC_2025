from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.KalmanFilterUpdate import KalmanFilterUpdate
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem

def discrete_state_update(A, B, E, xt, ut, dt, wt):
    return A @ xt + B @ ut + E @ dt + wt

def discrete_output_update(C, xt, vt):
    return C @ xt + vt

def augment_system(A, B, C, E, Q):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    A_aug = np.block([[A, E],
                      [np.zeros((2, n)), np.eye(2)]])
    B_aug = np.block([[B],
                      [np.zeros((2, m))]])
    C_aug = np.block([C, np.zeros((p, 2))])
    Q_aug = np.block([[Q, np.zeros((n, 2))],
                      [np.zeros((2, n)), np.eye(2)]])
    return A_aug, B_aug, C_aug, Q_aug

np.random.seed(42)

# Load Q (as in your code)
data_probQ = np.load(r"Results\Problem5\Problem_5_estimates.npz")
Q = data_probQ["Q"]

Ts = 1

def F3_func(t): return 100
def F4_func(t): return 120

x0, us, ds, p, R, R_d, delta_t = initialize()
Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t, F3=F3_func, F4=F4_func)

# True operating point (steady state)
x0_aug = np.concatenate((x0, ds))
xs_true = Model_Stochastic.GetSteadyState(x0_aug, us)
zs_true = Model_Stochastic.StateSensor(xs_true[:4])[:2]

# Linearized discrete-time model around steady state
Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs_true, ds, Ts)

Tf = 30*60
t = np.arange(0, Tf, delta_t)
N = len(t)

linear = False
static = True

# Augment model (adds random-walk disturbance states)
A_use, B_use, C_use, Q_use = augment_system(Ad, Bd, Cz, Ed, Q)

def run_case(disturbance_change: bool):

    if disturbance_change:
        def F3_func(t): return 100*1.25 if t >= Tf//3 else 100
        def F4_func(t): return 120
    else:
        def F3_func(t): return 100
        def F4_func(t): return 120

    x0, us, ds, p, R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t, F3=F3_func, F4=F4_func)


    # Disturbance trajectory 
    d = np.ones((N, 2)) * ds
    if disturbance_change:
        d[N // 3:] = np.array([100*1.25, 120])

    # Initial cov + initial deviations
    P = 10 * np.eye(A_use.shape[0])

    # True state deviation (about steady state)
    xt = np.zeros(Ad.shape[0])              # 4 states (deviation)
    # Filter state deviation (augmented: 4 + 2)
    xt_hat = np.zeros(A_use.shape[0])       # 6 states (deviation)

    # Noise sequences
    W = np.random.multivariate_normal(mean=np.zeros(Ad.shape[0]), cov=Q, size=N)
    V = np.random.multivariate_normal(mean=np.zeros(R.shape[0]),  cov=R, size=N)

    # Storage
    X_true = np.zeros((N, 4))
    X_hat  = np.zeros((N, 4))
    D_true = np.zeros((N, 2))
    D_hat  = np.zeros((N, 2))

    Z_true = np.zeros((N, 2))
    Z_hat  = np.zeros((N, 2))

    P_hist = np.zeros((N, A_use.shape[0], A_use.shape[0]))

    for k in range(N - 1):
        # True absolute physical state (for plotting)
        X_true[k, :] = xt[:4] + xs_true[:4]
        D_true[k, :] = d[k, :]

        # Measurement (2 outputs)
        if linear:
            zt_dev = discrete_output_update(Cz, xt, V[k][:-2])  # deviation output
            zt = zt_dev + zs_true                                # absolute
        else:
            yt = Model_Stochastic.StateSensor(xt[:4] + xs_true[:4])
            zt = yt[:2]

        # Kalman update (note: input is zero in deviation coordinates here)
        xt_hat, P = KalmanFilterUpdate(
            xt=xt_hat,
            ut=us * 0,
            yt=zt - zs_true,               # filter works on deviation
            A=A_use, B=B_use, C=C_use,
            P=P, Q=Q_use, R=R[:2, :2],
            stationary=static
        )
        P_hist[k, :, :] = P

        # Save estimates (absolute)
        X_hat[k, :] = xt_hat[:4] + xs_true[:4]
        D_hat[k, :] = xt_hat[4:6] + ds     # augmented disturbance states are deviations about ds

        # Estimated output + true output (absolute)
        if linear:
            zhat_dev = C_use @ xt_hat      # no extra noise for estimate
            Z_hat[k, :] = zhat_dev + zs_true
            Z_true[k, :] = zt
        else:
            yhat = Model_Stochastic.StateSensor(xt_hat[:4] + xs_true[:4])
            Z_hat[k, :] = yhat[:2]
            Z_true[k, :] = zt

        # Propagate true state deviation
        if linear:
            xt = discrete_state_update(Ad, Bd, Ed, xt, us - us, d[k] - ds, W[k])
        else:
            f = Model_Stochastic.FullEquation
            sol = solve_ivp(f, (t[k], t[k] + delta_t), xt + xs_true, method="RK45", args=(us,))
            xt = sol.y[:, -1] - xs_true

    # Final store (k = N-1)
    X_true[-1, :] = xt[:4] + xs_true[:4]
    D_true[-1, :] = d[-1, :]

    # Output covariance (2x2) over time from P
    Py = np.zeros((N, 2, 2))
    for k in range(N):
        Py[k] = C_use @ P_hist[k] @ C_use.T + R[:2, :2]

    # State stds for plotting (diagonal only)
    sx = np.sqrt(np.maximum(0.0, np.diagonal(P_hist[:, :4, :4], axis1=1, axis2=2)))  # (N,4)
    sd = np.sqrt(np.maximum(0.0, np.diagonal(P_hist[:, 4:6, 4:6], axis1=1, axis2=2)))# (N,2)
    sy = np.sqrt(np.maximum(0.0, np.diagonal(Py, axis1=1, axis2=2)))                  # (N,2)

    return {
        "t": t,
        "X_true": X_true, "X_hat": X_hat, "sx": sx,
        "D_true": D_true, "D_hat": D_hat, "sd": sd,
        "Z_true": Z_true, "Z_hat": Z_hat, "sy": sy,
    }

# Run both scenarios
res_step = run_case(disturbance_change=True)
res_none = run_case(disturbance_change=False)

# ---- Plot (3 rows × 2 columns) ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

def plot_states(ax, res, title):
    tt = res["t"]
    Xh, Xt, sx = res["X_hat"], res["X_true"], res["sx"]
    colors = ["dodgerblue", "tomato", "limegreen", "orange"]
    for i in range(4):
        ax.plot((tt/60)[:-1], Xt[:-1, i], ".", color="black", ms=2, alpha=0.9, label=f"True x{i+1}")
        ax.plot((tt/60)[:-1], Xh[:-1, i],   "--", color=colors[i], lw=2.0, label=f"KF x{i+1}")
        ax.fill_between((tt/60)[:-1], Xh[:-1, i] - 2*sx[:-1, i], Xh[:-1, i] + 2*sx[:-1, i], color=colors[i], alpha=0.15)
        
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18, ncols=2)

def plot_disturbances(ax, res):
    tt = res["t"]
    Dh, Dt, sd = res["D_hat"], res["D_true"], res["sd"]
    colors = ["purple", "magenta"]
    labels = [r"$\bar{d}_1$", r"$\bar{d}_2$"]
    for i in range(2):
        ax.plot((tt/60)[:-1], Dt[:-1, i], ".", color="black", ms=2, alpha=0.9,linewidth=1.6, label=f"True {labels[i]}")
        ax.plot((tt/60)[:-1], Dh[:-1, i],  "--", color=colors[i], lw=2.0, label=f"KF {labels[i]}")
        ax.fill_between((tt/60)[:-1], Dh[:-1, i] - 2*sd[:-1, i], Dh[:-1, i] + 2*sd[:-1, i], color=colors[i], alpha=0.2)

    ax.set_ylabel("Flow [m³/s]",fontsize=16)
    ax.set_title("Disturbance input", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)

def plot_outputs(ax, res):
    tt = res["t"]
    Zh, Zt, sy = res["Z_hat"], res["Z_true"], res["sy"]
    colors = ["dodgerblue", "tomato"]
    labels = ["$z_1$", "$z_2$"]
    for i in range(2):
        ax.plot((tt/60)[:-1], Zt[:-1, i], ".", color="black", ms=2, alpha=0.9, linewidth=1.6, label=f"True {labels[i]}")
        ax.plot((tt/60)[:-1], Zh[:-1, i],  "--", color=colors[i], lw=2.0, label=f"KF {labels[i]}")
        ax.fill_between((tt/60)[:-1], Zh[:-1, i] - 2*sy[:-1, i], Zh[:-1, i] + 2*sy[:-1, i], color=colors[i], alpha=0.2)
    ax.set_ylabel("Height [m]",fontsize=16)
    ax.set_title("Output", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)

# plot_states(axes[0, 0], res_step, "States with step change in disturbance")
# plot_states(axes[0, 1], res_none, "States with no step change in disturbance")

plot_disturbances(axes[0, 0], res_step)
plot_disturbances(axes[0, 1], res_none)
axes[1, 1].set_ylabel("")

plot_outputs(axes[1, 0], res_step)
plot_outputs(axes[1, 1], res_none)
axes[1, 1].set_ylabel("")

axes[1, 0].set_xlabel("Time [s]",fontsize=16)
axes[1, 1].set_xlabel("Time [s]",fontsize=16)

plt.tight_layout()
plt.savefig("Figures/Problem6/Problem5_KF_static_nonlinear.png", dpi=200)
plt.close()
