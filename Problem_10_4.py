import numpy as np
from src.MPC import MPC
from src.FourTankSystem import FourTankSystem
from params.initialize import initialize
import matplotlib.pyplot as plt
import sys
import params.parameters_tank as para

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


if __name__ == "__main__":
    problem = "Problem 4"
    
    print(f"Simulating: {problem}")
    
    p = para.parameters()
    a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p

    x0, us, ds, p, R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t, F3=F3_func, F4=F4_func)
    x0 = np.concatenate((x0, ds))
    xs = Model_Stochastic.GetSteadyState(x0, us)
    hs = xs[:2] / (rho*np.array([A1, A2]))
    data_prob4 = np.load(r"Results\Problem4\Problem_4_estimates.npz")
    data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
    Q = data_prob5["Q"][:4, :4]

    data = data_prob4

    Ad = data["A"]
    Bd = data["B"]
    Cz = data["C"]
    G = np.zeros((4, 2))
    good_goal = np.array([111.05, 100.0])  # Height of Tank 1 and 2

    u_op = np.array([250, 325])  # Operating point inputs

    # Tuned MPC parameters for better performance
    N_mpc = 15  # Increased prediction horizon
    N_t = 2000

    # For Problem 5: Use absolute values
    U_bar = np.ones((N_mpc, 2)) * u_op
    R_bar = np.ones((N_mpc, 2)) * good_goal
    x0_mpc = xs
    u0_mpc = np.zeros(2)

    Umin = np.array([0, 0])
    Umax = np.array([3000, 3000])
    Dmin = np.array([-5, -10])
    Dmax = np.array([10, 5])

    margin_low = np.array([10.0, 10.0])    # Very tight lower margin
    margin_high = np.array([10.0, 10.0])   # Very tight upper margin

    # For Problem 5: Use absolute values
    Rmin = good_goal - margin_low  # Lower output bounds [cm]
    Rmax = good_goal + margin_high  # Upper output bounds [cm]
    print("Output constraints:")
    print(f"  Setpoint:                 {good_goal} [cm]")
    print(f"  Constraint range: Tank 1: [{Rmin[0]:.2f}, {Rmax[0]:.2f}], "
            f"Tank 2: [{Rmin[1]:.2f}, {Rmax[1]:.2f}]")

    slack_quad_weight = 1000.0  # Quadratic penalty - high to strongly discourage violations
    slack_lin_weight = 10.0      # Linear penalty - lower for soft constraint behavior
    
    Ws2 = np.eye(2) * slack_quad_weight  # Quadratic penalty on lower bound slack (s)
    Wt2 = np.eye(2) * slack_quad_weight  # Quadratic penalty on upper bound slack (t)
    Ws1 = np.eye(2) * slack_lin_weight   # Linear penalty on lower bound slack (s)
    Wt1 = np.eye(2) * slack_lin_weight   # Linear penalty on upper bound slack (t)
    
    Wz = np.eye(2) * 2.0      # Strong tracking priority
    Wu = np.eye(2) * 1e-3      # Low penalty - allow sufficient control effort
    Wdu = np.eye(2) * 0.5      # Moderate penalty - smooth but responsive
    
    R = np.eye(2)*100

    mpc_controller = MPC(
        N=N_mpc,
        u0=u0_mpc,
        x0=x0_mpc,
        w0=ds,
        U_bar=U_bar,
        R_bar=R_bar,
        A=Ad, 
        B=Bd, 
        C=Cz, 
        G=G,
        Q=Q,
        R=R,
        hadd=hs,
        problem=problem, 
        Wz=Wz,      # Increased from 1: strong tracking priority
        Wu=Wu,    # Decreased from 1e-1: allow more control effort
        Wdu=Wdu,    # Decreased from 1: smoother but still responsive
        Umin=Umin,
        Umax=Umax,
        Dmin=Dmin,
        Dmax=Dmax,
        Rmax=Rmax,
        Rmin=Rmin,
        Ws2=Ws2,
        Wt2=Wt2,
        Ws1=Ws1,
        Wt1=Wt1,
    )

    xs_closedloop = xs
    

    t, x, u, d, h = Model_Stochastic.ClosedLoop(np.array([0, N_t]), xs_closedloop, mpc_controller)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, h[0, :], label='Height of Tank 1', color='dodgerblue')
    axes[0].plot(t, h[1, :], label='Height of Tank 2', color='tomato')
    axes[0].plot(t, h[2, :], label='Height of Tank 3', color='limegreen')
    axes[0].plot(t, h[3, :], label='Height of Tank 4', color='orange')

    setpoint_1 = R_bar[0, 0]
    setpoint_2 = R_bar[0, 1]
    print(setpoint_1, setpoint_2)
    axes[0].plot(t, t*0 + setpoint_1, label='Setpoint for Tank 1', color='dodgerblue', ls='--')
    axes[0].plot(t, t*0 + setpoint_2, label='Setpoint for Tank 2', color='tomato', ls='--')
    axes[0].fill_between(t, Rmin[0], Rmax[0], color='dodgerblue', alpha=0.2)
    axes[0].fill_between(t, Rmin[1], Rmax[1], color='tomato', alpha=0.2)
    axes[0].legend()
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Height [m]')
    axes[0].grid(True)
    axes[1].plot(t, u[0, :], label='Flow of Tank 1', color='dodgerblue')
    axes[1].plot(t, u[1, :], label='Flow of Tank 2', color='tomato')
    axes[1].legend()
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Flow [m³/s]')
    axes[1].grid(True)
    fig.savefig(f'figures/Problem10/Problem_10_Heights_4.png')
    plt.close()

    X_hankel = np.zeros((4, len(t)))

    for (i, t_i) in enumerate(t[:-1]):
        u_i = u[:, i]
        X_hankel[:, i+1] = Ad@X_hankel[:, i] + Bd@u_i

    Y_hankel = Cz@X_hankel + h[1:2,0]



    predicted_x_mpc = np.array(mpc_controller.predicted_x_mpc)
    predicted_y_mpc = np.array(mpc_controller.predicted_y_mpc)

    predicted_Px = np.array(mpc_controller.predicted_Px)
    predicted_Py = np.array(mpc_controller.predicted_Py)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, predicted_x_mpc[:, 0], label='Predicted Hankel 1', color='dodgerblue', ls='-.' )
    axes[0].plot(t, predicted_x_mpc[:, 1], label='Predicted Hankel 2', color='tomato', ls='-.' )

    axes[0].plot(t, X_hankel[0, :], label='Hankel 1', color='dodgerblue', ls='--')
    axes[0].plot(t, X_hankel[1, :], label='Hankel 2', color='tomato', ls='--')

    axes[0].fill_between(t, predicted_x_mpc[:, 0] - 2*np.sqrt(predicted_Px[:, 0, 0]), predicted_x_mpc[:, 0] + 2*np.sqrt(predicted_Px[:, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[0].fill_between(t, predicted_x_mpc[:, 1] - 2*np.sqrt(predicted_Px[:, 1, 1]), predicted_x_mpc[:, 1] + 2*np.sqrt(predicted_Px[:, 1, 1]), color='tomato', alpha=0.2)

    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Hankel states')

    axes[1].plot(t, predicted_y_mpc[:, 0] + hs[0], label='Predicted Output Tank 1', color='dodgerblue', ls='-.' )
    axes[1].plot(t, predicted_y_mpc[:, 1] + hs[1], label='Predicted Output Tank 2', color='tomato', ls='-.' )

    axes[1].plot(t, h[0, :], label='Actual Tank 1', color='dodgerblue')
    axes[1].plot(t, h[1, :], label='Actual Tank 2', color='tomato')
    
    axes[1].plot(t, Y_hankel[0, :], label='Hankel Output Tank 1', color='dodgerblue', ls='--')
    axes[1].plot(t, Y_hankel[1, :], label='Hankel Output Tank 2', color='tomato', ls='--')

    axes[1].fill_between(t, predicted_y_mpc[:, 0] + hs[0] - 2*np.sqrt(predicted_Py[:, 0, 0]), predicted_y_mpc[:, 0] + hs[0] + 2*np.sqrt(predicted_Py[:, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[1].fill_between(t, predicted_y_mpc[:, 1] + hs[1] - 2*np.sqrt(predicted_Py[:, 1, 1]), predicted_y_mpc[:, 1] + hs[1] + 2*np.sqrt(predicted_Py[:, 1, 1]), color='tomato', alpha=0.2)
    
    axes[1].legend()
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Height [m]')
    axes[1].grid(True)
    axes[1].set_title('Height of the tanks')

    fig.savefig(f'figures/Problem10/Problem_10_Heights_4_kalman.png')
    plt.close()