import numpy as np
from src.MPC import MPC
from src.FourTankSystem import FourTankSystem
from params.initialize import initialize
import matplotlib.pyplot as plt
import sys

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

    ################
    Umin = np.array([0, 0])
    Umax = np.array([3000, 3000])
    Dmin = np.array([-5, -10])
    Dmax = np.array([10, 5])

    u_op = np.array([250, 325])  # Operating point inputs
    d_op = np.array([100, 120])  # Operating point disturbances

    N_t = 2000
    N_mpc = 10

    good_goal = np.array([111.05, 100.0])  # Height of Tank 1 and 2

    good_goal = good_goal

    Wz = np.eye(2) * 2
    Wu = np.eye(2) * 1e-3
    Wdu = np.eye(2) * 0.5
    ###############

    problem = "Problem 5"
    
    print(f"Simulating: {problem}")
    
    x0, us, ds, p, R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t, F3=F3_func, F4=F4_func)
    x0 = np.concatenate((x0, ds))
    xs = Model_Stochastic.GetSteadyState(x0, us)
    data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
    Q = data_prob5["Q"]
    
    # Initialize variables for both problems
    xs_hankel = None
    hs_op_output = None
    xs_op = None  # Operating point steady state for Problem 4
    
    data = data_prob5
    Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs, ds, delta_t)
    G = np.block([
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.eye(2),
    ]).T
    # Set initial state and references based on problem type

    # For Problem 5: Use absolute values
    U_bar = np.ones((N_mpc, 2)) * u_op
    R_bar = np.ones((N_mpc, 2)) * good_goal
    R_bar_abs = R_bar
    x0_mpc = xs
    u0_mpc = us  # Use actual initial input from initialize()

    # Tuned weighting matrices:
    # Wz: High value (30) for strong output tracking - prioritizes reaching setpoint
    # Wu: Low value (1e-3) to allow sufficient control effort without excessive penalty
    # Wdu: Moderate value (0.8) for smooth but responsive input changes
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
        R=R_d,
        problem=problem, 
        Wz=Wz,      # Increased from 1: strong tracking priority
        Wu=Wu,    # Decreased from 1e-1: allow more control effort
        Wdu=Wdu,    # Decreased from 1: smoother but still responsive
        Umin=Umin,
        Umax=Umax,
        Dmin=Dmin,
        Dmax=Dmax,
    )
    
    xs_closedloop = xs

    t, x, u, d, h = Model_Stochastic.ClosedLoop(np.array([0, N_t]), xs_closedloop, mpc_controller)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, h[0, :], label='Height of Tank 1', color='dodgerblue')
    axes[0].plot(t, h[1, :], label='Height of Tank 2', color='tomato')
    axes[0].plot(t, h[2, :], label='Height of Tank 3', color='limegreen')
    axes[0].plot(t, h[3, :], label='Height of Tank 4', color='orange')
    # Plot setpoints (convert back to absolute if needed)
    R_bar_plot = R_bar_abs
    setpoint_1 = R_bar_plot[0, 0]
    setpoint_2 = R_bar_plot[0, 1]
    axes[0].plot(t, t*0 + setpoint_1, label='Setpoint for Tank 1', color='dodgerblue', ls='--')
    axes[0].plot(t, t*0 + setpoint_2, label='Setpoint for Tank 2', color='tomato', ls='--')
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
    problem_num = sys.argv[1] if len(sys.argv) > 1 else "5"
    fig.savefig(f'figures/Problem9/Problem_9_Heights_{problem_num}.png')
    plt.close()

    predicted_x_mpc = np.array(mpc_controller.predicted_x_mpc)
    predicted_y_mpc = np.array(mpc_controller.predicted_y_mpc)

    predicted_Px = np.array(mpc_controller.predicted_Px)
    predicted_Py = np.array(mpc_controller.predicted_Py)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, predicted_x_mpc[:, 0], label='Predicted Tank 1', color='dodgerblue', ls='-.' )
    axes[0].plot(t, predicted_x_mpc[:, 1], label='Predicted Tank 2', color='tomato', ls='-.' )

    axes[0].fill_between(t, predicted_x_mpc[:, 0] - 2*np.sqrt(predicted_Px[:, 0, 0]), predicted_x_mpc[:, 0] + 2*np.sqrt(predicted_Px[:, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[0].fill_between(t, predicted_x_mpc[:, 1] - 2*np.sqrt(predicted_Px[:, 1, 1]), predicted_x_mpc[:, 1] + 2*np.sqrt(predicted_Px[:, 1, 1]), color='tomato', alpha=0.2)
    
    axes[0].plot(t, x[0, :], label='Actual Tank 1', color='dodgerblue')
    axes[0].plot(t, x[1, :], label='Actual Tank 2', color='tomato')
    
    axes[0].legend()
    axes[0].set_ylabel('mass [kg]')
    axes[0].grid(True)
    axes[0].set_title('System states')
    
    axes[1].plot(t, predicted_y_mpc[:, 0], label='Predicted Output Tank 1', color='dodgerblue', ls='-.' )
    axes[1].plot(t, predicted_y_mpc[:, 1], label='Predicted Output Tank 2', color='tomato', ls='-.' )
    

    axes[1].fill_between(t, predicted_y_mpc[:, 0] - 2*np.sqrt(predicted_Py[:, 0, 0]), predicted_y_mpc[:, 0] + 2*np.sqrt(predicted_Py[:, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[1].fill_between(t, predicted_y_mpc[:, 1] - 2*np.sqrt(predicted_Py[:, 1, 1]), predicted_y_mpc[:, 1] + 2*np.sqrt(predicted_Py[:, 1, 1]), color='tomato', alpha=0.2)
    
    axes[1].plot(t, h[0, :], label='Actual Tank 1', color='dodgerblue')
    axes[1].plot(t, h[1, :], label='Actual Tank 2', color='tomato')
    
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Height of the tanks')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Height [m]')

    fig.savefig(f'figures/Problem9/Problem_9_Heights_5_kalman.png')
    plt.close()