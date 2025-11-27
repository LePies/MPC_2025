import numpy as np
from src.MPC import MPC
from src.FourTankSystem import FourTankSystem
from params.initialize import initialize
import matplotlib.pyplot as plt
import sys

def F3_func(t):
    # if t < 250:
    return 100
    # else:
    #     return 50
    # return 100


def F4_func(t):
    # if t < 250:
    return 120
    # else:
    #     return 50
    # return 120


if __name__ == "__main__":
    problem = "Problem 5"
    
    print(f"Simulating: {problem}")
    
    x0, us, ds, p, R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t,F3=F3_func, F4=F4_func)
    x0 = np.concatenate((x0, ds))
    xs = Model_Stochastic.GetSteadyState(x0, us)
    data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
    Q = np.block([
        [data_prob5["Q"], np.zeros((6, 2))],
        [np.zeros((2, 6)), 0.01*np.eye(2)]
    ])

    data = data_prob5
    Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs, ds, delta_t)
    
    good_goal = np.array([111.05, 100.0])  # Height of Tank 1 and 2

    hs = Model_Stochastic.StateSensor(xs[:4])[:2]
    good_goal = hs  + np.array([10, 10])
    # good_goal = hs

    u_op = np.array([250, 325])  # Operating point inputs

    # Tuned MPC parameters for better performance
    N_mpc = 30  # Increased prediction horizon
    N_t = 20*60

    # For Problem 5: Use absolute values
    U_bar = np.ones((N_mpc, 2)) * u_op
    R_bar = np.ones((N_mpc, 2)) * good_goal
    x0_mpc = xs
    u0_mpc = u_op
    
    mpc_controller = MPC(
        N=N_mpc,
        us=u_op,
        hs=hs,
        U_bar=U_bar,
        R_bar=R_bar,
        A=Ad,
        B=Bd,
        C=Cz,
        Q=Q,
        E=Ed,
        R=R[:2,:2],
        problem=problem, 
        Wz=np.eye(2) * 2,      # Increased from 1: strong tracking priority
        Wu=np.eye(2) * 0,    # Decreased from 1e-1: allow more control effort
        Wdu=np.eye(2) * 0.5,    # Decreased from 1: smoother but still responsive
    )
    
    xs_closedloop = xs


    t, x, u, d, h = Model_Stochastic.ClosedLoop(np.array([0, N_t]), xs_closedloop[:4], mpc_controller, d=xs_closedloop[4:])

    hs = h[0:2, 0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t/60, h[0, :], label='Height of Tank 1', color='dodgerblue')
    axes[0].plot(t/60, h[1, :], label='Height of Tank 2', color='tomato')
    axes[0].plot(t/60, h[2, :], label='Height of Tank 3', color='limegreen')
    axes[0].plot(t/60, h[3, :], label='Height of Tank 4', color='orange')

    setpoint_1 = R_bar[0, 0]
    setpoint_2 = R_bar[0, 1]
    axes[0].plot(t/60, t*0 + setpoint_1, label='Setpoint for Tank 1', color='dodgerblue', ls='--')
    axes[0].plot(t/60, t*0 + setpoint_2, label='Setpoint for Tank 2', color='tomato', ls='--')
    axes[0].legend()
    axes[0].set_xlabel('Time [min]')
    axes[0].set_ylabel('Height [m]')
    axes[0].grid(True)
    axes[1].plot(t/60, u[0, :], label='Flow of Tank 1', color='dodgerblue')
    axes[1].plot(t/60, u[1, :], label='Flow of Tank 2', color='tomato')
    axes[1].legend()
    axes[1].set_xlabel('Time [min]')
    axes[1].set_ylabel('Flow [m³/s]')
    axes[1].grid(True)
    fig.savefig(f'figures/Problem8/Problem_8_Heights_5.png')
    plt.close()

    predicted_x_mpc = np.array(mpc_controller.predicted_x_mpc)
    predicted_y_mpc = np.array(mpc_controller.predicted_y_mpc)

    predicted_Px = np.array(mpc_controller.predicted_Px)
    predicted_Py = np.array(mpc_controller.predicted_Py)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t/60, predicted_x_mpc[:, 0] + xs[0], label='Predicted Tank 1', color='dodgerblue', ls='-.' )
    axes[0].plot(t/60, predicted_x_mpc[:, 1] + xs[1], label='Predicted Tank 2', color='tomato', ls='-.' )

    axes[0].fill_between(t/60, predicted_x_mpc[:, 0] + xs[0] - 2*np.sqrt(predicted_Px[:, 0, 0]), predicted_x_mpc[:, 0] + xs[0] + 2*np.sqrt(predicted_Px[:, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[0].fill_between(t/60, predicted_x_mpc[:, 1] + xs[1] - 2*np.sqrt(predicted_Px[:, 1, 1]), predicted_x_mpc[:, 1] + xs[1] + 2*np.sqrt(predicted_Px[:, 1, 1]), color='tomato', alpha=0.2)
    
    axes[0].plot(t/60, x[0, :], label='Actual Tank 1', color='dodgerblue')
    axes[0].plot(t/60, x[1, :], label='Actual Tank 2', color='tomato')
    
    axes[0].legend()
    axes[0].set_ylabel('mass [kg]')
    axes[0].grid(True)
    axes[0].set_title('System states')
    
    axes[1].plot(t/60, predicted_y_mpc[:, 0] + hs[0], label='Predicted Output Tank 1', color='dodgerblue', ls='-.' )
    axes[1].plot(t/60, predicted_y_mpc[:, 1] + hs[1], label='Predicted Output Tank 2', color='tomato', ls='-.' )
    

    axes[1].fill_between(t/60, predicted_y_mpc[:, 0] + hs[0] - 2*np.sqrt(predicted_Py[:, 0, 0]), predicted_y_mpc[:, 0] + hs[0] + 2*np.sqrt(predicted_Py[:, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[1].fill_between(t/60, predicted_y_mpc[:, 1] + hs[1] - 2*np.sqrt(predicted_Py[:, 1, 1]), predicted_y_mpc[:, 1] + hs[1] + 2*np.sqrt(predicted_Py[:, 1, 1]), color='tomato', alpha=0.2)
    
    axes[1].plot(t/60, h[0, :], label='Actual Tank 1', color='dodgerblue')
    axes[1].plot(t/60, h[1, :], label='Actual Tank 2', color='tomato')
    
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Height of the tanks')
    axes[1].set_xlabel('Time [min]')
    axes[1].set_ylabel('Height [m]')
    
    fig.savefig(f'figures/Problem8/Problem_8_Heights_5_kalman.png')
    plt.close()

    print(good_goal)