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
    if len(sys.argv) > 1:
        problem = "Problem " + sys.argv[1]
    else:
        problem = "Problem 5"
    
    print(f"Simulating: {problem}")
    
    x0, us, ds, p, R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t, F3=F3_func, F4=F4_func)
    x0 = np.concatenate((x0, ds))
    xs = Model_Stochastic.GetSteadyState(x0, us)
    data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
    Q = data_prob5["Q"]
    
    if problem == "Problem 5":
        data = data_prob5
        Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs, ds, delta_t)
        G = np.block([
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.eye(2),
        ]).T
    elif problem == "Problem 4":  
        data = np.load(r"Results\Problem4\Problem_4_estimates.npz")
        Ad = data["A"]
        Bd = data["B"]
        Cz = data["C"]
        G = np.block([
            np.zeros((2, 2)),
            np.eye(2),
        ]).T
        Q = Q[:4,:4]
    else:
        raise ValueError("Invalid problem")
    

    good_goal = np.array([111.05, 100.0])  # Height of Tank 1 and 2

    # Tuned MPC parameters for better performance
    N_mpc = 15  # Increased prediction horizon for better long-term prediction
    N_t = 500

    # Input reference: set to expected steady-state input for the reference
    # Based on initial values (F1=250, F2=325), adjusted for the reference goal
    U_bar = np.ones((N_mpc, 2)) * np.array([280, 300])  # Tuned input reference
    R_bar = np.ones((N_mpc, 2)) * good_goal


    Umin = np.array([0, 0])
    Umax = np.array([3000, 3000])
    Dmin = np.array([-5, -10])
    Dmax = np.array([10, 5])
    # Tuned weighting matrices:
    # Wz: High value (30) for strong output tracking - prioritizes reaching setpoint
    # Wu: Low value (1e-3) to allow sufficient control effort without excessive penalty
    # Wdu: Moderate value (0.8) for smooth but responsive input changes
    mpc_controller = MPC(
        N=N_mpc, 
        u0=np.ones(2),
        x0=xs,
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
        Wz=np.eye(2) * 2,      # Increased from 1: strong tracking priority
        Wu=np.eye(2) * 1e-3,    # Decreased from 1e-1: allow more control effort
        Wdu=np.eye(2) * 0.5,    # Decreased from 1: smoother but still responsive
        Umin=Umin,
        Umax=Umax,
        Dmin=Dmin,
        Dmax=Dmax,
    )

    t, x, u, d, h = Model_Stochastic.ClosedLoop(np.array([0, N_t]), xs, mpc_controller)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, h[0, :], label='Height of Tank 1', color='dodgerblue')
    axes[0].plot(t, h[1, :], label='Height of Tank 2', color='tomato')
    axes[0].plot(t, h[2, :], label='Height of Tank 3', color='limegreen')
    axes[0].plot(t, h[3, :], label='Height of Tank 4', color='orange')
    axes[0].plot(t, t*0 + R_bar[0, 0], label='Setpoint for Tank 1', color='dodgerblue', ls='--')
    axes[0].plot(t, t*0 + R_bar[0, 1], label='Setpoint for Tank 2', color='tomato', ls='--')
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
    fig.savefig('figures/Problem9/Problem_9_Heights.png')
    plt.close()
