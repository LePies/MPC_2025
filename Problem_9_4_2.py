import numpy as np
from src.MPC import MPC
from src.FourTankSystem import FourTankSystem
from params.initialize import initialize
import matplotlib.pyplot as plt
import sys
import params.parameters_tank as para
from src.PlotMPC_sim import PlotMPC_sim

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
    problem = "Problem 4"
    
    print(f"Simulating: {problem}")
    
    p = para.parameters()
    a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p

    x0, us, ds, p, R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t, F3=F3_func, F4=F4_func)
    x0 = np.concatenate((x0, ds))
    xs = Model_Stochastic.GetSteadyState(x0, us)
    hs = xs[:2] / (rho*np.array([A1, A2]))
    data_prob4 = np.load(r"Results\Problem4\Problem_4_estimates_d.npz")
    data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
    data = data_prob4

    A = data["A"]
    B = data["B"]
    C = data["C"]
    E = data["E"]
    Dd = data["Dd"]
    Q = data["Q"]
    
    Ad = A[:-2, :-2]
    Bd = B[:-2, :]
    Cz = C[:, :-2]

    good_goal = hs + np.array([10, 10])  # Height of Tank 1 and 2

  
    u_op = np.array([250, 325])  # Operating point inputs

    # Tuned MPC parameters for better performance
    N_mpc = 30  # Increased prediction horizon
    N_t = 20*60

    # For Problem 5: Use absolute values
    U_bar = np.ones((N_mpc, 2)) * u_op
    R_bar = np.ones((N_mpc, 2)) * good_goal
    x0_mpc = xs
    u0_mpc = np.zeros(2)

    Umin = np.array([0, 0])
    Umax = np.array([3000, 3000])
    Dmin = np.array([-5, -10])
    Dmax = np.array([10, 5])
    
    R = np.eye(2)*100
    
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
        E=E,
        R=R[:2,:2],
        hadd=hs,
        problem=problem, 
        Wz=np.eye(2) * 2,      # Increased from 1: strong tracking priority
        Wu=np.eye(2) * 1e-3,    # Decreased from 1e-1: allow more control effort
        Wdu=np.eye(2) * 0.5,    # Decreased from 1: smoother but still responsive
        Umin=Umin,
        Umax=Umax,
        Dmin=Dmin,
        Dmax=Dmax,
        tune_mpc=False,
    )

    xs_closedloop = xs
    

    t, x, u, h = Model_Stochastic.ClosedLoop_Linearized(np.array([0, N_t]), xs_closedloop, mpc_controller)

    PlotMPC_sim(t=t, h=h, u=u, x=x, R_bar=R_bar, xs=xs, hs=hs, mpc_controller=mpc_controller, file_name="Problem_9_4_Linearized", problem="9")