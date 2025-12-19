import numpy as np
from src.MPC import MPC
from src.FourTankSystem import FourTankSystem
from params.initialize import initialize
import matplotlib.pyplot as plt
import sys
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

    hs = Model_Stochastic.StateSensor(xs[:4])
    good_goal = hs[:2]  + np.array([10, 10])
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
        hs=hs[:2],
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


    t, x, u, h = Model_Stochastic.ClosedLoop_Linearized(np.array([0, N_t]), xs_closedloop, mpc_controller)

    PlotMPC_sim(t=t, h=h, u=u, x=x, R_bar=R_bar, xs=xs, hs=hs, mpc_controller=mpc_controller, file_name="Problem_8_5_Linearized", problem="8")