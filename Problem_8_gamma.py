import numpy as np
from src.MPC import MPC
from src.FourTankSystem import FourTankSystem
from params.initialize import initialize
import matplotlib.pyplot as plt

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
    x0, us, ds, p , R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t,F3=F3_func,F4=F4_func)

    data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")

    # Discrete Kalman filter parameters 
    x0 = np.concatenate((x0, ds))  
    xs = Model_Stochastic.GetSteadyState(x0, us)
    Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs, ds,delta_t)

    G = np.block([
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.eye(2),
    ]).T

    N_mpc = 10
    N_t = 100
    U_bar = np.zeros((N_mpc, 2))
    R_bar = np.ones((N_mpc, 2))*100

    MPC = MPC(
        N=N_mpc, 
        u0 = np.zeros(2),
        x0 = xs,
        w0 = ds,
        U_bar = U_bar,
        R_bar = R_bar,
        A = Ad, 
        B = Bd, 
        C = Cz, 
        G = G,
        Q = data_prob5["Q"],
        R = R_d,
        problem="Problem 5", 
        Wz=np.eye(2)*20, 
        Wu=np.eye(2)*1e-3, 
        Wdu=np.eye(2),
    )

    t, x, u, d, h = Model_Stochastic.ClosedLoop(np.array([0,N_t]), xs, MPC)

    plt.plot(t, h[0,:])
    plt.plot(t, h[1,:])
    plt.plot(t, h[2,:])
    plt.plot(t, h[3,:])

    plt.plot(t, t*0 + R_bar[0,0])
    plt.plot(t, t*0 + R_bar[0,1])
    plt.show()

    plt.plot(t, R_bar[0,0] - h[0,:])
    plt.plot(t, R_bar[1,1] - h[1,:])
    plt.plot(t, t*0, ls="--")
    plt.show()

    plt.plot(t, u[0,:])
    plt.plot(t, u[1,:])
    plt.show()

    plt.plot(t, d[0,:])
    plt.plot(t, d[1,:])
    plt.show()