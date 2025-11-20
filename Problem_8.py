import numpy as np 
import matplotlib.pyplot as plt 
from src.QPSolver import qpsolver
from scipy.linalg import block_diag
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem

def get_gamma(N, problem="Problem 5"):
        """
        Get the Gamma matrix for the given problem and number of steps

        Parameters:
            N: number of steps
            problem: "Problem 5" or "Problem 4"
        Returns:
            Gamma: Gamma matrix
        """
        if problem == "Problem 5":
            data = np.load("Results/Problem5/Problem_5_estimates.npz")
        elif problem == "Problem 4":
            data = np.load("Results/Problem4/Problem_4_estimates.npz")
        else:
            raise ValueError("Invalid problem")

        markov_mat = data["markov_mat"]

        markov_mat_test = markov_mat[:, :, :N].reshape(2, -1)

        Gamma = np.zeros((2*N, 2*N))
        
        for i in range(N):
            markov_input = markov_mat_test[:, :2*(N-i)]
            markov_input = np.block([
                [np.zeros((2, 2*i)), markov_input]
            ])
            Gamma[:, 2*i:2*i+2] = markov_input.T

        return Gamma

def kron(W, N):
    I = np.eye(N)
    Wbar = np.kron(I,W)
    return Wbar

def MPC_uncontrained(xk, uk, wk, U_bar, R_bar, C, A, G, Wz, Wu, Wdu, N):

    U_bar = U_bar.flatten()
    R_bar = R_bar.flatten()

    noutput = uk.shape[0]

    Wz_bar = kron(Wz, N)
    Wu_bar = kron(Wu, N)
    Wdu_bar = kron(Wdu, N)
    I0 = np.block([np.eye(noutput) if i == 0 else np.zeros((noutput, noutput)) for i in range(N)]).T

    phix = np.block([(C@A**i).T for i in range(1,N+1)])
    phiw = np.block([(C@(A**i)@G).T for i in range(N)])

    print(phix.shape)
    print(phiw.shape)
    print(xk.shape)
    print(wk.shape)
    
    bk = phix.T@xk + phiw.T@wk

    print(bk.shape)
    print(R_bar.shape)

    ck = R_bar - bk

    Lamb = np.eye(N*noutput)
    Lamb += -1*np.eye(N*noutput,k=-1)
    Gamma = get_gamma(N, problem="Problem 5")

    Hu = Wu_bar.T@Wu_bar

    Hdu = Lamb.T@Wdu_bar.T@Wdu_bar@Lamb
    Hz = Gamma.T@Wz_bar.T@Wz_bar@Gamma
    H = Hu + Hdu + Hz

    gu = -Wu_bar.T@Wu_bar@U_bar

    gdu = -Lamb.T@Wdu_bar.T@Wdu_bar@I0@uk
    gz = -(Wz_bar@Gamma).T@Wz_bar@ck 
    g = gu+gdu+gz

    rhou = 1/2*U_bar.T@Wu_bar.T@Wu_bar@U_bar
    rhodu = 1/2*(Wdu_bar@I0@uk).T@(Wdu_bar@I0@uk)
    rhoz = 1/2*ck.T@Wz_bar@Wz_bar@ck
    rho = rhodu+rhou+rhoz

    uinit = np.zeros(H.shape[0])

    ufin, info = qpsolver(H, g, -np.inf, np.inf, H*0, -np.inf, np.inf, uinit)

    return ufin.reshape(-1, 2).T, info["f"] + rho

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

    # Discrete Kalman filter parameters 
    x0 = np.concatenate((x0, ds))  
    xs = Model_Stochastic.GetSteadyState(x0, us)
    Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs, ds,delta_t)

    N = 10
    U_bar = np.ones((N, 2))*250
    R_bar = np.ones((N, 2))*100
    Wz = np.eye(2)
    Wu = np.eye(2)
    Wdu = np.eye(2)

    G = np.block([
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.eye(2),
    ]).T

    ufin, cost = MPC_uncontrained(
        xs, us, ds, U_bar, R_bar, Cz, Ad, G, Wz, Wu, Wdu, N)
    
    plt.plot(ufin[0,:])
    plt.plot(ufin[1,:])
    plt.show()
    



















