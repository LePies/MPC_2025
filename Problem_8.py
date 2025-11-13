import numpy as np 
import matplotlib.pyplot as plt 
from src.QPSolver import qpsolver
from scipy.linalg import block_diag
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem

def kron(W):
    I = np.empty_like(W)
    Wbar = np.kron(I,W)
    return Wbar

def MPC_uncontrained(xk,uk,wk,U_bar,R_bar,C,A,Wz,Wu,Wdu,N,Markov_array):

    noutput = uk.shape()

    Wz_bar = kron(Wz)
    Wu_bar = kron(Wu)
    Wdu_bar = kron(Wdu)
    I0 = np.eye(noutput)

    phix = np.array([C@A**i for i in range(1,N+1)])
    phiw = np.array([C@A**i for i in range(N)])
    bk = phix@xk + phiw@wk
    ck = R_bar - bk

    Lamb = np.eye(N*noutput)
    Lamb += -1*np.eye(N*noutput,k=-1)
    Gamma = np.ones([N*noutput,N*noutput])

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

    uk = qpsolver(H, g, 1e-6, 1e6, A, 1e-6, 1e6, uk)

    return uk + rho

def F3_func(t):
    if t < -50:
        return 100
    else:
        return 50
    return 100
    
def F4_func(t):
    if t < -50:
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


















