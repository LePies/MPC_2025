import numpy as np 
import matplotlib.pyplot as plt 
from src.QPSolver import qpsolver
from scipy.linalg import block_diag


def kron(W):
    I = np.empty_like(W)
    Wbar = np.kron(I,W)

    return Wbar

def MPC_uncontrained(zinit,uinit,Wz,Wu,Wdu,N,Markov_array):

    noutput,moutput = Wu.shape()
    wz_bar = kron(Wz)
    Wu_bar = kron(Wu)
    Wdu_bar = kron(Wdu)

    Lamb = np.eye(N)
    Lamb += -1*np.eye(N,k=-1)

    Gamma = 0    













