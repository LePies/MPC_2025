import numpy as np
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import matplotlib.pyplot as plt
from scipy.linalg import eig
import control as ctrl
import sympy as sp


def MN_matrix_SISO(A,B,C):

    n = A.shape[0]          # number of states
    m = B.shape[1]          # number of inputs
    p = C.shape[0]          # number of measured outputs for zeros

    M = np.block([
        [A,            B],
        [C,      np.array([[0]])]
    ])

    N = np.block([
        [np.eye(n),     np.zeros((n, m))],
        [np.zeros((p, n)), np.zeros((p, m))]
    ])

    return M,N

def compute_zeros_poles(M, N, A):

    # Generalized eigenvalues of (M, N) 
    zeros = eig(M, N, left=False, right=False) # Returns only eigenvalues

    # Keep only finite zeros 
    finite_mask = np.isfinite(zeros)
    zeros = zeros[finite_mask]
    stable_ctrl = np.all(np.real(zeros) < 0) if zeros.size else False

    poles = eig(A, left=False, right=False) # Returns only eigenvalues 
    stable_sys = np.all(np.real(poles) < 0)

    return zeros, poles, stable_ctrl, stable_sys

t0 = 0
tf = 20*60 
m10 = 0
m20 = 0 
m30 = 0
m40 = 0
F1 = 250
F2 = 325
F3 = 100
F4 = 120
x0 = np.array([m10,m20,m30,m40])
u = np.array([F1,F2])
d = np.array([F3,F4])
p = para.parameters()
R_s = np.eye(4)*0.1
R_d = np.eye(2)*0.1
delta_t = 1


# Linearize continous time
Model_Deterministic = FourTankSystem(R_s*0, R_d*0, p, delta_t)
x0 = np.concatenate((x0, np.zeros(2)))  
d = np.array([])
xs = Model_Deterministic.GetSteadyState(x0, u, d)
xs = np.concatenate((xs, np.zeros(2))) 
Ac,Bc,Ec,C,Cz = Model_Deterministic.LinearizeContinousTime(xs,d)
Dz = np.zeros((Cz.shape[1], Bc.shape[1]))




Cz_SISO = Cz[:1, :]    
Bc_SISO = Bc[:, :1] 
Dz_SISO = Dz[:, :1]
print("SISO")
print(Cz_SISO)
print(Bc_SISO)
print(Dz_SISO)

M, N = MN_matrix_SISO(Ac,Bc_SISO,Cz_SISO)

zeros, poles, stable_ctrl, stable_sys = compute_zeros_poles(M, N, Ac)
Ts = 1
Ad, Bd, C, Cz = Model_Deterministic.LinearizeDiscreteTime(xs,d,Ts)

print("Poles:", poles)
print("System stable (cont.-time):", stable_sys)
print("Transmission zeros:", zeros)
print("Controller stable (zeros LHP):", stable_ctrl)

