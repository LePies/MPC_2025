import numpy as np
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import matplotlib.pyplot as plt
from scipy.linalg import eig
import control as ctrl
import sympy as sp
from params.initialize import initialize


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

def G(s,C,A,B):
    return C @ np.linalg.inv(s*np.eye(A.shape[0]) - A) @ B

def filter_zeros_poles(a,b,tol = 0.0001):

    # For each element in a, check if there is any element in b within ±tol
    mask_a = np.array([not np.any(np.abs(b - x) < tol) for x in a])
    mask_b = np.array([not np.any(np.abs(a - y) < tol) for y in b])

    # Filter arrays
    a_filtered = a[mask_a]
    b_filtered = b[mask_b]

    return a_filtered, b_filtered


x0, u, d, p , R_s, R_d, delta_t = initialize()

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
zeros, poles = filter_zeros_poles(zeros,poles)

print("Kp=", G(3,Cz_SISO,Ac,Bc_SISO)/0.01263919)
print("Poles:", poles)
print("System stable (cont.-time):", stable_sys)
print("Transmission zeros:", zeros)
print("Controller stable (zeros LHP):", stable_ctrl)

