import numpy as np
from src.FourTankSystem import FourTankSystem
from scipy.linalg import eig
from params.initialize import initialize
import pandas as pd

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

    zeros, poles = filter_zeros_poles(zeros,poles)

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

def SISO_system(B,C,input,output):

    if input == 1 and output == 1:
        B_siso = B[:, :1]
        C_siso = C[:1, :]
        return  B_siso, C_siso
    elif input == 2 and output == 1:
        B_siso = B[:, :1]
        C_siso = C[1:, :]
        return B_siso, C_siso
    elif input == 1 and output == 2:
        B_siso = B[:, 1:]
        C_siso = C[:1, :]
        return B_siso, C_siso
    elif input == 2 and output == 2:
        B_siso = B[:, 1:]
        C_siso = C[1:, :]
        return B_siso, C_siso

x0, u, d, p , R_s, R_d, delta_t = initialize()

# Linearize continous time
Model_Stochastic = FourTankSystem(R_s*0, R_d*0, p, delta_t)
x0 = np.concatenate((x0, np.zeros(2)))  
xs = Model_Stochastic.GetSteadyState(x0, u)
print(len(xs))

Ac,Bc,Ec,C,Cz = Model_Stochastic.LinearizeContinousTime(xs,d)

pairs = [(1, 1), (2, 1), (2, 2), (1, 2)]
Gains = pd.DataFrame(index=[1,2])
Tau = pd.DataFrame(index=[1,2])
Zeros = pd.DataFrame(index=[1,2])
Poles = pd.DataFrame(index=[1,2])

for pair in pairs:
    input = pair[0]
    output = pair[1]

    Bc_SISO,Cz_SISO = SISO_system(Bc,Cz,input,output)
    M, N = MN_matrix_SISO(Ac,Bc_SISO,Cz_SISO)
    zeros, poles, stable_ctrl, stable_sys = compute_zeros_poles(M, N, Ac)
    
    if len(poles) > 1:
        factor = poles[0]*poles[1]
    else:
        factor = poles[0]
    
    # Tau
    TauN = len(np.array([1/np.real(factor)]))
    zerospad = np.zeros(2-TauN)
    Tau[f"{pair}"] = np.concatenate([np.array([1/np.real(factor)]),zerospad])
    
    # Gain 
    GainN = len((G(0,Cz_SISO,Ac,Bc_SISO)/np.abs(factor))[0])
    zerospad = np.zeros(2-GainN)
    Gains[f"{pair}"] = np.concatenate([(G(0,Cz_SISO,Ac,Bc_SISO)/np.abs(factor)/np.abs(factor))[0],zerospad])
    
    # Zeros
    ZerosN = len(np.real(zeros))
    zerospad = np.zeros(2-ZerosN)
    Zeros[f"{pair}"] = np.concatenate([np.real(zeros),zerospad])
    
    # Poles
    PolesN = len(np.real(poles))
    zerospad = np.zeros(2-PolesN)
    Poles[f"{pair}"] = np.concatenate([np.real(poles),zerospad])

print("Tau:\n ", Tau)
print("Gain:\n ",Gains)
print("Zeros:\n ",Zeros)
print("Poles:\n ",Poles)
print("G0: ",  G(0,Cz_SISO,Ac,Bc_SISO))

test = False
if test:
    Bc_SISO,Cz_SISO = SISO_system(Bc,Cz,1,1)
    M, N = MN_matrix_SISO(Ac,Bc_SISO,Cz_SISO)
    zeros, poles, stable_ctrl, stable_sys = compute_zeros_poles(M, N, Ac)

    print("Kp=", G(0,Cz_SISO,Ac,Bc_SISO)/0.01263919)
    print("Poles:", poles)
    print("System stable (cont.-time):", stable_sys)
    print("Transmission zeros:", zeros)
    print("Controller stable (zeros LHP):", stable_ctrl)

