import numpy as np
from src.FourTankSystem import FourTankSystem
from scipy.linalg import eig
from params.initialize import initialize
import pandas as pd
import matplotlib.pyplot as plt

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
    
def markov_parameters(Ad, Bd, C, D=None, N=100):

    n = Ad.shape[0]
    p, m = C.shape[0], Bd.shape[1]
    if D is None:
        D = np.zeros((p, m))
    H = []
    H.append(D)  # H_0
    A_pow = np.eye(n)  # Ad^0
    for k in range(1, N):
        Hk = C @ A_pow @ Bd
        H.append(Hk)
        A_pow = Ad @ A_pow

    H = np.stack(H, axis=0) # convert list to array
    return H  

x0, u, d, p , R_s, R_d, delta_t = initialize()

#%% Linearize continous time
Model_Stochastic = FourTankSystem(R_s*0, R_d*0, p, delta_t)
x0 = np.concatenate((x0, np.zeros(2)))  
xs = Model_Stochastic.GetSteadyState(x0, u)

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
        pole_relevant = poles[1]
    else:
        factor = poles[0]
        pole_relevant = poles[0]
    
    # Tau
    TauN = len(np.array([1/np.real(factor)]))
    zerospad = np.zeros(2-TauN)
    Tau[f"{pair}"] = np.concatenate([np.array([1/np.real(pole_relevant)]),zerospad])
    
    # Gain 
    GainN = len((G(0,Cz_SISO,Ac,Bc_SISO)/np.abs(factor))[0])
    zerospad = np.zeros(2-GainN)
    Gains[f"{pair}"] = np.concatenate([G(0,Cz_SISO,Ac,Bc_SISO)[0],zerospad])
    
    # Zeros
    ZerosN = len(np.real(zeros))
    zerospad = np.zeros(2-ZerosN)
    Zeros[f"{pair}"] = np.concatenate([np.real(zeros),zerospad])
    
    # Poles
    PolesN = len(np.real(poles))
    zerospad = np.zeros(2-PolesN)
    Poles[f"{pair}"] = np.concatenate([np.real(poles),zerospad])

Tau.to_csv(r"Results\Problem5\Tau.txt", sep='\t', index=True)
Gains.to_csv(r"Results\Problem5\Gain.txt", sep='\t', index=True)
Zeros.to_csv(r"Results\Problem5\Zeros.txt", sep='\t', index=True)
Poles.to_csv(r"Results\Problem5\Poles.txt", sep='\t', index=True)

#%% Linearize Discrete time 
Ts = 1
Ad, Bd, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs, d, Ts)

# Markov parameters 
N = 100
H = markov_parameters(Ad, Bd, Cz, D=np.zeros((Cz.shape[0], Bd.shape[1])), N=N)

N, p, m = H.shape
time = np.arange(N)

# Impulse response (Markov parameters) 
fig, axes = plt.subplots(p, m, figsize=(10, 8), sharex=True)
fig.suptitle("Discrete-time Markov Parameters (Impulse Response Coefficients)", fontsize=14)

for i in range(p):
    for j in range(m):
        hij = H[:, i, j]
        ax = axes[i, j]
        ax.plot(time, hij)
        ax.set_title(f"y{i+1} \u2190 u{j+1}")
        ax.grid(True)

        # The leftmost column
        if j == 0:
            ax.set_ylabel(rf"$y_k$  $(H_k)_{{{i+1},{j+1}}}$")

        else:
            ax.set_ylabel("")

        # The bottom row
        if i == p - 1:
            ax.set_xlabel("time")
        else:
            ax.set_xlabel("")

plt.tight_layout()
plt.savefig(r"Figures\Problem5\Problem_5_Markov_Parameters.png")

# Step responses
S = np.cumsum(H, axis=0)  # (N, p, m)

fig, axes = plt.subplots(p, m, figsize=(10, 8), sharex=True)
fig.suptitle("Discrete-time Step Responses (Unit Step Input)\n Cumulative Markov Parameters", fontsize=14)

for i in range(p):
    for j in range(m):
        sij = S[:, i, j]
        ax = axes[i, j]
        ax.plot(time, sij)
        ax.set_title(f"y{i+1} \u2190 u{j+1}")
        ax.grid(True)

        # The leftmost column
        if j == 0:
            ax.set_ylabel(rf"$y_k$  $(H_k)_{{{i+1},{j+1}}}$")
        else:
            ax.set_ylabel("")

        # The bottom row
        if i == p - 1:
            ax.set_xlabel("time")
        else:
            ax.set_xlabel("")

plt.tight_layout()
plt.savefig(r"Figures\Problem5\Problem_5_Stepresponse.png")

