import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import scipy as sp

p = para.parameters()
a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p
R_s = np.eye(4)*0
R_d = np.eye(2)*0

T_end = 60*20

delta_t = 1
u_array = np.zeros((2, T_end))
u_array[0, :] = 250
u_array[1, :] = 325
d_array = np.zeros((2, T_end))
d_array[0, :] = 100
d_array[1, :] = 120

Model = FourTankSystem(R_s, R_d, p, delta_t, F3 = 100, F4 = 120)
xs = Model.GetSteadyState(np.array([0, 0, 0, 0]), np.array([250, 325]), np.array([100, 120]))

# noise level = 0.1, step = 0.5

df = pd.read_csv('Results/Problem4/Problem_4_df.csv')

df = df[df['noise_level'] == 0.0]

k = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        k[i, j] = df[df[f'K_{i+1}{j+1}'] != 0][f'K_{i+1}{j+1}'].values.mean()

T = np.zeros(4)

T = dict()
T["1 -> 1"] = df[df["Manipulated input"] == 0]["tau_1"].values.mean()
T["2 -> 2"] = df[df["Manipulated input"] == 1]["tau_2"].values.mean()
T["1 -> 4"] = df[df["Manipulated input"] == 0]["tau_4"].values.mean()
T["2 -> 3"] = df[df["Manipulated input"] == 1]["tau_3"].values.mean()


print("[")
print(f"\t{k[0, 0]}\t\t\t{k[0, 1]}")
print(f"\t------------------\t\t"+"-"*40+"\t")
print(f"\t   {T["1 -> 1"]} s + 1\t\t\t ({T["2 -> 2"]} s + 1)({T["1 -> 4"]} s + 1)")
print("")
print(f"\t  {k[1, 0]}\t\t\t  {k[1, 1]}")
print(f"\t" + "-"*25 + "\t\t"+"-"*25+"\t")
print(f"\t({T["1 -> 1"]} s + 1)({T["2 -> 3"]} s + 1)\t\t {T["2 -> 2"]} s + 1")
print("]")

F1 = 250
F2 = 325


def G(s): 
    G = np.array([
        [
            k[0, 0]/(T["1 -> 1"]*s + 1),
            k[0, 1]/((T["1 -> 2"]*s + 1)*(T["1 -> 4"]*s + 1))
        ],
        [
            k[1, 0]/((T["2 -> 1"]*s + 1)*(T["2 -> 3"]*s + 1)),
            k[1, 1]/(T["2 -> 2"]*s + 1)
        ]
    ])
    return G


Ac = [
    np.array([
        [-1/T["1 -> 1"]]
    ]),
    np.array([
        [-(T["2 -> 2"] + T["1 -> 4"])/(T["2 -> 2"]*T["1 -> 4"]), 1],
        [-1/(T["2 -> 2"]*T["1 -> 4"]), 0]
    ]),
    np.array([
        [-(T["1 -> 1"] + T["2 -> 3"])/(T["1 -> 1"]*T["2 -> 3"]), 1],
        [-1/(T["1 -> 1"]*T["2 -> 3"]), 0]
    ]),
    np.array([
        [-1/T["2 -> 2"]],
    ])
]

Bc = [
    np.array([
        [k[0, 0]/T["1 -> 1"]]
    ]),
    np.array([
        [0],
        [k[1, 0]/((T["2 -> 2"]*T["1 -> 4"]))]
    ]),
    np.array([
        [0],
        [k[0, 1]/((T["1 -> 1"]*T["2 -> 3"]))]
    ]),
    np.array([
        [k[1, 1]/T["2 -> 2"]]
    ])
]

Cc = [
    np.array([
        [
            1
        ]
    ]),
    np.array([
        [
            1, 0
        ]
    ]),
    np.array([
        [
            1, 0
        ]
    ]),
    np.array([
        [
            1
        ]
    ])
]

Dc = np.array([
    0,
    0,
    0,
    0
])

Ts = 1

A = []
B = []

for i in range(4):
    Ac_i = Ac[i]
    Bc_i = Bc[i]

    n = Ac_i.shape[0]
    m = Bc_i.shape[1]

    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac_i
    M[:n, n:] = Bc_i
    Md = sp.linalg.expm(M * Ts)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]

    A.append(Ad)
    B.append(Bd)

print(Ac[0])
print(A[0])
print(Bc[0])
print(B[0])

def H(i, j, k_arr):
    idx = i*2 + j
    res = []
    Ad = A[idx]
    A_pow = np.eye(Ad.shape[0])
    for k in k_arr:
        if k == 0:
            res.append(Dc[idx].flatten())
        else:
            res.append((Cc[idx]@A_pow@B[idx]).flatten())
            A_pow = A_pow @ Ad
    return np.array(res)

t_array = np.arange(0,T_end)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for i in range(2):
    for j in range(2):
        axes[i, j].step(t_array/60, H(i, j, t_array))
        axes[i, j].set_title(rf'$H_{{{i+1}{j+1}}}$')
        axes[i, j].grid(True, alpha=0.5)
        axes[i, j].set_xlabel('Time [min]')
        axes[i, j].set_ylabel('Markov parameters')
plt.savefig(f'figures/Problem4/Problem_4_Markow.png')
plt.close()

S_11 = np.cumsum(H(0, 0, t_array))
S_12 = np.cumsum(H(0, 1, t_array))
S_21 = np.cumsum(H(1, 0, t_array))
S_22 = np.cumsum(H(1, 1, t_array))

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].step(t_array/60, S_11, label="Step parameters", color='black', ls='--')
axes[0, 0].set_title(r'$S_{11}$')
axes[0, 0].set_xlabel('Time [min]')
axes[0, 0].set_ylabel('Step response')
axes[0, 1].step(t_array/60, S_12, label="Step parameters", color='black', ls='--')
axes[0, 1].set_title(r'$S_{12}$')
axes[0, 1].set_xlabel('Time [min]')
axes[0, 1].set_ylabel('Step response')
axes[1, 0].step(t_array/60, S_21, label="Step parameters", color='black', ls='--')
axes[1, 0].set_title(r'$S_{21}$')
axes[1, 0].set_xlabel('Time [min]')
axes[1, 0].set_ylabel('Step response')
axes[1, 1].step(t_array/60, S_22, label="Step parameters", color='black', ls='--')
axes[1, 1].set_title(r'$S_{22}$')
axes[1, 1].set_xlabel('Time [min]')
axes[1, 1].set_ylabel('Step response')

steps = [0.1, 0.25, 0.5]
for step in steps:
    u_array_1 = np.copy(u_array)
    u_array_1[0] = u_array_1[0]*(1 + step)
    t, x, u, d, h = Model.OpenLoop((0, T_end), xs, u_array_1, d_array)

    hs = xs/(rho*np.array([A1,A2,A3,A4]))
    h_norm = (h - hs[:, None]) / (F1*step)

    axes[0, 0].plot(t/60, h_norm[0, :], label=f"Sim (step {step})")
    axes[0, 1].plot(t/60, h_norm[1, :], label=f"Sim (step {step})")

    u_array_2 = np.copy(u_array)
    u_array_2[1] = u_array_2[1]*(1 + step)
    t, x, u, d, h = Model.OpenLoop((0, T_end), xs, u_array_2, d_array)

    hs = xs/(rho*np.array([A1,A2,A3,A4]))
    h_norm = (h - hs[:, None]) / (F2*step)

    axes[1, 0].plot(t/60, h_norm[0, :], label=f"Sim (step {step})")
    axes[1, 1].plot(t/60, h_norm[1, :], label=f"Sim (step {step})")
# plt.savefig(f'figures/Problem4/Problem_4_Stepresponse.png')
axes[0, 0].grid(alpha = 0.25)
axes[1, 0].grid(alpha = 0.25)
axes[0, 1].grid(alpha = 0.25)
axes[1, 1].legend(loc='center right')
axes[1, 1].grid(alpha = 0.25)
plt.savefig(f'figures/Problem4/Problem_4_Stepresponse.png')
plt.close()

def Hankel_Matrix(p_rows=10, q_cols=10):
    """
    Compute the Hankel matrix from Markov parameters
    
    Parameters:
    p_rows: number of block rows
    q_cols: number of block columns
    
    Returns:
    H_hankel: Hankel matrix of shape (p*ny, q*nu) where ny=2, nu=2
    """
    ny = 2  # number of outputs
    nu = 2  # number of inputs
    
    # Initialize Hankel matrix
    H_hankel = np.zeros((p_rows * ny, q_cols * nu))
    
    # Fill the Hankel matrix with Markov parameters
    for i in range(p_rows):
        for j in range(q_cols):
            # Get Markov parameters for time step i+j
            h_ij = np.zeros((ny, nu))
            
            # Fill the block at position (i,j)
            for out_idx in range(ny):
                for in_idx in range(nu):
                    # Get the Markov parameter for output out_idx, input in_idx, at time i+j
                    markov_param = H(out_idx, in_idx, [i + j])
                    h_ij[out_idx, in_idx] = markov_param[0][0]
            
            # Place the block in the Hankel matrix
            H_hankel[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = h_ij
    
    return H_hankel

# Compute and display the Hankel matrix
print("Computing Hankel Matrix...")
H_hankel = Hankel_Matrix(p_rows=25, q_cols=25)
# print(H_hankel)

# Compute SVD of Hankel matrix for system order estimation
K, S, Lt = np.linalg.svd(H_hankel)

# Plot singular values
plt.figure(figsize=(10, 6))
plt.semilogy(S[:], linestyle='--', color='black', marker='o', markersize=4)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Values of Hankel Matrix')
plt.grid(True, alpha=0.3)
plt.savefig('figures/Problem4/Problem_4_Hankel_SingularValues.png')
plt.close()

# Estimate system order based on singular values
# Look for a significant drop in singular values
threshold = 0.01  # 1% of the largest singular value
significant_singular_values = S[S > threshold * S[0]]
estimated_order = len(significant_singular_values)

N = estimated_order

S_sqrt = np.diag(np.sqrt(S[:N]))


Oo = K[:, :N]@S_sqrt
Cc = S_sqrt@Lt[:N, :]

B = Cc[:,:2]
C = Oo[:2, :]
A = np.linalg.inv(S_sqrt)@K[:, :N].T@H_hankel@Lt[:N, :].T@np.linalg.inv(S_sqrt)
A_test = np.linalg.pinv(Oo) @ Oo
print(np.linalg.norm(A - A_test))

D = np.array([
    [H(0, 0, [0])[0][0], H(0, 1, [0])[0][0]],
    [H(1, 0, [0])[0][0], H(1, 1, [0])[0][0]]
])

data = {
    "A": A,
    "B": B,
    "C": C,
    "D": D
}

print(data["A"].shape)
print(data["B"].shape)
print(data["C"].shape)
print(data["D"].shape)

np.savez("Results/Problem4/Problem_4_estimates.npz", **data)