import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import scipy as sp
import sys

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
df_d = pd.read_csv('Results/Problem4/Problem_4_df_d.csv')

df = df[df['noise_level'] == 0.0]
df_d = df_d[df_d['noise_level'] == 0.0]

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

tau_d1_1 = df_d[df_d["Manipulated input"] == 0]["tau_1"].values.mean()
tau_d1_2 = df_d[df_d["Manipulated input"] == 0]["tau_2"].values.mean()
tau_d2_1 = df_d[df_d["Manipulated input"] == 1]["tau_3"].values.mean()
tau_d2_2 = df_d[df_d["Manipulated input"] == 1]["tau_4"].values.mean()

k_d1 = df_d[df_d["Manipulated input"] == 0]["K_11"].values.mean()
k_d2 = df_d[df_d["Manipulated input"] == 1]["K_22"].values.mean()

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
    ]),
    np.array([
        [-(tau_d1_1 + tau_d1_2)/(tau_d1_1*tau_d1_2), 1],
        [-1/(tau_d1_1*tau_d1_2), 0]
    ]),
    np.array([
        [0]
    ]),
    np.array([
        [0]
    ]),
    np.array([
        [-(tau_d2_1 + tau_d2_2)/(tau_d2_1*tau_d2_2), 1],
        [-1/(tau_d2_1*tau_d2_2), 0]
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
    ]),
    np.array([
        [0],
        [k_d1/(tau_d1_1*tau_d1_2)]
    ]),
    np.array([
        [0]
    ]),
    np.array([
        [0]
    ]),
    np.array([
        [0],
        [k_d2/(tau_d2_1*tau_d2_2)]
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
    ]),
    np.array([
        [
            1
        ]
    ]),
    np.array([
        [
            1, 0
        ]
    ])
]

Dc = np.array([
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
])

Ts = 1

A = []
B = []

for i in range(8):
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

def get_markov_mat(k):
    if len(sys.argv) > 1 and sys.argv[1] == "d":
        return np.array([
            [H(0, 0, [k])[0][0], H(0, 1, [k])[0][0], H(2, 0, [k])[0][0], H(3, 0, [k])[0][0]],
            [H(1, 0, [k])[0][0], H(1, 1, [k])[0][0], H(2, 1, [k])[0][0], H(3, 1, [k])[0][0]]
        ])
    else:
        return np.array([
            [H(0, 0, [k])[0][0], H(1, 0, [k])[0][0]],
            [H(0, 1, [k])[0][0], H(1, 1, [k])[0][0]]
        ])

t_array = np.arange(0,T_end)
tk_array = np.arange(0,T_end*100)
N = len(tk_array)

if len(sys.argv) > 1 and sys.argv[1] == "d":
    markov_mat = np.zeros((2, 4, N))
else:
    markov_mat = np.zeros((2, 2, N))


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for i in range(2):
    for j in range(2):
        H_ij = H(i, j, tk_array)
        axes[i, j].step(t_array/60, H_ij[:T_end,0])
        markov_mat[i, j, :] = H_ij[:,0]
        axes[i, j].set_title(rf'$H_{{{i+1}{j+1}}}$')
        axes[i, j].grid(True, alpha=0.5)
        axes[i, j].set_xlabel('Time [min]')
        axes[i, j].set_ylabel('Markov parameters')
plt.savefig(f'figures/Problem4/Problem_4_Markow.png')
plt.close()

# Populate disturbance columns when option "d" is used
if len(sys.argv) > 1 and sys.argv[1] == "d":
    # Column 2: disturbance 1 -> outputs
    # H(2,0) = idx 4: disturbance 1 -> output 1
    H_d1_o1 = H(2, 0, tk_array)
    markov_mat[0, 2, :] = H_d1_o1[:,0]
    # H(2,1) = idx 5: disturbance 1 -> output 2 (zero, but populate anyway)
    H_d1_o2 = H(2, 1, tk_array)
    markov_mat[1, 2, :] = H_d1_o2[:,0]
    
    # Column 3: disturbance 2 -> outputs
    # H(3,0) = idx 6: disturbance 2 -> output 1 (zero, but populate anyway)
    H_d2_o1 = H(3, 0, tk_array)
    markov_mat[0, 3, :] = H_d2_o1[:,0]
    # H(3,1) = idx 7: disturbance 2 -> output 2
    H_d2_o2 = H(3, 1, tk_array)
    markov_mat[1, 3, :] = H_d2_o2[:,0]

S_11 = np.cumsum(H(0, 0, t_array))
S_12 = np.cumsum(H(0, 1, t_array))
S_21 = np.cumsum(H(1, 0, t_array))
S_22 = np.cumsum(H(1, 1, t_array))
S_13 = np.cumsum(H(2, 0, t_array))
S_24 = np.cumsum(H(3, 1, t_array))

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
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

axes[0, 2].step(t_array/60, S_13, label="Step parameters", color='black', ls='--')
axes[0, 2].set_title(r'$S_{13}$')
axes[0, 2].set_xlabel('Time [min]')
axes[0, 2].set_ylabel('Step response')
axes[1, 2].step(t_array/60, S_24, label="Step parameters", color='black', ls='--')
axes[1, 2].set_title(r'$S_{24}$')
axes[1, 2].set_xlabel('Time [min]')
axes[1, 2].set_ylabel('Step response')

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

for step in steps:
    d_array_1 = np.copy(d_array)
    d_array_1[0] = d_array_1[0]*(1 + step)

    t, x, u, d, h = Model.OpenLoop((0, T_end), xs, u_array, d_array_1)
    hs = xs/(rho*np.array([A1,A2,A3,A4]))
    h_norm = (h - hs[:, None]) / (d_array[0]*step)
    axes[0, 2].plot(t/60, h_norm[0, :], label=f"Sim (step {step})")
    
    
    d_array_2 = np.copy(d_array)
    d_array_2[1] = d_array_2[1]*(1 + step)

    t, x, u, d, h = Model.OpenLoop((0, T_end), xs, u_array, d_array_2)
    hs = xs/(rho*np.array([A1,A2,A3,A4]))
    h_norm = (h - hs[:, None]) / (d_array[1]*step)
    axes[1, 2].plot(t/60, h_norm[1, :], label=f"Sim (step {step})")

axes[0, 2].grid(alpha = 0.25)
axes[1, 2].grid(alpha = 0.25)

plt.savefig(f'figures/Problem4/Problem_4_Stepresponse.png')
plt.close()

def Hankel_matrix(markov_params, r, s):

    two_n_total, p, m = markov_params.shape
    n_total = two_n_total // 2
    n = n_total - 1
    print(f"Markov params shape: ({n_total}, {p}, {m})")
    print(f"Hankel matrix will be: ({r*p}, {s*m})")
    
    H = np.zeros((n_total*p, n_total*m))

    # Properly create the (block) Hankel matrix of shape (r*p, s*m)
    # H = [[h0, h1, ..., h_{s-1}],
    #      [h1, h2, ..., h_s],
    #      ...
    #      [h_{r-1}, ..., h_{r+s-2}]]
    # Each block h_k is (p x m)
    H = np.zeros((n_total * p, n_total * m))
    
    # Fill the Hankel matrix block by block
    for i in range(n_total):
        for j in range(n_total):
            # Each block (i,j) gets Markov parameter M[i+j]
            H[i*p:(i+1)*p, j*m:(j+1)*m] = markov_params[i+j, :, :]

    print(H.shape)
    H_tilde = H[1:, 1:]
    H = H[:-1, :-1]

    # SVD decomposition
    K, S, Lt = sp.linalg.svd(H)

    # Determine system order from singular values using relative threshold
    # Use relative threshold to handle different scales
    threshold = S[0] * 1e-6  # Relative to largest singular value (changed back from 1e-10)
    n = np.sum(S > threshold)
    
    print(f"Threshold: {threshold:.6e} (S[0] * 1e-6)")
    print(f"Detected system order: {n}")
    
    # Note: The minimal realization might be smaller than expected due to shared dynamics
    # For a 2x2 system (2 inputs, 2 outputs), you might expect 4 states, but shared
    # time constants can reduce this. The singular values show the true system order.
    # If you want to force a specific order, you can manually set n here.
    
    # Calculate observability and controllability matrices
    Sigma_sqrt = np.diag(np.sqrt(S[:n]))
    Or = K[:, :n] @ Sigma_sqrt  # observability matrix
    Cr = Sigma_sqrt @ Lt[:n, :]  # controllability matrix

    # Extract C matrix (first p rows of observability matrix)
    C = Or[:p, :]

    # Extract A matrix using shift-invariance property
    # Or has structure: [C; CA; CA^2; ...; CA^(r-1)]
    # Or[:-p, :] = [C; CA; ...; CA^(r-2)]
    # Or[p:, :] = [CA; CA^2; ...; CA^(r-1)]
    # Relationship: Or[p:, :] = Or[:-p, :] @ A
    # Therefore: A = pinv(Or[:-p, :]) @ Or[p:, :]
    Or_up = Or[:-p, :]  # Remove last p rows
    Or_down = Or[p:, :]  # Remove first p rows
    
    # Check condition number to ensure well-conditioned
    cond_num = np.linalg.cond(Or_up)
    if cond_num > 1e12:
        print(f"Warning: High condition number for A extraction: {cond_num:.2e}")
    
    A = np.linalg.pinv(Or_up) @ Or_down
    
    # Extract B matrix (first m columns of controllability matrix)
    # The controllability matrix Cr has structure: Cr = [B, AB, A^2B, ...]
    # So the first m columns give us B
    B = Cr[:, :m]

    return H, A, B, C, S

# Compute and display the Hankel matrix
print("Computing Hankel Matrix...")
# Note: markov_mat[k] contains H_k (k=0,1,2,...)
# For Hankel matrix, we need H_1, H_2, H_3, ... (skip H_0 = D)
if len(sys.argv) > 1 and sys.argv[1] == "d":
    markov_params = np.zeros((N, 2, 4))
else:
    markov_params = np.zeros((N, 2, 2))
for k in range(markov_mat.shape[2]):
    markov_params[k, :, :] = markov_mat[:, :, k]

# So we shift: markov_params_hankel[k] = markov_params[k+1] for k=0,1,2,...
markov_params_hankel = markov_params[1:]  # Skip H_0, take H_1 through H_N
# Increase Hankel matrix size to better capture system dynamics
# For 2x2 system: r=8, s=8 gives (16, 16) matrix (good balance)
N = 20
H_hankel, A, B, C, S = Hankel_matrix(markov_params_hankel[:N, :, :], 8, 8)
# print(H_hankel)

# Compute SVD of Hankel matrix for system order estimation
K, S, Lt = np.linalg.svd(H_hankel)

# Plot singular values
plt.figure()
plt.semilogy(S[:], linestyle='--', color='black', marker='o', markersize=4)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Values of Hankel Matrix')
plt.grid(True, alpha=0.3)
plt.savefig('figures/Problem4/Problem_4_Hankel_SingularValues.png')
plt.close()

# Extract D matrix (first Markov parameter, k=0)
# D should be markov_params[0] which is the feedthrough matrix
D = markov_params[0]

def get_Q(Ad, Gc):
    A_cont = sp.linalg.logm(Ad) / delta_t

    Mat = np.block([[-A, Gc@Gc.T], [np.zeros((A.shape[0], A.shape[1])), A.T]])
    PHI = sp.linalg.expm(Mat * delta_t)

    Phi_11 = PHI[:A.shape[0], :A.shape[0]]
    Phi_12 = PHI[:A.shape[0], A.shape[0]:]
    Phi_22 = PHI[A.shape[0]:, A.shape[0]:]

    return Phi_22.T @ Phi_12

if len(sys.argv) > 1 and sys.argv[1] == "d":
    Ed = B[:, 2:]
    B = B[:, :2]
    Dd = D[:, 2:]
    D = D[:, :2]

    A = np.block([
        [A,              Ed],
        [np.zeros((Ed.shape[1], A.shape[0])), np.eye(Ed.shape[1])]
    ])


    B = np.block([
        [B],
        [np.zeros((Ed.shape[1], B.shape[1]))]
    ])
    
    C = np.block([C, Dd])

    Gc = np.zeros((A.shape[0], Ed.shape[1]))
    Gc[-2,-2] = 1
    Gc[-1,-1] = 1

    Q = get_Q(A, Gc)
    data = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "markov_mat": markov_mat,
        "E": Ed,
        "Dd": Dd,
        "Q": Q
    }
    print("E shape: ", Ed.shape)
    print("Dd shape: ", Dd.shape)
    np.savez("Results/Problem4/Problem_4_estimates_d.npz", **data)
else:
    data = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "markov_mat": markov_mat
    }
    np.savez("Results/Problem4/Problem_4_estimates.npz", **data)



print("A shape: ", data["A"].shape)
print("B shape: ", data["B"].shape)
print("C shape: ", data["C"].shape)
print("D shape: ", data["D"].shape)
print("markov_mat shape: ", data["markov_mat"].shape)