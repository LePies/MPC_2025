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

def get_markov_mat(k):
    return np.array([
        [H(0, 0, [k])[0][0], H(0, 1, [k])[0][0]],
        [H(1, 0, [k])[0][0], H(1, 1, [k])[0][0]]
    ])


N = 20*60
k_array = np.arange(0, N)

markov_mat = np.zeros((2, 2, N))
for k in k_array:
    markov_mat[:, :, k] = get_markov_mat(k)

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

def Hankel_matrix(markov_params, r, s):

    n, p, m = markov_params.shape
    print(n, p, m)
    print((r*p, s*m))
    H = np.zeros((r*p, s*m))

    # Create Hankel matrix 
    for i in range(r):
        for j in range(s):
            H[i*p:(i+1)*p, j*m:(j+1)*m] = markov_params[i+j]

    # Split Hankel matrix into past and future parts
    #Hp = H[:p*r//2, :m*s//2]  # past
    #Hf = H[p*r//2:, :m*s//2]  # future 
    Hp = H
    # SVD decomposition
    U, S, Vt = np.linalg.svd(Hp, full_matrices=False)

    # Determine system order from singular values 
    n = np.sum(S > 1e-10)  # threshold for numerical stability  

    # Calculate observability and controllability matrices
    Sigma_sqrt = np.diag(np.sqrt(S[:n]))
    Or = U[:, :n] @ Sigma_sqrt  # observability matrix
    Cr = Sigma_sqrt @ Vt[:n, :]  # controllability matrix

    # Extract A, B, C matrices
    C = Or[:p, :]
    A = np.linalg.pinv(Or[:-p, :]) @ Or[p:, :]
    B = Cr[:, :m]

    return H, A, B, C, S

# Compute and display the Hankel matrix
print("Computing Hankel Matrix...")
markov_params = np.zeros((N, 2, 2))
for k in range(markov_mat.shape[2]):
    markov_params[k, :, :] = markov_mat[:, :, k]
H_hankel, A, B, C, S = Hankel_matrix(markov_params, 5, 5)
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

D = np.array([
    [H(0, 0, [0])[0][0], H(0, 1, [0])[0][0]],
    [H(1, 0, [0])[0][0], H(1, 1, [0])[0][0]]
])

data = {
    "A": A,
    "B": B,
    "C": C,
    "D": D,
    "markov_mat": markov_mat
}

print(data["A"].shape)
print(data["B"].shape)
print(data["C"].shape)
print(data["D"].shape)
print(data["markov_mat"].shape)

np.savez("Results/Problem4/Problem_4_estimates.npz", **data)