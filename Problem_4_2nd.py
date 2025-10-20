import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# noise level = 0.1, step = 0.5

df = pd.read_csv('figures/Problem4/Problem_4_df.csv')

df = df[df['noise_level'] == 0.0]

k = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        k[i, j] = df[df[f'K_{i+1}{j+1}'] != 0][f'K_{i+1}{j+1}'].values.mean()

T = np.zeros(4)

for i in range(4):
    T[i] = df[df[f'tau_{i+1}'] != 0][f'tau_{i+1}'].values.mean()

print("[")
print(f"\t{k[0, 0]}\t\t\t{k[0, 1]}")
print(f"\t------------------\t\t"+"-"*40+"\t")
print(f"\t   {T[0]} s + 1\t\t\t ({T[1]} s + 1)({T[3]} s + 1)")
print("")
print(f"\t  {k[1, 0]}\t\t\t  {k[1, 1]}")
print(f"\t" + "-"*25 + "\t\t"+"-"*25+"\t")
print(f"\t({T[0]} s + 1)({T[2]} s + 1)\t\t {T[1]} s + 1")
print("]")

F1 = 250
F2 = 325


def G(s): 
    tau = T  # Use T as time constants
    G = np.array([
        [
            k[0, 0]/(tau[0]*s + 1),
            k[0, 1]/((tau[1]*s + 1)*(tau[3]*s + 1))
        ],
        [
            k[1, 0]/((tau[0]*s + 1)*(tau[2]*s + 1)),
            k[1, 1]/(tau[1]*s + 1)
        ]
    ])
    return G


Ac = [
    np.array([
        [-1/T[0]]
    ]),
    np.array([
        [-(T[1] + T[3])/(T[1]*T[3]), 1],
        [-1/(T[1]*T[3]), 0]
    ]),
    np.array([
        [-(T[0] + T[2])/(T[0]*T[2]), 1],
        [-1/(T[0]*T[2]), 0]
    ]),
    np.array([
        [-1/T[1]],
    ])
]

Bc = [
    np.array([
        [k[0, 0]/T[0]]
    ]),
    np.array([
        [0],
        [k[1, 0]/((T[1]*T[3]))]
    ]),
    np.array([
        [0],
        [k[0, 1]/((T[0]*T[2]))]
    ]),
    np.array([
        [k[1, 1]/T[1]]
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
    C_i = Cc[i]
    D_i = Dc[i]
    Mc = np.block([
        [Ac_i, Bc_i],
        [Bc_i.T*0, np.zeros((1, 1))]
    ])

    M = np.expm1(Mc*Ts)
    A_temp = M[:Ac_i.shape[0], :Ac_i.shape[0]]
    B_temp = M[:Ac_i.shape[0], Ac_i.shape[0]:]
    if i == 2:
        print(A_temp)

    A.append(A_temp)
    B.append(B_temp)


def H(i, j, k_arr):
    idx = i*2 + j
    res = []
    for k in k_arr:
        if k == 0:
            res.append(Dc[idx].flatten())
        else:
            res.append((Cc[idx]@np.linalg.matrix_power(A[idx], int(k-1))@B[idx]).flatten())
    return np.array(res)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for i in range(2):
    for j in range(2):
        axes[i, j].step(np.arange(10), H(i, j, np.arange(10)))
        axes[i, j].set_title(f'H_{i+1}{j+1}')
plt.savefig(f'figures/Problem4/Problem_4_Markow.png')
plt.show()
