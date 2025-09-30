import numpy as np
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import matplotlib.pyplot as plt

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

Model_Deterministic = FourTankSystem(R_s*0, R_d*0, p, delta_t)

u_array = np.zeros((2, tf))
u_array[0, :] = F1
u_array[1, :] = F2
d_array = np.zeros((2, tf))
d_array[0, :] = F3
d_array[1, :] = F4

colors = ['dodgerblue', 'tomato', 'limegreen', 'orange']
ls = ['-', '-', '-']

xs = Model_Deterministic.GetSteadyState(x0, u, d)

t, x, u_out, d_out, h = Model_Deterministic.OpenLoop((t0, tf), x0, u_array, d_array)

fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

for i in range(4):
    axes[0,0].plot(
        t/60, h[i, :],
        label=f'Height of Tank {i+1}',
        color=colors[i],
        ls=ls[0]
    )
    if i < 2:
        axes[0,1].plot(
            t/60, u_out[i, :],
            label=f'Flow {i+1} ($u_{i+1}$)',
            color=colors[i], ls=ls[0]
        )
    else:   
        axes[0,1].plot(
            t/60, d_out[i-2, :],
            label=f'Flow {i+1} ($d_{i+1}$)',
            color=colors[i],
            ls=ls[0]
        )


Model_Stochastic = FourTankSystem(R_s, R_d, p, delta_t)
noise = np.random.normal(0, np.sqrt(20), size=(2, tf))
xs = Model_Stochastic.GetSteadyState(x0, u, d)

# Computing steady state of model  2
t, x, u_out, d_out, h = Model_Stochastic.OpenLoop((t0, tf), xs, u_array, d_array + noise)

for i in range(4):
    axes[1,0].plot(
        t/60, h[i, :],
        label=f'Height of Tank {i+1}', 
        color=colors[i], 
        ls=ls[1]
    )
    if i < 2:
        axes[1,1].plot(
            t/60, u_out[i, :],
            label=f'Flow of Tank {i+1} ($u_{i+1}$)',
            color=colors[i],
            ls=ls[1]
        )
    else:
        axes[1,1].plot(
            t/60, d_out[i-2, :],
            label=f'Flow of Tank {i+1} ($d_{i+1}$)',
            color=colors[i],
            ls=ls[1]
        )

# Extending state for model 3
x_extended = np.concatenate([x0, d_array[:, 0]])

# Compute steady state for extended state of model 3 
xs_extended = Model_Stochastic.GetSteadyState(x_extended, u)

# Computing openloop for model 3
t, x, u_out, d_out, h = Model_Stochastic.OpenLoop((t0, tf), xs_extended, u_array, d_array)

for i in range(4):
    axes[2, 0].plot(
        t/60, h[i, :],
        label=f'Height of Tank {i+1}',
        color=colors[i],
        ls=ls[2]
    )
    if i < 2:
        axes[2, 1].plot(
            t/60, u_out[i, :],
            label=f'Flow {i+1} ($u_{i+1}$)',
            color=colors[i],
            ls=ls[2]
        )
    else:
        axes[2, 1].plot(
            t/60, d_out[i-2, :],
            label=f'Flow {i+1} ($d_{i+1}$)',
            color=colors[i],
            ls=ls[2]
        )

axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=2, fancybox=True, shadow=True)
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=2, fancybox=True, shadow=True)

axes[0, 0].set_ylim(0, np.max(h)*1.1)
axes[1, 0].set_ylim(0, np.max(h)*1.1)
axes[2, 0].set_ylim(0, np.max(h)*1.1)
axes[0, 1].set_ylim(0, np.max(u)*1.1)
axes[1, 1].set_ylim(0, np.max(u)*1.1)
axes[2, 1].set_ylim(0, np.max(u)*1.1)

axes[0,0].set_ylabel('Model 1 (Deterministic)\nHeight [m]')
axes[0,1].set_ylabel('Flow [m³/s]')
axes[1,0].set_ylabel('Model 2 (Piecewise constant noise)\nHeight [m]')
axes[1,1].set_ylabel('Flow [m³/s]')
axes[2,0].set_ylabel('Model 3 (SDE)\nHeight [m]')
axes[2,1].set_ylabel('Flow [m³/s]')
axes[2,0].set_xlabel('Time [min]')
axes[2,1].set_xlabel('Time [min]')

axes[0,0].grid(True, linestyle='--', alpha=0.5)
axes[0,1].grid(True, linestyle='--', alpha=0.5)
axes[1,0].grid(True, linestyle='--', alpha=0.5)
axes[1,1].grid(True, linestyle='--', alpha=0.5)
axes[2,0].grid(True, linestyle='--', alpha=0.5)
axes[2,1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(pad=2.0)
plt.savefig('figures/Problem_2.png')
plt.show()