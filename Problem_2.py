import numpy as np
from FourTankSystem import FourTankSystem
import parameters_tank as para
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

t, x, u, d, h = Model_Deterministic.OpenLoop((t0, tf), x0, u_array, d_array)

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
axes[0].plot(t, h[0, :], label='H1')
axes[0].plot(t, h[1, :], label='H2')
axes[0].plot(t, h[2, :], label='H3')
axes[0].plot(t, h[3, :], label='H4')
axes[0].set_title('Height of Tanks (Deterministic)')
axes[0].legend()

axes[1].plot(t, u[0, :], label='F1')
axes[1].plot(t, u[1, :], label='F2')
axes[1].plot(t, d[0, :], label='F3')
axes[1].plot(t, d[1, :], label='F4')
axes[1].set_title('Flow of Tanks (Deterministic)')
axes[1].legend()

plt.tight_layout()
plt.show()

Model_Stochastic = FourTankSystem(R_s, R_d, p, delta_t)

noise = np.random.normal(0, np.sqrt(20), size=(2, tf))
t, x, u, d, h = Model_Stochastic.OpenLoop((t0, tf), x0, u_array, d_array + noise)

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

axes[0].plot(t, h[0, :], label='H1')
axes[0].plot(t, h[1, :], label='H2')
axes[0].plot(t, h[2, :], label='H3')
axes[0].plot(t, h[3, :], label='H4')
axes[0].set_title('Height of Tanks (Stochastic)')
axes[0].legend()

axes[1].plot(t, u[0, :], label='F1')
axes[1].plot(t, u[1, :], label='F2')
axes[1].plot(t, d[0, :], label='F3')
axes[1].plot(t, d[1, :], label='F4')
axes[1].set_title('Flow of Tanks (Stochastic)')
axes[1].legend()

plt.tight_layout()
plt.show()

state0 = np.concatenate([x0, d_array[:, 0]])

t, x, u, d, h = Model_Stochastic.OpenLoop((t0, tf), state0, u_array)

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
axes[0].plot(t, h[0, :], label='H1')
axes[0].plot(t, h[1, :], label='H2')
axes[0].plot(t, h[2, :], label='H3')
axes[0].plot(t, h[3, :], label='H4')
axes[0].set_title('Height of Tanks (Stochastic)')
axes[0].legend()

axes[1].plot(t, u[0, :], label='F1')
axes[1].plot(t, u[1, :], label='F2')
axes[1].plot(t, d[0, :], label='F3')
axes[1].plot(t, d[1, :], label='F4')
axes[1].set_title('Flow of Tanks (Stochastic)')
axes[1].legend()

plt.tight_layout()
plt.show()