
from traceback import print_tb
import numpy as np
import params.parameters_tank as para
import matplotlib.pyplot as plt
import src.PIDcontrolor as pid
from src.FourTankSystem import FourTankSystem

t0 = 0
tf = 20*90
m10 = 0
m20 = 0
m30 = 0
m40 = 0
F1 = 250
F2 = 325
F3 = 100
F4 = 10
x0 = np.array([m10,m20,m30,m40])
u = np.array([F1,F2])
d = np.array([F3,F4])
p = para.parameters()
a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p

delta_t = 1

# Define distrubances
goal = np.array([10.0, 10.0])  # Height of Tank 1 and 2
u0 = np.array([F1, F2])
Rvv = np.eye(4)*0.01
umin = 1
umax = 10000

Model = FourTankSystem(Rvv, Rvv, p, delta_t)

xs = Model.GetSteadyState(x0, u, d)
print("Steady state (m):", xs)
print("Steady state (h):", xs/(rho*np.array([A1,A2,A3,A4])))

state_0 = np.concatenate([x0, d])


setpoint = goal

controller_p = pid.PIDController(1, 0, 0, setpoint, delta_t, umin, umax)

t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_p)

colors = ['dodgerblue', 'tomato', 'limegreen', 'orange']
ls = ['-', '-', '-']

fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
for i in range(4):
    axes[0, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[0])
    if i < 2:
        axes[0, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[0])
    else:
        axes[0, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[0])

axes[0, 0].set_ylim(0, np.max(h)*1.1)
axes[0, 1].set_ylim(0, np.max(u)*1.1)
axes[0, 0].set_ylabel('K_p = 1\nHeight [m]')
axes[0, 1].set_ylabel('Flow [m³/s]')
axes[0, 0].grid(True, linestyle='--', alpha=0.5)
axes[0, 1].grid(True, linestyle='--', alpha=0.5)


controller_pi = pid.PIDController(5, 0, 0, setpoint, delta_t, umin, umax)

t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pi)

for i in range(4):
    axes[1, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[1])
    if i < 2:
        axes[1, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])
    else:
        axes[1, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])

axes[1, 0].set_ylim(0, np.max(h)*1.1)
axes[1, 1].set_ylim(0, np.max(u)*1.1)
axes[1, 0].set_ylabel('PI Controller\nHeight [m]')
axes[1, 1].set_ylabel('Flow [m³/s]')
axes[1, 0].grid(True, linestyle='--', alpha=0.5)
axes[1, 1].grid(True, linestyle='--', alpha=0.5)

controller_pid = pid.PIDController(5, 0, 0, setpoint, delta_t, umin, umax)

t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pid)

for i in range(4):
    axes[2, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[2])
    if i < 2:
        axes[2, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])
    else:
        axes[2, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])

axes[2, 0].set_ylim(0, np.max(h)*1.1)
axes[2, 1].set_ylim(0, np.max(u)*1.1)
axes[2, 0].set_ylabel('PID Controller\nHeight [m]')
axes[2, 1].set_ylabel('Flow [m³/s]')
axes[2, 0].set_xlabel('Time [m]')
axes[2, 1].set_xlabel('Time [m]')
axes[2, 0].grid(True, linestyle='--', alpha=0.5)
axes[2, 1].grid(True, linestyle='--', alpha=0.5)

axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
        ncol=2, fancybox=True, shadow=True)
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
        ncol=2, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('figures/Problem_3.png')
plt.show()
