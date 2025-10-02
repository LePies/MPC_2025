from scipy.optimize import fsolve
import numpy as np
import params.parameters_tank as para
import matplotlib.pyplot as plt
import src.PIDcontrolor as pid
from src.FourTankSystem import FourTankSystem
import sys

t0 = 0
tf = 20*90
m10 = 0
m20 = 0
m30 = 0
m40 = 0
F1 = 250
F2 = 325
F3 = 100
F4 = 100
x0 = np.array([m10,m20,m30,m40])
u = np.array([F1,F2])
d = np.array([F3,F4])
p = para.parameters()
a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p

delta_t = 1

# Define distrubances
goal = np.array([15.0, 15.0])  # Height of Tank 1 and 2
u0 = np.array([F1, F2])
Rvv = np.eye(4)*0.01
umin = 1
umax = 10000

Model = FourTankSystem(Rvv, Rvv, p, delta_t, F3 = F3, F4 = F4)

xs = Model.GetSteadyState(x0, u, d)
print("Steady state (m):", xs)
print("Steady state (h):", xs/(rho*np.array([A1,A2,A3,A4])))

state_0 = np.concatenate([xs, d])

if sys.argv[1] == 'test':
    u_array = np.zeros((2, tf))
    u_array[0, :] = F1
    u_array[1, :] = F2
    
    print(Model.StateEquation(0, state_0, u_array[:, 0]))
    xs = fsolve(lambda x: Model.StateEquation(0, x, u_array[:, 0]), state_0)
    print(xs)
    print(Model.StateEquation(0, xs, u_array[:, 0]))
    sys.exit()
    t,x,y,d,h = Model.OpenLoop((t0, tf), state_0, u_array)
    
    plt.plot(t/60, h[0, :], label='Height of Tank 1', color='dodgerblue', ls='-')
    plt.plot(t/60, h[1, :], label='Height of Tank 2', color='tomato', ls='-')
    plt.plot(t/60, h[2, :], label='Height of Tank 3', color='limegreen', ls='-')
    plt.plot(t/60, h[3, :], label='Height of Tank 4', color='orange', ls='-')
    
    plt.plot([t[0]/60, t[-1]/60], [xs[0]/(rho*A1), xs[0]/(rho*A2)], label='Setpoint', color='black', ls='--')
    plt.plot([t[0]/60, t[-1]/60], [xs[1]/(rho*A1), xs[1]/(rho*A2)], label='Setpoint', color='black', ls='--')
    plt.plot([t[0]/60, t[-1]/60], [xs[2]/(rho*A3), xs[2]/(rho*A4)], label='Setpoint', color='black', ls='--')
    plt.plot([t[0]/60, t[-1]/60], [xs[3]/(rho*A3), xs[3]/(rho*A4)], label='Setpoint', color='black', ls='--')
    
    plt.legend()
    plt.xlabel('Time [m]')
    plt.ylabel('Height [m]')
    plt.show()
    sys.exit()

setpoint = goal

Kp = 10

if sys.argv[1] == 'p':
    controller_p = pid.PIDController(5, 0, 0, setpoint, delta_t, umin, umax)

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
    axes[0, 0].set_ylabel('K_p = 5\nHeight [m]')
    axes[0, 1].set_ylabel('Flow [m³/s]')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)


    controller_p = pid.PIDController(Kp, 0, 0, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_p)

    for i in range(4):
        axes[1, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[1])
        if i < 2:
            axes[1, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])
        else:
            axes[1, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])

    axes[1, 0].set_ylim(0, np.max(h)*1.1)
    axes[1, 1].set_ylim(0, np.max(u)*1.1)
    axes[1, 0].set_ylabel(f'K_p = {Kp}\nHeight [m]')
    axes[1, 1].set_ylabel('Flow [m³/s]')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    controller_p = pid.PIDController(100, 0, 0, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_p)

    for i in range(4):
        axes[2, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[2])
        if i < 2:
            axes[2, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])
        else:
            axes[2, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])

    axes[2, 0].set_ylim(0, np.max(h)*1.1)
    axes[2, 1].set_ylim(0, np.max(u)*1.1)
    axes[2, 0].set_ylabel('K_p = 500\nHeight [m]')
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
    plt.savefig('figures/Problem_3_Kp.png')
    # plt.show()

Ki = 0.1
if sys.argv[1] == 'pi':
    controller_pi = pid.PIDController(Kp, 0.05, 0, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pi)

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
    axes[0, 0].set_ylabel('K_i = 0.05\nHeight [m]')
    axes[0, 1].set_ylabel('Flow [m³/s]')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)


    controller_pi = pid.PIDController(Kp, Ki, 0, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pi)

    for i in range(4):
        axes[1, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[1])
        if i < 2:
            axes[1, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])
        else:
            axes[1, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])

    axes[1, 0].set_ylim(0, np.max(h)*1.1)
    axes[1, 1].set_ylim(0, np.max(u)*1.1)
    axes[1, 0].set_ylabel(f'K_i = {Ki}\nHeight [m]')
    axes[1, 1].set_ylabel('Flow [m³/s]')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    controller_pi = pid.PIDController(Kp, 5, 0, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pi)

    for i in range(4):
        axes[2, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[2])
        if i < 2:
            axes[2, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])
        else:
            axes[2, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])

    axes[2, 0].set_ylim(0, np.max(h)*1.1)
    axes[2, 1].set_ylim(0, np.max(u)*1.1)
    axes[2, 0].set_ylabel('K_i = 5\nHeight [m]')
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
    plt.savefig('figures/Problem_3_Ki.png')
    # plt.show()

Kd = 10
if sys.argv[1] == 'pid':
    controller_pid = pid.PIDController(Kp, Ki, 1, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pid)

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
    axes[0, 0].set_ylabel('K_d = 1\nHeight [m]')
    axes[0, 1].set_ylabel('Flow [m³/s]')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)


    controller_pid = pid.PIDController(Kp, Ki, Kd, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pid)

    for i in range(4):
        axes[1, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[1])
        if i < 2:
            axes[1, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])
        else:
            axes[1, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[1])

    axes[1, 0].set_ylim(0, np.max(h)*1.1)
    axes[1, 1].set_ylim(0, np.max(u)*1.1)
    axes[1, 0].set_ylabel(f'K_d = {Kd}\nHeight [m]')
    axes[1, 1].set_ylabel('Flow [m³/s]')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    controller_pid = pid.PIDController(Kp, Ki, 500, setpoint, delta_t, umin, umax)

    t, x, u, d, h = Model.ClosedLoop((t0, tf), state_0, controller_pid)

    for i in range(4):
        axes[2, 0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i], ls=ls[2])
        if i < 2:
            axes[2, 1].plot(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])
        else:
            axes[2, 1].plot(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i], ls=ls[2])

    axes[2, 0].set_ylim(0, np.max(h)*1.1)
    axes[2, 1].set_ylim(0, np.max(u)*1.1)
    axes[2, 0].set_ylabel('K_d = 25\nHeight [m]')
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
    plt.savefig('figures/Problem_3_Kd.png')
    # plt.show()
