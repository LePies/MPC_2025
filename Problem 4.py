import trashbin.system as sys
import numpy as np
import params.parameters_tank as para
import matplotlib.pyplot as plt
import trashbin.compute_steady_state as css
import trashbin.simulations as sim
import src.PIDcontrolor as pid

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
a1,a2,a3,a4,A1,A2,A3,A4,gamma1,gamma2,g,rho = p

# Computing steady state
xs = css.compute_steady_state(sys.f_modified, x0, u, d, p)
print("Steady state (m):", xs)
print("Steady state (h):", xs/(rho*np.array([A1,A2,A3,A4])))


# Define distrubances
Kp = 4.0
Ki = 0.1
Kd = 0.5
r = np.array([10.0, 10.0])
u0 = np.array([F1, F2])
Rvv = np.eye(4)*0.01
umin = 1
umax = 10000
pid = pid.PIDController(Kp, Ki, Kd, r, 1, umin, umax)

D = np.zeros((2, tf))
D[0, :] = np.random.normal(F3, 1, size=tf)
D[1, :] = np.random.normal(F4, 1, size=tf)
delta_t = 1
Qww = np.eye(2)*1000

T, X, U, Y, Z = sim.closed_loop(sys.f_SDE, sys.g_sensor, sys.h_sensor, t0, tf, x0, r, u0, D, delta_t,p, Qww, Rvv, pid)

# Plotting example
fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
axes[0].plot(T.ravel(), X[0, :]/(rho*A1), label='Tank 1')
axes[0].plot(T.ravel(), X[1, :]/(rho*A2), label='Tank 2')
axes[0].plot(T.ravel(), X[2, :]/(rho*A3), label='Tank 3')
axes[0].plot(T.ravel(), X[3, :]/(rho*A4), label='Tank 4')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Height [m]')
axes[0].legend()
axes[0].set_title('Step Response of Tanks')
axes[1].step(T.ravel(), U[0, :], label='F1')
axes[1].step(T.ravel(), U[1, :], label='F2')
axes[1].step(T.ravel(), D[0, :], label='F3')
axes[1].step(T.ravel(), D[1, :], label='F4')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Flow [m³/s]')
axes[1].legend()

plt.show()

