import system as sys
import numpy as np
import parameters_tank as para 
import matplotlib.pyplot as plt
import compute_steady_state as css
import simulations as sim

t0 = 0
tf = 20*90
m10 = 0
m20 = 0
m30 = 0
m40 = 0
F1 = 250
F2 = 325
F3 = 0
F4 = 0 
x0 = np.array([m10,m20,m30,m40])
u = np.array([F1,F2])
d = np.array([F3,F4])
p = para.parameters()
a1,a2,a3,a4,A1,A2,A3,A4,gamma1,gamma2,g,rho = p

# Computing steady state
xs = css.compute_steady_state(sys.f_modified, x0, u, d, p)
print("Steady state (m):", xs)
print("Steady state (h):", xs/(rho*np.array([A1,A2,A3,A4])))

# Step response in open loop
U = np.zeros((2, tf))
U[0, :] = F1
U[1, :] = F2

# Define distrubances
D = np.zeros((2, tf))
D[0, :] = F1
D[1, :] = F2    

# Simulate deterministic open-loop response
step1 = 200
U[:, step1:] = np.array([F1 * 1.05, F2  * 1.05])[:, None]           
D[:, step1:] = np.array([F1 * 0.95, F2  * 0.95])[:, None]
step2 = 600
U[:, step2:] = np.array([F1 * 1.10, F2  * 1.10])[:, None]
D[:, step2:] = np.array([F1 * 0.90, F2  * 0.90])[:, None]
step3 = 1000    

T, X = sim.openloop(sys.f_modified, t0, tf, x0, U, D, p)

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
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Flow [m³/s]')
axes[1].legend()

plt.show()




