import numpy as np
from FourTankSystem import FourTankSystem
import parameters_tank as para
import matplotlib.pyplot as plt
import compute_steady_state as css

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

# Computing steady state of model 1
xs = css.compute_steady_state(Model_Deterministic.StateEquation, x0, u)[:4]

t, x, _, _, h = Model_Deterministic.OpenLoop((t0, tf), xs, u_array, d_array)

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
axes[0].plot(t, h[0, :], label='$h_1$')
axes[0].plot(t, h[1, :], label='$h_2$')
axes[0].plot(t, h[2, :], label='$h_3$')
axes[0].plot(t, h[3, :], label='$h_4$')
axes[0].set_title('Heights of liquids in tanks')
axes[0].legend()

axes[1].plot(t, u_array[0, :], label='$F_1$ ($u_1$)')
axes[1].plot(t, u_array[1, :], label='$F_2$ ($u_2$)')
axes[1].plot(t, d_array[0, :], label='$F_3$ ($d_1$)')
axes[1].plot(t, d_array[1, :], label='$F_4$ ($d_2$)')
axes[1].set_title('Flow rate of tanks')
axes[1].legend()
fig.suptitle('Open loop simulation of modified four tank system Model 1 (deterministic)', fontsize=16)

plt.tight_layout()
plt.show()

Model_Stochastic = FourTankSystem(R_s, R_d, p, delta_t)
noise = np.random.normal(0, np.sqrt(20), size=(2, tf))

# Computing steady state of model 2
xs = css.compute_steady_state(Model_Stochastic.StateEquation, x0, u)
t, x, _, d, h = Model_Stochastic.OpenLoop((t0, tf), xs, u_array, d_array + noise)

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

axes[0].plot(t, h[0, :], label='$h_1$')
axes[0].plot(t, h[1, :], label='$h_2$')
axes[0].plot(t, h[2, :], label='$h_3$')
axes[0].plot(t, h[3, :], label='$h_4$')
axes[0].set_title('Heights of liquids in tanks')
axes[0].legend()

axes[1].plot(t, u_array[0, :], label='$F_1$ ($u_1$)')
axes[1].plot(t, u_array[1, :], label='$F_2$ ($u_2$)')
axes[1].plot(t, d[0, :], label='$F_3$ ($d_1$)')
axes[1].plot(t, d[1, :], label='$F_4$ ($d_2$)')
axes[1].set_title('Flow rate of tanks')
axes[1].legend()
fig.suptitle('Open loop simulation of modified four tank system Model 2 (stochastic disturbance)', fontsize=16)

plt.tight_layout()
plt.show()

# Extending state for model 3
x_extended = np.concatenate([x0, d_array[:, 0]])

# Compute steady state for extended state of model 3
xs_extended = css.compute_steady_state(Model_Stochastic.StateEquation, x_extended, u)

# Computing openloop for model 3
t, x, _, d, h = Model_Stochastic.OpenLoop((t0, tf), xs_extended, u_array)

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
axes[0].plot(t, h[0, :], label='$h_1$')
axes[0].plot(t, h[1, :], label='$h_2$')
axes[0].plot(t, h[2, :], label='$h_3$')
axes[0].plot(t, h[3, :], label='$h_4$')
axes[0].set_title('Heights of liquids in tanks')
axes[0].legend()

axes[1].plot(t, u_array[0, :], label='$F_1$ ($u_1$)')
axes[1].plot(t, u_array[1, :], label='$F_2$ ($u_2$)')
axes[1].plot(t, d[0, :], label='$F_3$ ($d_1$)')
axes[1].plot(t, d[1, :], label='$F_4$ ($d_2$)')
axes[1].set_title('Flow rate of tanks')
axes[1].legend()

fig.suptitle('Open loop simulation of modified four tank system Model 3 (SDE)', fontsize=16)

plt.tight_layout()
plt.show()