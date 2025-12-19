import numpy as np
from src.MPC_economic import MPC_economic_LP
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem
import matplotlib.pyplot as plt

x0, u, d, p , R_s, R_d, delta_t = initialize()
a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p
N_t = 60*20

Model = FourTankSystem(R_s, R_d, p, delta_t)

xs = Model.GetSteadyState(np.concatenate((x0, d)), u)
hs = xs[:2] / (rho*np.array([A1, A2]))
print(hs)

t_array = np.arange(0, N_t, delta_t)
c_array = 200 - 190*np.exp(-0.01*t_array)

H = 40
U = 1000

N = 30
mpc_economic = MPC_economic_LP(c_array, H, U, hs=hs, N=N, Dmin = -np.ones(N*2)*50, Dmax = np.ones(N*2)*50)
# print(mpc_economic.kron(mpc_economic.Wu).shape)


t, x, u, d, h = Model.ClosedLoop(np.array([0, N_t]), xs, mpc_economic)

fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
axes[0].plot(t/60, h[0, :], label='Height of Tank 1', color='dodgerblue')
axes[0].plot(t/60, h[1, :], label='Height of Tank 2', color='tomato')
axes[0].plot(np.array([t[0], t[-1]])/60, [H, H], label='Minimum height', color='gray', ls='--')
axes[0].legend()
axes[0].set_xlabel('Time [min]')
axes[0].set_ylabel('Height [m]')
axes[0].grid(True)
axes[1].plot(t/60, u[0, :], label='Flow of Tank 1', color='dodgerblue')
axes[1].plot(t/60, u[1, :], label='Flow of Tank 2', color='tomato')
axes[1].legend()
axes[1].set_ylabel('Flow [m³/s]')
axes[1].set_xlabel('Time [min]')
axes[1].grid(True)
axes[2].step(t/60, c_array, label='Cost', color='black', ls='--')
axes[2].legend()
axes[2].set_xlabel('Time [min]')
axes[2].set_ylabel('Cost')
axes[2].grid(True)
axes[3].step(t/60, c_array * (u[0, :] + u[1, :]), label='c * u', color='black', ls='--')
axes[3].legend()
axes[3].set_xlabel('Time [min]')
axes[3].set_ylabel('Cost')
axes[3].grid(True)
plt.savefig('figures/Problem13/problem_13_2_plot.png')
plt.show()

print(np.sum(c_array * (u[0, :] + u[1, :])))