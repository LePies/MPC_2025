import numpy as np
from src.MPC_economic import MPC_economic
from params.initialize import initialize
from src.FourTankSystem import FourTankSystem
import matplotlib.pyplot as plt

x0, u, d, p , R_s, R_d, delta_t = initialize()
a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p
N_t = 60*20

Model = FourTankSystem(R_s, R_d, p, delta_t)

xs = Model.GetSteadyState(np.concatenate((x0, d)), u)
hs = xs[:2] / (rho*np.array([A1, A2]))


t_array = np.arange(0, 60*20, delta_t)
c_array = 200 - 190*np.exp(-0.01*t_array)

H = np.array([1, 1])
U = np.array([1, 1])

mpc_economic = MPC_economic(c_array, H, U, hs)

t, x, u, d, h = Model.ClosedLoop(np.array([0, N_t]), xs, mpc_economic)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t/60, h[0, :], label='Height of Tank 1', color='dodgerblue')
axes[0].plot(t/60, h[1, :], label='Height of Tank 2', color='tomato')
axes[0].legend()
axes[0].set_xlabel('Time [min]')
axes[0].set_ylabel('Height [m]')
axes[0].grid(True)
axes[1].plot(t/60, u[0, :], label='Flow of Tank 1', color='dodgerblue')
axes[1].plot(t/60, u[1, :], label='Flow of Tank 2', color='tomato')
axes[1].legend()
plt.show()