import numpy as np
import params.parameters_tank as para
import matplotlib.pyplot as plt
from src.FourTankSystem import FourTankSystem
import pandas as pd


t0 = 0
tf = 30*60
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
a1,a2,a3,a4,A1,A2,A3,A4,gamma1,gamma2,g,rho = p

delta_t = 1
Nt = tf//delta_t

noise_leves = [0, 0.01, 0.1, 0.5]
colors = ['dodgerblue', 'tomato', 'limegreen', 'orange']
ls = ['-', '-', '-']

df = pd.DataFrame(columns=['Manipulated input', 'noise_level', 'step', 'K_11', 'K_12', 'K_21', 'K_22', 'tau_1', 'tau_2', 'tau_3', 'tau_4', 'delta_y_1', 'delta_y_2', 'delta_u_1', 'delta_u_2'])

df_i = 0

fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True)

for u_idx in range(2):
    for k, noise_level in enumerate(noise_leves):
        Rvv = np.eye(4)*noise_level

        Model = FourTankSystem(Rvv, Rvv, p, delta_t, F3 = F3, F4 = F4, sigma_f3 = noise_level, sigma_f4 = noise_level)

        xs = Model.GetSteadyState(x0, u, d)


        # Computing steady state
        xs = Model.GetSteadyState(x0, u, d)

        hs = xs/(rho*np.array([A1,A2,A3,A4]))

        u_array = np.zeros((2, Nt))
        u_array[0, :] = F1
        u_array[1, :] = F2


        steps = [0.1, 0.25, 0.5]

        ss_max = 0

        for j, step in enumerate(steps):
            if u_idx == 0:
                u_array[0, Nt//4:] = F1*(1 + step)
            else:
                u_array[1, Nt//4:] = F2*(1 + step)
            


            state_0 = np.concatenate([xs, d])

            t, x, u_out, d_out, h = Model.OpenLoop((t0, Nt*delta_t), state_0, u_array)

            y = Model.StateOutput(h)

            t_mod = t[Nt//4]
            if u_idx == 0:
                h_norm = (h - hs[:, None]) / (step*F1)
            else:
                h_norm = (h - hs[:, None]) / (step*F2)
            u_norm = (u_out - u[:, None]) / u_out
            ss_new = np.mean(h_norm[:, -Nt//10:], axis = 1)

            tau_idxs = np.where(h_norm / ss_new[:,None] > 0.63)
            tau_idx = np.array([tau_idxs[1][tau_idxs[0] == i][0] for i in range(4)])
            tau = t[tau_idx] - t_mod
            tau[tau < 0] = 0

            y_0 = np.mean(y[:, :Nt//10], axis = 1)
            y_tk = np.mean(y[:, -Nt//10:], axis = 1)
            delta_y = y_tk - y_0
            if u_idx == 0:
                k11 = ss_new[0]
                k12 = 0
                k21 = ss_new[1]
                k22 = 0
            else:
                k11 = 0
                k12 = ss_new[0]
                k21 = 0
                k22 = ss_new[1]

            df.loc[df_i] = [u_idx, noise_level, step, k11, k12, k21, k22, tau[0], tau[1], tau[2], tau[3], delta_y[0], delta_y[1], step*F1, step*F2]
            df_i += 1

            if np.max(ss_new) > ss_max:
                ss_max = np.max(ss_new)

            print(f"Step Response:\n\tstep: {step}\n\tss: {ss_new}\n\ttaus: {tau}")

            if u_idx == 0:
                for i in range(4):
                    axes[k, j].grid(True, linestyle='--', alpha=0.5)
                    axes[k, j].plot(t/60, h_norm[i, :], label=f'Normalized Height of Tank {i+1}', color=colors[i], ls=ls[0])
                    # if i < 2:
                    # axes[j, -1].plot(t/60, u_norm[i, :], label=f'Normalized Flow of Tank {i+1}', color=colors[i], ls=ls[0])
                    axes[k, j].vlines((t_mod + tau[i])/60, 0, h_norm[i, tau_idx[i]], color=colors[i], ls='--', alpha=0.5)
                    axes[k, j].hlines(h_norm[i, tau_idx[i]], 0, (t_mod + tau[i])/60, color=colors[i], ls='--', alpha=0.5)

                    # axes[j, -1].vlines(tau[i]/60, 0, u_norm[0, tau_idx[i]], color=colors[i], ls='--', alpha=0.5)
                    # axes[j, -1].hlines(u_norm[0, tau_idx[i]], 0, (t_mod + tau[i])/60, color=colors[i], ls='--', alpha=0.5)

        axes[0, 2].legend(loc='upper center', bbox_to_anchor=(-0.75, 1.30),
            ncol=2, fancybox=True, shadow=True)

        print(f'Noise level: {noise_level}\nPlot saved to Problem_4_noise_{noise_level}.png')
        print("-"*10)

# plt.tight_layout()
axes[3, 0].set_xlabel('Time [m]\nStep = 0.1')
axes[3, 1].set_xlabel('Time [m]\nStep = 0.25')
axes[3, 2].set_xlabel('Time [m]\nStep = 0.5')
axes[0, 0].set_ylabel('Noiseless\nNormalized Height [m]')
axes[1, 0].set_ylabel('Noise Level = 0.01\nNormalized Height [m]')
axes[2, 0].set_ylabel('Noise Level = 0.1\nNormalized Height [m]')
axes[3, 0].set_ylabel('Noise Level = 0.5\nNormalized Height [m]')

plt.savefig(f'figures/Problem4/Problem_4_all.png')
plt.close()

df.to_csv('figures/Problem4/Problem_4_df.csv', index=False)
