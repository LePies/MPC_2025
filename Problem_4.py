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
F4 = 10
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

df = pd.DataFrame(columns=['noise_level', 'step', 'k_1', 'k_2', 'k_3', 'k_4', 'tau_1', 'tau_2', 'tau_3', 'tau_4'])

df_i = 0

for noise_level in noise_leves:
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

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

    for j, step in enumerate(steps):
        u_array[0, Nt//4:] = F1*(1 + step)
        u_array[1, Nt//4:] = F2*(1 + step)

        state_0 = np.concatenate([xs, d])

        t, x, u_out, d_out, h = Model.OpenLoop((t0, Nt*delta_t), state_0, u_array)

        h_norm = (h - hs[:, None]) / hs[:, None]
        u_norm = (u_out - u[:, None]) / u[:, None]
        ss_new = np.mean(h_norm[:, -Nt//10:], axis = 1)

        tau_idx = np.where(h_norm / ss_new[:,None] > 0.63)[1][:]
        tau = t[tau_idx]

        df.loc[df_i] = [noise_level, step, ss_new[0], ss_new[1], ss_new[2], ss_new[3], tau[0], tau[1], tau[2], tau[3]]
        df_i += 1

        if np.max(ss_new) > ss_max:
            ss_max = np.max(ss_new)

        print(f"Step Response:\n\tstep: {step}\n\tss: {ss_new}\n\ttaus: {tau}")

        for i in range(4):  
            axes[j, 0].plot(t/60, h_norm[i, :], label=f'Normalized Height of Tank {i+1}', color=colors[i], ls=ls[0])
            if i < 2:
                axes[j, 1].plot(t/60, u_norm[i, :], label=f'Normalized Flow of Tank {i+1}', color=colors[i], ls=ls[0])

    for j in range(3):
        axes[j, 0].set_xlabel('Time [m]')
        axes[j, 1].set_xlabel('Time [m]')
        axes[j, 0].set_ylabel('Normalized Height [m]')
        axes[j, 1].set_ylabel('Normalized Flow [m³/s]')
        axes[j, 0].grid(True, linestyle='--', alpha=0.5)
        axes[j, 1].grid(True, linestyle='--', alpha=0.5)
        axes[j, 0].set_ylim(0, ss_max*1.1)
        axes[j, 1].set_ylim(0, step*1.1)

    axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
        ncol=2, fancybox=True, shadow=True)
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
        ncol=2, fancybox=True, shadow=True)
    plt.savefig(f'figures/Problem4/Problem_4_noise_{noise_level}.png')
    plt.close()

    print(f'Noise level: {noise_level}\nPlot saved to Problem_4_noise_{noise_level}.png')
    print("-"*10)

df.to_csv('figures/Problem4/Problem_4_df.csv', index=False)
