import numpy as np
import matplotlib.pyplot as plt

colors = ['dodgerblue', 'tomato', 'limegreen', 'orange']

def PlotSimulation(ax, t, x, u, d, h, legend = False):
    for i in range(4):
        ax[0].plot(t/60, h[i, :], label=f'Height of Tank {i+1}', color=colors[i])
        if i < 2:
            ax[1].step(t/60, u[i, :], label=f'Flow of Tank {i+1}', color=colors[i])
        else:
            ax[1].step(t/60, d[i-2, :], label=f'Flow of Tank {i+1}', color=colors[i])
    ax[0].set_ylim(0, np.max(h)*1.1)
    ax[1].set_ylim(0, max(np.max(u), np.max(d))*1.1)
    ax[0].set_ylabel('Height [m]')
    ax[1].set_ylabel('Flow [m³/s]')
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[1].grid(True, linestyle='--', alpha=0.5)
    if legend:
        ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
            ncol=2, fancybox=True, shadow=True)
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
            ncol=2, fancybox=True, shadow=True)

    return
