import numpy as np
import matplotlib.pyplot as plt

def PlotMPC_sim(**kwargs):
    t = kwargs['t']
    h = kwargs['h']
    u = kwargs['u']
    x = kwargs['x']
    R_bar = kwargs['R_bar']
    xs = kwargs['xs']
    hs = kwargs['hs']
    mpc_controller = kwargs['mpc_controller']
    problem = kwargs['problem']
    file_name = "figures/Problem" + problem + "/" + kwargs.pop('file_name')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t[1:-2]/60, h[0, 1:-2], label='Height of Tank 1', color='dodgerblue')
    axes[0].plot(t[1:-2]/60, h[1, 1:-2], label='Height of Tank 2', color='tomato')
    axes[0].plot(t[1:-2]/60, h[2, 1:-2], label='Height of Tank 3', color='limegreen')
    axes[0].plot(t[1:-2]/60, h[3, 1:-2], label='Height of Tank 4', color='orange')

    setpoint_1 = R_bar[0, 0]
    setpoint_2 = R_bar[0, 1]
    axes[0].plot(t[1:-2]/60, t[1:-2]*0 + setpoint_1, label='Setpoint for Tank 1', color='dodgerblue', ls='--')
    axes[0].plot(t[1:-2]/60, t[1:-2]*0 + setpoint_2, label='Setpoint for Tank 2', color='tomato', ls='--')
    axes[0].legend()
    axes[0].set_xlabel('Time [min]')
    axes[0].set_ylabel('Height [m]')
    axes[0].grid(True)
    axes[1].plot(t[1:-2]/60, u[0, 1:-2], label='Flow of Tank 1', color='dodgerblue')
    axes[1].plot(t[1:-2]/60, u[1, 1:-2], label='Flow of Tank 2', color='tomato')
    axes[1].legend()
    axes[1].set_xlabel('Time [min]')
    axes[1].set_ylabel('Flow [m³/s]')
    axes[1].grid(True)
    fig.savefig(file_name + ".png")
    plt.close()

    predicted_x_mpc = np.array(mpc_controller.predicted_x_mpc)
    predicted_y_mpc = np.array(mpc_controller.predicted_y_mpc)

    predicted_Px = np.array(mpc_controller.predicted_Px)
    predicted_Py = np.array(mpc_controller.predicted_Py)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t[1:-2]/60, predicted_x_mpc[1:-2, 0] + xs[0], label='Predicted Tank 1', color='dodgerblue', ls='-.' )
    axes[0].plot(t[1:-2]/60, predicted_x_mpc[1:-2, 1] + xs[1], label='Predicted Tank 2', color='tomato', ls='-.' )

    axes[0].fill_between(t[1:-2]/60, predicted_x_mpc[1:-2, 0] + xs[0] - 2*np.sqrt(predicted_Px[1:-2, 0, 0]), predicted_x_mpc[1:-2, 0] + xs[0] + 2*np.sqrt(predicted_Px[1:-2, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[0].fill_between(t[1:-2]/60, predicted_x_mpc[1:-2, 1] + xs[1] - 2*np.sqrt(predicted_Px[1:-2, 1, 1]), predicted_x_mpc[1:-2, 1] + xs[1] + 2*np.sqrt(predicted_Px[1:-2, 1, 1]), color='tomato', alpha=0.2)
    
    axes[0].plot(t[1:-2]/60, x[0, 1:-2], label='Actual Tank 1', color='dodgerblue')
    axes[0].plot(t[1:-2]/60, x[1, 1:-2], label='Actual Tank 2', color='tomato')
    
    axes[0].legend()
    axes[0].set_ylabel('mass [kg]')
    axes[0].grid(True)
    axes[0].set_title('System states')
    
    axes[1].plot(t[1:-2]/60, predicted_y_mpc[1:-2, 0] + hs[0], label='Predicted Output Tank 1', color='dodgerblue', ls='-.' )
    axes[1].plot(t[1:-2]/60, predicted_y_mpc[1:-2, 1] + hs[1], label='Predicted Output Tank 2', color='tomato', ls='-.' )
    

    axes[1].fill_between(t[1:-2]/60, predicted_y_mpc[1:-2, 0] + hs[0] - 2*np.sqrt(predicted_Py[1:-2, 0, 0]), predicted_y_mpc[1:-2, 0] + hs[0] + 2*np.sqrt(predicted_Py[1:-2, 0, 0]), color='dodgerblue', alpha=0.2)
    axes[1].fill_between(t[1:-2]/60, predicted_y_mpc[1:-2, 1] + hs[1] - 2*np.sqrt(predicted_Py[1:-2, 1, 1]), predicted_y_mpc[1:-2, 1] + hs[1] + 2*np.sqrt(predicted_Py[1:-2, 1, 1]), color='tomato', alpha=0.2)
    
    axes[1].plot(t[1:-2]/60, h[0, 1:-2], label='Actual Tank 1', color='dodgerblue')
    axes[1].plot(t[1:-2]/60, h[1, 1:-2], label='Actual Tank 2', color='tomato')
    
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Height of the tanks')
    axes[1].set_xlabel('Time [min]')
    axes[1].set_ylabel('Height [m]')
    
    fig.savefig(file_name + "_kalman.png")
    plt.close()