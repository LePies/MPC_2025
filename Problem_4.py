import numpy as np
import params.parameters_tank as para
import matplotlib.pyplot as plt
from src.FourTankSystem import FourTankSystem
import pandas as pd
from scipy.optimize import curve_fit


def format_2nd_order_tf(K, tau1, tau2, input_name='d', output_name='y'):
    """
    Format a second-order overdamped transfer function as a string.
    
    Parameters:
    -----------
    K : float
        Steady-state gain
    tau1 : float
        First time constant
    tau2 : float
        Second time constant
    input_name : str
        Name of input (e.g., 'd1', 'u1')
    output_name : str
        Name of output (e.g., 'y1', 'y2')
    
    Returns:
    --------
    str : Formatted transfer function string
    """
    # Factored form: G(s) = K / ((tau1*s + 1)(tau2*s + 1))
    factored = f"G_{output_name}{input_name}(s) = {K:.6f} / (({tau1:.4f}s + 1)({tau2:.4f}s + 1))"
    
    # Expanded form: G(s) = K / (tau1*tau2*s^2 + (tau1+tau2)*s + 1)
    a = tau1 * tau2
    b = tau1 + tau2
    expanded = f"G_{output_name}{input_name}(s) = {K:.6f} / ({a:.6f}s² + {b:.4f}s + 1)"
    
    return f"{factored}\n    or\n{expanded}"


def second_order_step_response(t, K, tau1, tau2):
    """
    Step response of a 2nd order overdamped system.
    
    G(s) = K / ((tau1*s + 1)(tau2*s + 1))
    
    Step response: y(t) = K * (1 - (tau1*exp(-t/tau1) - tau2*exp(-t/tau2))/(tau1-tau2))
    
    Parameters:
    -----------
    t : array-like
        Time array
    K : float
        Steady-state gain
    tau1 : float
        First time constant (must be != tau2)
    tau2 : float
        Second time constant (must be != tau1)
    
    Returns:
    --------
    y : array
        Step response values
    """
    t = np.asarray(t)
    # Avoid division by zero if tau1 == tau2
    if abs(tau1 - tau2) < 1e-6:
        # If tau1 == tau2, use the limit: y(t) = K * (1 - (1 + t/tau1)*exp(-t/tau1))
        return K * (1 - (1 + t/tau1) * np.exp(-t/tau1))
    else:
        return K * (1 - (tau1 * np.exp(-t/tau1) - tau2 * np.exp(-t/tau2)) / (tau1 - tau2))


def fit_2nd_order_step_response(t, y_data, K_guess=None, tau1_guess=None, tau2_guess=None):
    """
    Fit a 2nd order overdamped step response to data using curve fitting.
    
    Parameters:
    -----------
    t : array-like
        Time array (should start from step time, i.e., t[0] = 0 at step)
    y_data : array-like
        Step response data
    K_guess : float, optional
        Initial guess for gain (default: max of y_data)
    tau1_guess : float, optional
        Initial guess for first time constant
    tau2_guess : float, optional
        Initial guess for second time constant
    
    Returns:
    --------
    K : float
        Fitted gain
    tau1 : float
        Fitted first time constant
    tau2 : float
        Fitted second time constant
    """
    t = np.asarray(t)
    y_data = np.asarray(y_data)
    
    # Remove any NaN or inf values
    valid = np.isfinite(y_data) & np.isfinite(t) & (t >= 0)
    t_clean = t[valid]
    y_clean = y_data[valid]
    
    if len(t_clean) < 3:
        raise ValueError("Not enough valid data points for fitting")
    
    # Initial guesses
    if K_guess is None:
        K_guess = np.max(y_clean) if len(y_clean) > 0 else 1.0
    
    if tau1_guess is None:
        # Rough estimate: time to reach 63% of steady state
        ss_val = np.mean(y_clean[-len(y_clean)//10:]) if len(y_clean) > 10 else K_guess
        if ss_val > 0:
            idx_63 = np.argmax(y_clean > 0.63 * ss_val)
            tau1_guess = t_clean[idx_63] if idx_63 > 0 else t_clean[-1] / 3
        else:
            tau1_guess = t_clean[-1] / 3
    
    if tau2_guess is None:
        # Rough estimate: time to reach 86.5% of steady state
        ss_val = np.mean(y_clean[-len(y_clean)//10:]) if len(y_clean) > 10 else K_guess
        if ss_val > 0:
            idx_865 = np.argmax(y_clean > 0.865 * ss_val)
            tau2_guess = t_clean[idx_865] if idx_865 > 0 else t_clean[-1] / 2
        else:
            tau2_guess = t_clean[-1] / 2
    
    # Ensure tau1 < tau2 (for overdamped system)
    if tau1_guess > tau2_guess:
        tau1_guess, tau2_guess = tau2_guess, tau1_guess
    
    # Ensure reasonable bounds
    t_max = np.max(t_clean)
    bounds = ([0, 0.1, 0.1], [K_guess * 2, t_max * 2, t_max * 2])
    
    try:
        # Fit the curve
        popt, _ = curve_fit(
            second_order_step_response,
            t_clean,
            y_clean,
            p0=[K_guess, tau1_guess, tau2_guess],
            bounds=bounds,
            maxfev=5000
        )
        
        K_fit, tau1_fit, tau2_fit = popt
        
        # Ensure tau1 < tau2
        if tau1_fit > tau2_fit:
            tau1_fit, tau2_fit = tau2_fit, tau1_fit
        
        return K_fit, tau1_fit, tau2_fit
    
    except Exception as e:
        print(f"Warning: Curve fitting failed: {e}")
        print(f"  Using initial guesses: K={K_guess:.4f}, tau1={tau1_guess:.2f}, tau2={tau2_guess:.2f}")
        return K_guess, tau1_guess, tau2_guess


t0 = 0
tf = 30*60
m10 = 0
m20 = 0
m30 = 0
m40 = 0
F1 = 250
F2 = 325
F3 = lambda t: 100
F4 = lambda t: 120
x0 = np.array([m10,m20,m30,m40])
u = np.array([F1,F2])
d = np.array([F3(0),F4(0)])
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
                k22 = 0
                k21 = ss_new[1]
                k12 = 0
            else:
                k11 = 0
                k22 = ss_new[1]
                k21 = 0
                k12 = ss_new[0]

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


df_d = pd.DataFrame(columns=['Manipulated input', 'noise_level', 'step', 'K_11', 'K_12', 'K_21', 'K_22', 'tau_1', 'tau_2', 'tau_3', 'tau_4', 'delta_y_1', 'delta_y_2', 'delta_u_1', 'delta_u_2'])
df_d_i = 0
for d_idx in range(2):
    Rvv = np.eye(4)*0

    xs = Model.GetSteadyState(x0, u, d)
    hs = xs/(rho*np.array([A1,A2,A3,A4]))
    u_array = np.zeros((2, Nt))
    u_array[0, :] = F1
    u_array[1, :] = F2

    steps = [0.1, 0.25, 0.5]
    for j, step in enumerate(steps):
        if d_idx == 0:
            def F3_step(t):
                if t < Nt//4*delta_t:
                    return F3(t)
                else:
                    return F3(t)*(1 + step)
            F4_step = F4
        else:
            def F4_step(t):
                if t < Nt//4*delta_t:
                    return F4(t)
                else:
                    return F4(t)*(1 + step)
            F3_step = F3
        
        Model = FourTankSystem(Rvv, Rvv, p, delta_t, F3 = F3_step, F4 = F4_step, sigma_f3 = noise_level, sigma_f4 = noise_level)
        xs = Model.GetSteadyState(x0, u, d)
        hs = xs/(rho*np.array([A1,A2,A3,A4]))
        u_array = np.zeros((2, Nt))
        u_array[0, :] = F1
        u_array[1, :] = F2
        t, x, u_out, d_out, h = Model.OpenLoop((t0, Nt*delta_t), state_0, u_array)
        y = Model.StateOutput(h)
        t_mod = t[Nt//4]

        # For normalized disturbance step size (for d1 and d2)
        if d_idx == 0:
            disturbance_mag = step * F3(0)
        else:
            disturbance_mag = step * F4(0)

        # For each tank, normalize to the disturbance step
        h_norm = (h - hs[:, None]) / disturbance_mag

        # Steady-state gain as the mean of last 10% (per tank)
        ss_new = np.mean(h_norm[:, -Nt//10:], axis=1)

        # ------ Curve Fitting for 2nd Order Disturbance Systems ------

        # Fit 2nd order transfer function for d1->y1 and d2->y2 using curve fitting
        if d_idx == 0:
            # d1 -> y1
            trace_idx = 0
            h_trace = h_norm[trace_idx, :]
            
            # Extract time relative to step (t_mod is when step occurs)
            t_rel = t - t_mod
            # Only use data after the step
            mask = t_rel >= 0
            t_fit = t_rel[mask]
            y_fit = h_trace[mask]
            
            # Fit 2nd order step response
            K_fit, tau1_fit, tau2_fit = fit_2nd_order_step_response(
                t_fit, y_fit,
                K_guess=ss_new[trace_idx],
                tau1_guess=None,  # Will be estimated from data
                tau2_guess=None   # Will be estimated from data
            )
            
            # Save the fitted time constants
            tau_1_2o = tau1_fit
            tau_2_2o = tau2_fit

            # Save result: second order for d1->y1
            k11 = K_fit
            k12 = 0
            k21 = 0
            k22 = 0
            tau_vals = [tau_1_2o, tau_2_2o, np.nan, np.nan]
            
            print(f"Second Order Fitted Step Response (curve fitting) - d1->y1:\n\tstep: {step}\n\tfitted K: {K_fit:.6f}\n\tfitted taus: tau1={tau_1_2o:.2f}, tau2={tau_2_2o:.2f}\n\tss from data: {ss_new[trace_idx]:.6f}")

        else:
            # d2 -> y2
            trace_idx = 1
            h_trace = h_norm[trace_idx, :]
            
            # Extract time relative to step (t_mod is when step occurs)
            t_rel = t - t_mod
            # Only use data after the step
            mask = t_rel >= 0
            t_fit = t_rel[mask]
            y_fit = h_trace[mask]
            
            # Fit 2nd order step response
            K_fit, tau1_fit, tau2_fit = fit_2nd_order_step_response(
                t_fit, y_fit,
                K_guess=ss_new[trace_idx],
                tau1_guess=None,  # Will be estimated from data
                tau2_guess=None   # Will be estimated from data
            )
            
            # Save the fitted time constants
            tau_1_2o = tau1_fit
            tau_2_2o = tau2_fit

            # Save result: second order for d2->y2
            k11 = 0
            k12 = 0
            k21 = 0
            k22 = K_fit
            tau_vals = [np.nan, np.nan, tau_1_2o, tau_2_2o]
            
            print(f"Second Order Fitted Step Response (curve fitting) - d2->y2:\n\tstep: {step}\n\tfitted K: {K_fit:.6f}\n\tfitted taus: tau1={tau_1_2o:.2f}, tau2={tau_2_2o:.2f}\n\tss from data: {ss_new[trace_idx]:.6f}")
        
        # Print transfer function
        if d_idx == 0:
            # d1 -> y1
            print(f"\nTransfer Function d1 -> y1:")
            print(format_2nd_order_tf(k11, tau_1_2o, tau_2_2o, input_name='d1', output_name='y1'))
        else:
            # d2 -> y2
            print(f"\nTransfer Function d2 -> y2:")
            print(format_2nd_order_tf(k22, tau_1_2o, tau_2_2o, input_name='d2', output_name='y2'))
        print("-"*50)

        y_0 = np.mean(y[:, :Nt//10], axis=1)
        y_tk = np.mean(y[:, -Nt//10:], axis=1)
        delta_y = y_tk - y_0

        # Store values: tau_1,tau_2,tau_3,tau_4 in csv (tau_1, tau_2 = d1->y1, tau_3, tau_4 = d2->y2)
        df_d.loc[df_d_i] = [
            d_idx, 0, step, k11, k12, k21, k22,
            tau_vals[0], tau_vals[1], tau_vals[2], tau_vals[3],
            delta_y[0], delta_y[1], step*F3(0), step*F4(0)
        ]
        df_d_i += 1

        # plt.plot(t/60, h_norm[0, :], label=f'Normalized Height of Tank 1', color=colors[0], ls=ls[0])
        # # plt.plot(t/60, h_norm[1, :], label=f'Normalized Height of Tank 2', color=colors[1], ls=ls[0])
        # plt.plot([t[Nt//4]/60, t[Nt//4]/60], [0, np.max(h_norm[0, :])], color='black', ls='--')
        # plt.legend()
        # plt.show()

df.to_csv('Results/Problem4/Problem_4_df.csv', index=False)
df_d.to_csv('Results/Problem4/Problem_4_df_d.csv', index=False)
