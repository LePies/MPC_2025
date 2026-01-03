import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\4_semester\Model predictive control\MPC_2025\old_casadi\CasADi_framework_example\Problem12_4_NMPC_Data.xlsx"

# Read sheets
targets = pd.read_excel(file, sheet_name="Targets")
inputs  = pd.read_excel(file, sheet_name="Inputs")
cont    = pd.read_excel(file, sheet_name="Continuous")

# --- Continuous signals (smooth lines) ---
t_cont = cont["time_min"].to_numpy()
# heights are z1,z2 in Continuous; tanks 3-4 might be x3/x4 or not present depending on model.
# From your export: Continuous has z1,z2 and x1..x4. If tanks 3-4 heights aren't in z, you likely want x3/x4 transformed.
# But in your screenshot you plot 4 "heights". If your system has 4 height outputs, export them as z1..z4.
# For now, we plot z1,z2 and (optionally) x3,x4 as proxies if that's what you intend.
z1 = cont["z1"].to_numpy()
z2 = cont["z2"].to_numpy()

# If you truly have height outputs for tanks 3 and 4 and exported them, replace these with cont["z3"], cont["z4"].
# Otherwise (common in four-tank), x3,x4 are masses; remove these two lines if not meaningful.
z3 = cont["x3_kg"].to_numpy() * 0 + np.nan  # placeholder: remove if you don't have z3
z4 = cont["x4_kg"].to_numpy() * 0 + np.nan  # placeholder: remove if you don't have z4

# --- Targets (stairs) ---
t_samp = targets["time_min"].to_numpy()
zbar1  = targets["zbar1"].to_numpy()
zbar2  = targets["zbar2"].to_numpy()

# Make steps look like MATLAB stairs that hold the last value
t_step = np.r_[t_samp, t_samp[-1] + (t_samp[-1] - t_samp[-2] if len(t_samp) > 1 else 0)]
zbar1_step = np.r_[zbar1, zbar1[-1]]
zbar2_step = np.r_[zbar2, zbar2[-1]]

# --- Inputs (stairs) ---
t_u = inputs["time_min"].to_numpy()
u1  = np.concatenate(([250],inputs["u1"][:-1].to_numpy()))
u2  = np.concatenate(([325],inputs["u2"][:-1].to_numpy()))

t_u_step = np.r_[t_u, t_u[-1] + (t_u[-1] - t_u[-2] if len(t_u) > 1 else 0)]
u1_step  = np.r_[u1, u1[-1]]
u2_step  = np.r_[u2, u2[-1]]

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# ===== Top: Heights =====
ax1.plot(t_cont, z1, color="dodgerblue", linewidth=2.5, label="Height of Tank 1")
ax1.plot(t_cont, z2, color="tomato", linewidth=2.5, label="Height of Tank 2")

# If you have z3/z4 exported, plot them here:
# ax1.plot(t_cont, cont["z3"].to_numpy(), linewidth=2.5, label="Height of Tank 3")
# ax1.plot(t_cont, cont["z4"].to_numpy(), linewidth=2.5, label="Height of Tank 4")

# Targets (setpoints)
ax1.step(t_step, zbar1_step, color="dodgerblue", where="post", linestyle="--", linewidth=2.0, label="Setpoint for Tank 1")
ax1.step(t_step, zbar2_step, color="tomato", where="post", linestyle="--", linewidth=2.0, label="Setpoint for Tank 2")

# Optional shaded bands around targets (edit band size)
band = 5.0  # in same units as z (cm or m depending on your export)
ax1.fill_between(t_step, zbar1_step - band, zbar1_step + band, step="post", alpha=0.15)
ax1.fill_between(t_step, zbar2_step - band, zbar2_step + band, step="post", alpha=0.15)

ax1.set_ylabel("Height [m]")  # change to [cm] if that's your unit
ax1.grid(True)
ax1.legend(loc="lower right", ncol=2)

# ===== Bottom: Inputs =====
ax2.step(t_u_step, u1_step, color="dodgerblue", where="post", linewidth=2.5, label="Flow of Tank 1")
ax2.step(t_u_step, u2_step, color="tomato", where="post", linewidth=2.5, label="Flow of Tank 2")

ax2.set_xlabel("Time [min]")
ax2.set_ylabel(r"Flow [m$^3$/s]")  # change to cm^3/s if your units are cm^3/s
ax2.grid(True)
ax2.legend(loc="upper left")

# match your screenshot x-range
ax2.set_xlim(0, 20)

plt.tight_layout()
plt.savefig(r"Figures/Problem12/Problem_12_4_NMPC_plot.png", dpi=200)
plt.show()


