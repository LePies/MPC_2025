import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\4_semester\Model predictive control\MPC_2025\old_casadi\CasADi_framework_example\Problem13_nonlinear.xlsx"

# Read sheets
inputs  = pd.read_excel(file, sheet_name="Inputs")
cont    = pd.read_excel(file, sheet_name="Continuous")

# --- Continuous signals (smooth lines) ---
t_cont = cont["time_min"].to_numpy()

z1 = cont["z1"].to_numpy()
z2 = cont["z2"].to_numpy()


# --- Inputs (stairs) ---
t_u = inputs["time_min"].to_numpy()
u1  = inputs["u1"].to_numpy()
u2  = inputs["u2"].to_numpy()

t_u_step = np.r_[t_u, t_u[-1] + (t_u[-1] - t_u[-2] if len(t_u) > 1 else 0)]
u1_step  = np.r_[u1, u1[-1]]
u2_step  = np.r_[u2, u2[-1]]

# ---------- Plot ----------
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(13, 8), sharex=True)

# ===== Top: Heights =====
ax1.plot(t_cont, z1, color="dodgerblue", linewidth=2.5, label="Height of Tank 1")
ax1.plot(t_cont, z2, color="tomato", linewidth=2.5, label="Height of Tank 2")
ax1.axhline(40, color='black', linestyle='--', linewidth=1.5, label="Minimum Height")
ax1.set_ylabel("Height [m]")  # change to [cm] if that's your unit
ax1.grid(True)
ax1.legend(loc="lower right", ncol=2)

# ===== Bottom: Inputs =====
ax2.step(t_u_step, u1_step, color="dodgerblue", where="post", linewidth=2.5, label="Flow of Tank 1")
ax2.step(t_u_step, u2_step, color="tomato", where="post", linewidth=2.5, label="Flow of Tank 2")

ax2.set_ylabel(r"Flow [m$^3$/s]")  # change to cm^3/s if your units are cm^3/s
ax2.grid(True)
ax2.legend(loc="upper left")

# match your screenshot x-range
#ax2.set_xlim(0, 20)

def cost(t):
    return 200-190*np.exp(-0.01*t*60)

ax3.plot(t_u, cost(t_u), '--',color="black", linewidth=2.5, label="Cost function")
ax3.set_ylabel("Cost")
ax3.grid(True)
ax3.legend(loc="upper right")

ax4.plot(t_u, cost(t_u)*u1+cost(t_u)*u2,'--', color="black", linewidth=2.5, label=r"$c* u$")
ax4.set_xlabel("Time [min]")    
ax4.legend(loc="upper right")
ax4.set_ylabel("Cost-weighted Input")
ax4.grid(True)


print("Total cost:", np.sum(cost(t_u)*u1+cost(t_u)*u2))
plt.tight_layout()
plt.savefig(r"Figures/Problem13/Problem_13_nonlinear.png", dpi=200)
plt.show()


