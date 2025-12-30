import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load EKF data (Problem 12.2)
data = pd.read_excel(
    r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\4_semester\Model predictive control\MPC_2025\old_casadi\CasADi_framework_example\Problem12_2_EKF_Data.xlsx",
    sheet_name="EKF_Data"
)
std = pd.read_excel(
    r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\4_semester\Model predictive control\MPC_2025\old_casadi\CasADi_framework_example\Problem12_2_EKF_Data.xlsx",
    sheet_name="EKF_Std"
)

std_x1 = std["std_x1"]
std_x2 = std["std_x2"]
std_x3 = std["std_x3"]
std_x4 = std["std_x4"]

# Extract signals
time = data["time_min"]

y1 = data["y1_meas"]
y2 = data["y2_meas"]

z1_hat = data["z1_hat"]
z2_hat = data["z2_hat"]

x1_hat = data["x1_hat_kg"]
x2_hat = data["x2_hat_kg"]
x3_hat = data["x3_hat_kg"]
x4_hat = data["x4_hat_kg"]

# ------------------- Plot ------------------- #
plt.figure(figsize=(12, 8))

# ---- Subplot 1: Outputs ----
plt.subplot(2, 1, 1)

plt.plot(time, z1_hat, color='dodgerblue', linewidth=3, label=r'Estimated height of Tank 1')
plt.plot(time, z2_hat, color='tomato', linewidth=3, label=r'Estimated height of Tank 2')

plt.scatter(time, y1, s=25, marker='.', label=r'Measured height of Tank 1')
plt.scatter(time, y2, s=25, marker='.', label=r'Measured height of Tank 2')

plt.grid(True)
plt.ylabel('Height [m]')
plt.legend(ncol=2)
plt.xlabel("")
plt.xlim(0, 20)

# ---- Subplot 2: EKF state estimates ----
plt.subplot(2, 1, 2)

plt.subplot(2, 1, 2)

plt.plot(time, x1_hat, color='dodgerblue', linewidth=3, label='Tank 1')
plt.fill_between(
    time,
    x1_hat - 2*std_x1,
    x1_hat + 2*std_x1,
    color='dodgerblue',
    alpha=0.9
)

plt.plot(time, x2_hat, color='tomato', linewidth=3, label='Tank 2')
plt.fill_between(
    time,
    x2_hat - 2*std_x2,
    x2_hat + 2*std_x2,
    color='tomato',
    alpha=0.9
)

plt.plot(time, x3_hat, color='limegreen', linewidth=3, label=r'Estimated mass of Tank 3')
plt.plot(time, x4_hat, color='orange', linewidth=3, label=r'Estimated mass of Tank 4')

plt.grid(True)
plt.xlabel('Time [min]')
plt.ylabel('Mass [kg]')
plt.legend(ncol=2)
plt.xlim(0, 20)

plt.tight_layout()
plt.savefig(r"Figures/Problem12/Problem_12_2_EKF_estimates.png")
plt.close()
