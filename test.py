import matplotlib.pyplot as plt

from src.PlotSimulation import PlotSimulation
from src.FourTankSystem import FourTankSystem


Rvv = np.eye(4)*0.01
p = para.parameters()
delta_t = 1
F3 = 100
F4 = 120

x0 = np.array([0, 0, 0, 0])
u = np.array([250, 325])
d = np.array([100, 120])

Model = FourTankSystem(Rvv, Rvv, p, delta_t, F3 = F3, F4 = F4)

state_0 = np.concatenate([x0, d])
t, x, u, d, h = Model.OpenLoop((0, 100), state_0, u)

fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

PlotSimulation(axes, t, x, u, d, h, legend=True)
plt.show()