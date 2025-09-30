import numpy as np
from src.FourTankSystem import FourTankSystem
import params.parameters_tank as para
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv, solve

t0 = 0
tf = 20*60 
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
R_s = np.eye(4)*0.1
R_d = np.eye(2)*0.1
delta_t = 1


# Linearize continous time
Model_Deterministic = FourTankSystem(R_s*0, R_d*0, p, delta_t)
xs = Model_Deterministic.GetSteadyState(x0, u, d)
Ac,Bc,C,Cz = Model_Deterministic.LinearizeContinousTime(xs,d)
print(Ac)
print(Bc)
print(C)
print(Cz)

poles = eig(Ac)[0]
print("Poles:",poles,"\nSystem stable:",np.all(poles<0))
G_transfer = lambda s: C @ np.linalg.inv(s*np.eye(Ac.shape[0]) - Ac) @ Bc


