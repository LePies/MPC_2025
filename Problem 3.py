import system as sys
import numpy as np
import parameters_tank as para 
import matplotlib.pyplot as plt
import compute_steady_state as css
import simulations as sim
import controllers as controllers

t0 = 0
tf = 20*90
m10 = 0
m20 = 0
m30 = 0
m40 = 0
F1 = 250
F2 = 325
F3 = 100
F4 = 120
x0 = np.array([m10,m20,m30,m40])
u0 = np.array([F1,F2])
d0 = np.array([F3,F4])
p = para.parameters()
a1,a2,a3,a4,A1,A2,A3,A4,gamma1,gamma2,g,rho = p

# Computing steady state
xs = css.compute_steady_state(sys.f_modified, x0, u0, d0, p)
print("Steady state (m):", xs)
print("Steady state (h):", xs/(rho*np.array([A1,A2,A3,A4])))


r = np.array([1.0, 2.0])  # Set point
Kc = 0.8 #Kp = 4, Ki = 0.1, Kd = 0.5 
Ti = 0.025*Kc
Td = 0.8/Kc
Ts = 1.0
umin = 10
umax = 10000
N = 0
p_inputs = Kc, umin, umax
pi_inputs = 0, Kc, Ti, Ts, umin, umax
pid_inputs = 0, Kc, Ti, Ts,Td, umin, umax, 0

 
#%% Determinitic open-loop simulation

# Define disturbances
D = np.zeros((2, tf))
D[0, :] = F3
D[1, :] = F4 

Rvv = np.eye(4)*0.01

T,X,U,Y,Z = sim.closed_loop(sys.f,sys.g,sys.h,t0,tf,x0,r,u0,p,Rvv,controllers.PID_controller,pid_inputs)

# Plotting example
fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
axes[0].plot(T.ravel(), Y[0, :], label='$H_1$')
axes[0].plot(T.ravel(), Y[1, :], label='$H_2$')
axes[0].plot(T.ravel(), Y[2, :], label='$H_3$')
axes[0].plot(T.ravel(), Y[3, :], label='$H_4$')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Height [m]')
axes[0].legend()
axes[0].set_title('Open loop simulation of deterministic model')
axes[1].step(T.ravel(), U[0, :], label='$F_1$')
axes[1].step(T.ravel(), U[1, :], label='$F_2$')
axes[1].step(T.ravel(), D[0, :], label='$F_3$')
axes[1].step(T.ravel(), D[1, :], label='$F_4$')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Flow [m³/s]')
axes[1].legend()

#%% Open loop simulation of stochastic model

# Define stochastic distrubances
D = np.zeros((2, tf))
mu1 = np.array([100])
mu2 = np.array([120])
sigma1 = 0.1
sigma2 = 0.1
F3 = np.random.normal(mu1,sigma1)[0]
F4 = np.random.normal(mu2,sigma2)[0] 
D[0, :] = F3
D[1, :] = F4

Rvv = np.eye(4)*0.01

T,X,U,Y,Z = sim.closed_loop(sys.f,sys.g_sensor,sys.h_sensor,t0,tf,x0,r,u0,p,Rvv,controllers.PID_controller,pid_inputs)

# Plotting example
fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True)
axes[0].plot(T.ravel(), Y[0, :], label='$H_1$')
axes[0].plot(T.ravel(), Y[1, :], label='$H_2$')
axes[0].plot(T.ravel(), Y[2, :], label='$H_3$')
axes[0].plot(T.ravel(), Y[3, :], label='$H_4$')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Height [m]')
axes[0].legend()
axes[0].set_title('Open loop simulation of stochastic model')
axes[1].step(T.ravel(), U[0, :], label='$F_1$')
axes[1].step(T.ravel(), U[1, :], label='$F_2$')
axes[1].step(T.ravel(), D[0, :], label='$F_3$')
axes[1].step(T.ravel(), D[1, :], label='$F_4$')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Flow [m³/s]')
axes[1].legend()




plt.show()




