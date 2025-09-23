import numpy as np 

def P_controller(us,y,r,inputs):
    Kc,umin,umax  = inputs

    e = r-y
    v = us + Kc*e
    u = np.clip(v, umin, umax)

    return u

def PI_controller(us, y, r, inputs):
    i, Kc, Ti, Ts, umin, umax = inputs
    e = r - y
    i_candidate = i + (Kc * Ts / Ti) * e
    v = us + Kc * e + i_candidate
    u = np.clip(v, umin, umax)
    # simple anti-windup
    i_new = i_candidate if np.all(u == v) else i
    return u, i_new

def PID_controller2(us, y, r, inputs):
    i, Kc, Ti, Ts, Td, umin, umax, e_prev = inputs
    e = r - y
    p = Kc * e
    i = i + Ti * e * Ts
    d = (e - e_prev) * Td / Ts  # derivative of error
    v = us + p + i + d
    u = np.clip(v, umin, umax)
    return u, i, e

def PID_controller(us, y, r, inputs):
    i, Kc, Ti, Ts, Td, umin, umax, e_prev = inputs
    e = r - y
    p = Kc * e
    i = i + Ts * e * Kc / Ti
    d = (e - e_prev) *Kc * Td/Ts  # derivative of error
    v = us + p + i + d
    u = np.clip(v, umin, umax)
    return u, i, e


