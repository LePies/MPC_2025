import numpy as np 
from scipy.integrate import solve_ivp


def discrete_openloop(sys,t0,tf,x0,U,p,A,B,C):
 
    X = np.zeros([4,tf])
    T = np.zeros([1,tf])

    xt = x0

    for idx,t in enumerate(range(t0,tf)):

        ut = U[:,idx]
        xt, yt = sys(xt,ut,A,B,C)

        X[:,idx] = xt
        T[:,idx] = t

    return T,X

def openloop(f,g,h,t0,tf,x0,U,D,p,Rvv = 0):

    X = np.zeros([4,tf])
    Y = np.zeros([4,tf])
    Z = np.zeros([2,tf])
    T = np.zeros([1,tf])

    xt = x0

    for idx,t in enumerate(range(t0,tf)):

        ut = U[:,idx]
        dt = D[:,idx]
        yt = g(xt,p,Rvv) # output
        zt = h(yt) # measurement

        sol = solve_ivp(f, (t,t+1), xt, method='RK45',args = (ut,dt,p))

        xt = sol.y[:,-1]
        X[:,idx] = xt
        T[:,idx] = sol.t[-1] 
        Y[:,idx] = yt
        Z[:,idx] = zt

    return T,X,Y,Z


def openloop_SDE(f,g,h,t0,tf,x0,U,D,p,Qww,Rvv):

    X = np.zeros([4,tf])
    T = np.zeros([1,tf])
    Y = np.zeros([4,tf])
    Z = np.zeros([2,tf])

    xt = x0

    delta_t = (tf-t0) / tf

    for idx,t in enumerate(range(t0,tf)):

        ut = U[:,idx]
        dt = D[:,idx]
        yt = g(xt,p,Rvv) # output 
        zt = h(yt) # measurement

        xt = xt + f(xt,ut,dt,delta_t,p,Qww) 

        X[:,idx] = xt
        T[:,idx] = delta_t + t if idx == 0 else T[:,idx-1] + delta_t
        Y[:,idx] = yt
        Z[:,idx] = zt

    return T,X,Y,Z

def closed_loop(f,g,h,t0,tf,x0,r,u0,p,Rvv,controller,inputs):
    
    name = controller.__name__.lower()
        # parse inputs based on controller type
    if name.startswith("pid"):
        i, Kc, Ti, Ts, Td, umin, umax, e = inputs
    elif name.startswith("pi"):
        i,Kc,Ti,Ts,umin,umax  = inputs
    else: 
        Kc,umin,umax = inputs

    X = np.zeros([4,tf])
    U = np.zeros([2,tf])
    T = np.zeros([1,tf])
    Y = np.zeros([4,tf])
    Z = np.zeros([2,tf])

    xt = x0
    ut = u0

    for idx,t in enumerate(range(t0,tf)):

        yt = g(xt,p,Rvv) # output
        zt = h(yt) # measurement

        if name.startswith("pid"):
            inputs = i, Kc, Ti, Ts, Td, umin, umax, e
            ut, i, e = controller(ut, zt, r, inputs)
        elif name.startswith("pi"):
            inputs = i,Kc,Ti,Ts,umin,umax 
            ut, i = controller(ut,zt,r,inputs)
        else: 
            inputs = Kc,umin,umax
            ut = controller(ut,zt,r,inputs)

        sol = solve_ivp(f, (t,t+1), xt, method='RK45',args = (ut,p))

        xt = sol.y[:,-1]
        X[:,idx] = xt
        T[:,idx] = sol.t[-1] 
        U[:,idx] = ut 
        Y[:,idx] = yt
        Z[:,idx] = zt

    return T,X,U,Y,Z

