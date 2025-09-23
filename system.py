import numpy as np 
import parameters_tank as para
import scipy as sp

def f(t,xt,ut,p):

    # Unpack parameters
    m = xt
    F = ut 
    a = p[:4] 
    A = p[4:8]
    gamma = p[8:10]
    g = p[10]
    rho = p[11:]

    # Inflows 
    qin = np.array([gamma[0]*F[0],gamma[1]*F[1],(1-gamma[1])*F[1],(1-gamma[0])*F[0]])

    # Outflows
    h = m/(rho*A)
    qout = a*np.sqrt(2*g*h)

    # Mass balances eq. 126 page 12
    x1 = rho*(qin[0]+qout[2]-qout[0])
    x2 = rho*(qin[1]+qout[3]-qout[1])
    x3 = rho*(qin[2]-qout[2] )
    x4 = rho*(qin[3]-qout[3] )

    xdot = np.array([x1.T,x2.T,x3.T,x4.T]).T

    return xdot.T[0]

def f_modified(t,xt,ut,dt,p):

    # Unpack parameters
    m = xt
    F = ut 
    F_dist = dt
    a = p[:4]
    A = p[4:8]
    gamma = p[8:10]
    g = p[10]
    rho = p[11:]

    # Inflows 
    qin = np.array([gamma[0]*F[0],gamma[1]*F[1],(1-gamma[1])*F[1],(1-gamma[0])*F[0]])

    # Outflows
    h = m/(rho*A)
    qout = a*np.sqrt(2*g*h)

    # Mass balances eq. 126 page 12
    x1 = rho*(qin[0]+qout[2]-qout[0])
    x2 = rho*(qin[1]+qout[3]-qout[1])
    x3 = rho*(qin[2]-qout[2] + F_dist[0])
    x4 = rho*(qin[3]-qout[3] + F_dist[1])

    xdot = np.array([x1.T,x2.T,x3.T,x4.T]).T

    return xdot[0]
    
def f_SDE(t,xt,ut,dt,delta_t,p,Qww):

    dw = np.random.multivariate_normal(mean=np.zeros(Qww.shape[0]), cov=Qww * delta_t)
    xdot = f_modified(0,xt,ut,dt,p)*delta_t + diffusion_matrix(p)@dw

    return xdot

def diffusion_matrix(p):
    rho = float(p[-1]) 
    G = rho * np.array([
        [0.0, 0.0],   # m1: no direct noise
        [0.0, 0.0],   # m2: no direct noise
        [1, 0.0],    # m3: noise from ω3
        [0.0, 1],    # m4: noise from ω4
    ])
    return G

def g(xt,p,Rvv = 0):

    rho = p[-1]
    A = p[4:8]

    h1 = xt[0]/ (rho*A[0])
    h2 = xt[1]/ (rho*A[1])
    h3 = xt[2]/ (rho*A[2])
    h4 = xt[3]/ (rho*A[3])

    y = np.array([h1.T,h2.T,h3.T,h4.T])

    return y 

def g_sensor(xt,p,Rvv):

    y = g(xt,p) + np.random.multivariate_normal(np.zeros(Rvv.shape[0]),Rvv)

    return y

def h_sensor(yt):
    S = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    z = S @ yt

    return z

def h(yt):

    S = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    z = S @ yt

    return z

def discrete_system(x,u,A,B,C):
    x_next = A @ x + B @ u
    y = C @ x
    return x_next, y

def linearize_continous(xs):

    a1,a2,a3,a4,A1,A2,A3,A4,gamma1,gamma2,g,rho = para.parameters()
    h1 = xs[0] / (rho * A1)
    h2 = xs[1] / (rho * A2)
    h3 = xs[2] / (rho * A3)
    h4 = xs[3] / (rho * A4)

    A_vals = np.array([A1,A2,A3,A4])
    a_vals = np.array([a1,a2,a3,a4])
    h_vals = np.array([h1,h2,h3,h4])

    T = A_vals*np.sqrt(2*g*h_vals)/(a_vals*g)

    A=np.array([[-1/T[0], 0, 1/T[2], 0],
                [0,-1/T[1], 0, 1/T[3]],
                [0, 0,-1/T[2], 0],
                [0, 0, 0,-1/T[3]]])
    B=np.array([[rho*gamma1, 0],[0, rho*gamma2],[0, rho*(1-gamma2)],[rho*(1-gamma1), 0]])
    C=np.diag(1./(rho*A_vals))
    Cz=C[:2,:]

    return A,B,C,Cz

def linearize_discrete(xs,Ts):
    A, B, C, Cz = linearize_continous(xs)

    # Augment A and B for matrix exponential
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = sp.linalg.expm(M * Ts)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    
    return Ad, Bd, C, Cz

