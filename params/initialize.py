import params.parameters_tank as para
import numpy as np

def initialize():

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
    R_s = np.eye(4)*0.1
    R_d = np.eye(2)*0.1
    delta_t = 1

    return x0, u, d, p , R_s, R_d, delta_t