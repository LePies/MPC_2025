import numpy as np 
import scipy.optimize as opt
from torch import threshold


def optimize_Qn(theta_0,X): 

    def M_setar(theta, x):
        x = np.asarray(x)
        return np.where(x <= threshold, theta[0] + theta[1]*x, theta[2] + theta[3]*x)

    def Qn(theta):
        return np.sum(((X[1:] - M_setar(theta, X[:-1]))**2))

    result = opt.minimize(Qn, theta_0)

    return result.x