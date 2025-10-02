import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


class FourTankSystem:
    def __init__(
        self,
        R_s,
        R_d,
        p,
        delta_t,
        sigma_f3 = 20,
        sigma_f4 = 20,
        a_f3 = 1,
        a_f4 = 1,
        F3 = 100,
        F4 = 120
    ):
        self.R_s = R_s
        self.R_d = R_d
        self.delta_t = delta_t
        self.a = p[:4]
        self.A = p[4:8]
        self.rho = p[-1]
        self.gamma = p[8:10]
        self.g = p[10]
        self.sigma_f3 = sigma_f3
        self.sigma_f4 = sigma_f4
        self.a_f3 = a_f3
        self.a_f4 = a_f4
        self.F3 = F3
        self.F4 = F4

    def CheckInputDimension(self, states, d):
        if states.shape[0] == 4:
            if d.shape[0] != 2:
                raise ValueError("Wrong shape of d was given\nif states (x) is 4x1 then d must be a 2x1 array")
            return
        elif states.shape[0] == 6:
            if d.shape[0] != 0:
                raise ValueError("Wrong shape of d was given\nif states (x) is 6x1 then d must be a 0x1 array")
            return
        else:
            raise ValueError("Wrong shape of states was given\nstates must be either a 4x1 array or a 6x1 array")

    def StateEquation(self, t, states, u):

        x = states[:4]
        d = states[4:]
        qin = np.array([self.gamma[0]*u[0],self.gamma[1]*u[1],(1-self.gamma[1])*u[1],(1-self.gamma[0])*u[0]])
        h = x/(self.rho*self.A)
        qout = self.a*np.sqrt(2*self.g*h)
        x1dot = self.rho*(qin[0]+qout[2]-qout[0])
        x2dot = self.rho*(qin[1]+qout[3]-qout[1])
        x3dot = self.rho*(qin[2]-qout[2] + self.rho * d[0])
        x4dot = self.rho*(qin[3]-qout[3] + d[1])

        return np.array([x1dot, x2dot, x3dot, x4dot, 0, 0]) 

    def StateSensor(self, x):
        y = x/(self.rho*self.A)
        noise = np.random.multivariate_normal(np.zeros(self.R_s.shape[0]),self.R_s)
        return y + noise

    def StateOutput(self, y):
        S = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        z = S @ y
        return z

    def DisturbanceEquation(self, t, states, u):
        x = states[:4]
        d = states[4:]
        a = np.array([self.a_f3, self.a_f4])
        F_bar = np.array([self.F3, self.F4])
        sigma = np.array([self.sigma_f3, self.sigma_f4])
        dw = np.random.normal(0, self.delta_t,2)
        ddot = a*(F_bar - d) + sigma*dw
        return ddot

    def DisturbanceSensor(self, d):
        noise = np.random.multivariate_normal(np.zeros(self.R_d.shape[0]),self.R_d)
        return d + noise

    def DisturbanceOutput(self, z):
        S = np.eye(2)
        return S @ z

    def FullEquation(self, t, states, u):
        x = states[:4]
        d = states[4:]
        xdot = self.StateEquation(t, states, u)
        ddot = self.DisturbanceEquation(t, states, u)
        states_dot = np.concatenate([xdot[:4], ddot])
        return states_dot

    def SetLoopStates(self, states, d):
        if states.shape[0] == 4:
            if d.shape[0] != 2:
                raise ValueError("Wrong shape of d was given\nif states (x) is 4x1 then d must be a 2x1 array")
            if np.size(d.shape) == 1:
                states = np.concatenate([states, d])
            else:
                states = np.concatenate([states, d[:, 0]])
            return states, True
        else:
            return states, False

    def GetSteadyState(self, states, u, d = np.array([])):
        self.CheckInputDimension(states, d)

        state_0, is_deterministic = self.SetLoopStates(states, d)
        d = state_0[4:]
        if callable(u):
            def f_steady(x):
                states = np.concatenate([x, d])
                ut = u(states)
                d_states = self.StateEquation(0, states, ut)
                return d_states[4:]
        else:
            def f_steady(x):
                states = np.concatenate([x, d])
                d_states = self.StateEquation(0, states, u)
                return d_states[:4]
        steady_state = fsolve(f_steady, state_0[:4])
        return steady_state[:4]

    def OpenLoop(self, tspan, states, u, d = np.array([])):
        if np.size(tspan) != 2:
            raise ValueError("Wrong shape of tspan was given\ntspan must be a 2x1 array")
        self.CheckInputDimension(states, d)

        t0 = tspan[0]
        tf = tspan[1]
        t_array = np.arange(t0, tf, self.delta_t)

        states, is_deterministic = self.SetLoopStates(states, d)

        states_array = np.zeros([states.shape[0], t_array.shape[0]])
        states_array[:, 0] = states

        h_array = np.zeros([self.R_s.shape[0], t_array.shape[0]])
        h_array[:, 0] = self.StateSensor(states_array[:4, 0])

        f = self.StateEquation if is_deterministic else self.FullEquation

        for i in range(1, t_array.shape[0]):
            ut = u[:, i-1]
            sol = solve_ivp(f, (t_array[i-1], t_array[i]), states_array[:, i-1], method='RK45',args = (ut,))
            state_new = sol.y[:,-1]

            if is_deterministic:
                x_new = state_new
                x_new[4:] = d[:, i]
            else:
                x_new = state_new
            states_array[:, i] = x_new
            h_array[:, i] = self.StateSensor(states_array[:4, i])

        x_array = states_array[:4, :]
        d_array = states_array[4:, :]
        return t_array, x_array, u, d_array, h_array

    def ClosedLoop(self, tspan, states, controller, d = np.array([])):
        if np.size(tspan) != 2:
            raise ValueError("Wrong shape of tspan was given\ntspan must be a 2x1 array")
        self.CheckInputDimension(states, d)

        t0 = tspan[0]
        tf = tspan[1]
        t_array = np.arange(t0, tf, self.delta_t)

        states, is_deterministic = self.SetLoopStates(states, d)

        states_array = np.zeros([states.shape[0], t_array.shape[0]])
        states_array[:, 0] = states

        h_array = np.zeros([self.R_s.shape[0], t_array.shape[0]])
        h_array[:, 0] = self.StateSensor(states_array[:4, 0])

        u_array = np.zeros([2, t_array.shape[0]])

        f = self.StateEquation if is_deterministic else self.FullEquation

        for i in range(1, t_array.shape[0]):
            zt = self.StateOutput(h_array[:, i-1])
            ut = controller.update(zt)
            sol = solve_ivp(f, (t_array[i-1], t_array[i]), states_array[:, i-1], method='RK45',args = (ut,))
            state_new = sol.y[:, -1]

            if is_deterministic:
                x_new = state_new
                x_new[4:] = d[:, i]
            else:
                x_new = state_new

            states_array[:, i] = x_new
            h_array[:, i] = self.StateSensor(states_array[:4, i])
            u_array[:, i-1] = ut

        x_array = states_array[:4, :]
        d_array = states_array[4:, :]
        return t_array, x_array, u_array, d_array, h_array 
    
    def LinearizeDeterminitistic(self,xs):

        h1 = xs[0] / (self.rho * self.A[0])
        h2 = xs[1] / (self.rho * self.A[1])
        h3 = xs[2] / (self.rho * self.A[2])
        h4 = xs[3] / (self.rho * self.A[3])

        A_vals = np.array([self.A[0],self.A[1],self.A[2],self.A[3]])
        a_vals = np.array([self.a[0],self.a[1],self.a[2],self.a[3]])
        h_vals = np.array([h1,h2,h3,h4])

        T = A_vals*np.sqrt(2*self.g*h_vals)/(a_vals*self.g)

        A=np.array([[-1/T[0], 0, 1/T[2], 0],
                    [0,-1/T[1], 0, 1/T[3]],
                    [0, 0,-1/T[2], 0],
                    [0, 0, 0,-1/T[3]]])
        B=np.array([[self.rho*self.gamma[0], 0],[0, self.rho*self.gamma[1]],[0, self.rho*(1-self.gamma[1])],[self.rho*(1-self.gamma[0]), 0]])
        E = np.array([[0,0],[0,0],[self.rho,0],[0,self.rho]])
        C=np.diag(1./(self.rho*A_vals))
        Cz=C[:2,:]

        return A,B,C,E,Cz

    def LinearizeContinousTime(self,xs,d):

        xs, is_deterministic = self.SetLoopStates(xs,d)

        Ass,Bs,C,E,Cz = self.LinearizeDeterminitistic(xs)

        if is_deterministic:
            return Ass,Bs,C,E,Cz
        
        Ads = np.array([[0,0],
                        [0,0],
                        [self.rho,0],
                        [0,self.rho]])
        Asd = np.zeros((4, 2))
        Add = np.array([[-self.a_f3], 
                        [-self.a_f4]])
        Ac = np.block([
            [Ass, Ads],
            [Asd, Add]
        ])

        Bd = np.zeros((2,2))
        Bc = np.block([[Bs],[Bd]])
        
        return Ac,Bc,C,Cz
    


