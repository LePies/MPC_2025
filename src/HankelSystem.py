import numpy as np
from scipy.integrate import solve_ivp

class HankelSystem:
    """
    Hankel system is a system of the form:
    x_dot = A*x + B*u
    y = C*x + D*u
    """
    
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    
    def SteadyState(self, u0, discrete_time: bool = False):
        """
        Calculate steady state.
        
        For continuous-time: 0 = A*x + B*u, so x = -A^(-1)*B*u
        For discrete-time: x = A*x + B*u, so x = (I - A)^(-1)*B*u
        
        Args:
            u0: Input vector
            discrete_time: If True, use discrete-time formula; if False, use continuous-time
        
        Returns:
            x: Steady state vector
        """
        u0 = np.asarray(u0)
        # Ensure u0 is a column vector for matrix multiplication
        if u0.ndim == 1:
            u0 = u0.reshape(-1, 1)
        
        if discrete_time:
            # Discrete-time: x = A*x + B*u at steady state
            # (I - A)*x = B*u
            I = np.eye(self.A.shape[0])
            x_ss = np.linalg.solve(I - self.A, self.B @ u0)
        else:
            # Continuous-time: 0 = A*x + B*u at steady state
            # A*x = -B*u
            x_ss = -np.linalg.solve(self.A, self.B @ u0)
        
        # Return as 1D array
        return x_ss.flatten()
    
    def GetEigenvalues(self):
        return np.linalg.eig(self.A)[0]
    
    def Simulation(self, x0: np.ndarray, u: np.ndarray, t_span: tuple,
                   delta_t: float, discrete_time: bool = True):
        """
        Simulate the system.
        
        Parameters:
        -----------
        x0 : initial state
        u : input (constant or time-varying)
        t_span : (t0, tf) time span
        delta_t : time step
        discrete_time : if True, use discrete-time simulation (x[k+1] = A*x[k] + B*u[k])
                       if False, use continuous-time simulation (x_dot = A*x + B*u)
        """
        if discrete_time:
            return self._simulate_discrete(x0, u, t_span, delta_t)
        else:
            return self._simulate_continuous(x0, u, t_span, delta_t)
    
    def _simulate_continuous(self, x0: np.ndarray, u: np.ndarray, t_span: tuple,
                             delta_t: float):
        """Continuous-time simulation using solve_ivp"""
        def f(t, x):
            return self.A@x + self.B@u
        # Ensure x0 is 1D for solve_ivp
        x0 = np.asarray(x0).flatten()
        t_eval = np.linspace(
            t_span[0], t_span[1],
            int((t_span[1] - t_span[0])/delta_t)
        )
        sol = solve_ivp(f, t_span, x0, method='RK45', t_eval=t_eval)
        # Calculate outputs y = C*x + D*u
        u_col = np.asarray(u).reshape(-1, 1) if np.asarray(u).ndim == 1 else np.asarray(u)
        y = self.C @ sol.y + self.D @ u_col
        return sol.t, y, sol.y
    
    def _simulate_discrete(self, x0: np.ndarray, u: np.ndarray, t_span: tuple,
                           delta_t: float):
        """Discrete-time simulation: x[k+1] = A*x[k] + B*u[k], y[k] = C*x[k] + D*u[k]"""
        # Ensure x0 is 1D
        x0 = np.asarray(x0).flatten()
        u = np.asarray(u)
        
        # Create time array
        t = np.arange(t_span[0], t_span[1] + delta_t, delta_t)
        n_steps = len(t) - 1
        
        # Initialize state and output arrays
        n_states = len(x0)
        n_outputs = self.C.shape[0]
        x = np.zeros((n_states, n_steps + 1)) 
        y = np.zeros((n_outputs, n_steps + 1))
        
        # Set initial condition
        x[:, 0] = x0
        
        # Ensure u is properly shaped (constant or time-varying)
        if u.ndim == 1:
            # Constant input - broadcast to all time steps
            u_col = u.reshape(-1, 1)
        else:
            # Time-varying input
            u_col = u
        
        # Calculate initial output
        if u_col.shape[1] == 1:
            y[:, 0] = (self.C @ x[:, 0] + self.D @ u_col[:, 0]).flatten()
        else:
            y[:, 0] = (self.C @ x[:, 0] + self.D @ u_col[:, 0]).flatten()
        
        # Simulate discrete-time system
        for k in range(n_steps):
            # Get input for current time step
            if u_col.shape[1] == 1:
                u_k = u_col[:, 0]  # Constant input
            else:
                u_k = u_col[:, k] if k < u_col.shape[1] else u_col[:, -1]
            
            # Discrete-time update: x[k+1] = A*x[k] + B*u[k] 
            x[:, k+1] = self.A @ x[:, k] + self.B @ u_k
            
            # Output: y[k+1] = C*x[k+1] + D*u[k+1]'
            y[:, k+1] = (self.C @ x[:, k+1] + self.D @ u_k).flatten()
        
        return t, y, x