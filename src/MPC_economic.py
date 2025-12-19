import numpy as np
from src.MPC import MPC
from src.QPSolver import qpsolver

class MPC_economic(MPC):
    def __init__(self, cost_array, H, U, **kwargs):
        self.H_limit = H
        self.U_limit = U
        
        # Load system matrices from Problem 4 results to initialize parent MPC
        try:
            data = np.load("Results/Problem4/Problem_4_estimates_d.npz")
            A = data["A"]
            B = data["B"]
            C = data["C"]
            E = data["E"]
            Q = data["Q"]
            
            # Extract state-space matrices (removing disturbance states if present, based on Problem_10_4 usage)
            # Assuming A is (nx+nd)x(nx+nd) and we want nx x nx
            Ad = A[:-2, :-2]
            Bd = B[:-2, :]
            Cz = C[:, :-2]
            
            # Default parameters if not provided in kwargs
            N = kwargs.get('N', 30)
            us = kwargs.get('us', np.array([250, 325]))
            hs = kwargs.get('hs', np.array([10, 10]))
            
            # Remove them from kwargs to avoid duplicate argument error in super().__init__
            kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['N', 'us', 'hs']}
            
            # Default reference trajectories
            U_bar = np.zeros((N, 2))
            R_bar = np.zeros((N, 2))
            
            R_noise = np.eye(2) # Measurement noise covariance
            
            super().__init__(
                N=N,
                us=us,
                hs=hs,
                U_bar=U_bar,
                R_bar=R_bar,
                A=Ad,
                B=Bd,
                C=Cz,
                Q=Q,
                R=R_noise,
                E=E,
                problem="Problem 4",
                **kwargs_clean
            )
            
        except FileNotFoundError:
            print("Warning: Results/Problem4/Problem_4_estimates_d.npz not found.")
            print("Cannot initialize MPC parent class correctly without system matrices.")
            # Depending on usage, might want to raise error or continue if the user handles it
            pass
        except Exception as e:
            print(f"Error initializing MPC parent: {e}")
            raise e

        self.cost_array = cost_array
        self.current_step = 0 # Track current simulation step for cost array slicing
        
        # Store absolute bounds
        self.Umin_abs = self.compute_Umin(np.array([0, 0]))
        self.Umax_abs = self.compute_Umax(np.array([U, U]))
        self.Rmin_abs = self.compute_Rmin(np.array([H, H]))
        self.Rmax_abs = self.compute_Rmax(np.array([1e5, 1e5])) 
        
        # Initialize working bounds (will be updated in set_xk)
        # We manually call set_xk here to ensure bounds are correctly set relative to operating point
        # This is necessary because super().__init__ called set_xk before _abs variables were set
        self.set_xk()
        
        self.Wt1 = np.zeros((self.N*self.noutput, self.N*self.noutput))
        self.Wt2 = np.zeros((self.N*self.noutput, self.N*self.noutput))
        self.Wz = np.zeros((self.N*self.noutput, self.N*self.noutput))
        self.Wdu = np.zeros((self.N*self.ninput, self.N*self.ninput))
        self._compute_Ws()
        self._compute_Wu()
        
    def _compute_Ws(self):
        n_array = np.concatenate([[1/n for _ in range(self.noutput)] for n in range(1, self.N + 1)])
        self.Ws1 = 1e3*np.diag(n_array)
        self.Ws2 = 1e6*np.diag(n_array)
    
    def _compute_Wu(self):
        # Compute Wu based on current_step
        # If simulation goes beyond cost_array length, use last value
        indices = np.arange(self.current_step, self.current_step + self.N)
        # Clip indices to max length of cost_array
        indices = np.clip(indices, 0, len(self.cost_array) - 1)
        
        current_costs = self.cost_array[indices]
        
        c_n_array = np.concatenate([[current_costs[i] for _ in range(self.ninput)] for i in range(self.N)])
        self.Wu = np.diag(c_n_array)

    def kron(self, W):
        return W

    def set_xk(self):
        # Same as parent: reset state
        self.xk = np.zeros(self.A.shape[0])
        self.xk_kf = np.zeros(self.A_kf.shape[0])

        # Reset bounds to absolute values before shifting
        # This prevents the "shrinking bounds" bug in the parent class
        if hasattr(self, 'Rmin_abs'):
            self.Rmin = self.Rmin_abs.copy()
            self.Rmax = self.Rmax_abs.copy()
            self.Umin = self.Umin_abs.copy()
            self.Umax = self.Umax_abs.copy()
        
        # Now apply the shift relative to operating point (same logic as parent)
        # Note: self.uadd and self.hs are set in _initialize/init
        if self.Rmin is not None:
            self.Rmin = self.Rmin - np.tile(self.hs, self.N)
        if self.Rmax is not None:
            self.Rmax = self.Rmax - np.tile(self.hs, self.N)
        if self.Umin is not None:
            self.Umin = self.Umin - np.tile(self.uadd, self.N)
        if self.Umax is not None:
            self.Umax = self.Umax - np.tile(self.uadd, self.N)
            
        # IMPORTANT: Do NOT append to predicted arrays in set_xk if this is just a bound reset!
        # Parent set_xk appends to predicted arrays, but we don't want to duplicate entries if called multiple times.
        # However, parent set_xk initializes them.
        # We can just clear them or let them be. 
        # But wait, parent set_xk is ONLY called in __init__. It is NOT called in update loop.
        # So it's safe to append here as it's initialization.
        if len(self.predicted_x_mpc) == 0: # Only append if empty (initialization)
             self.predicted_x_mpc.append(self.xk_kf)
             self.predicted_y_mpc.append(self.C_kf@self.xk_kf)
             self.predicted_Px.append(self.Pk)
             self.predicted_Py.append(self.C_kf@self.Pk@self.C_kf.T)
        
        self.wk = np.zeros(self.E.shape[1])

    def update(self, zk):
        # Update Wu before calling parent update or solver
        self._compute_Wu()
        
        # Call parent update
        u_applied = super().update(zk)
        
        # Increment step counter
        self.current_step += 1
        
        return u_applied

class MPC_economic_LP(MPC):
    def __init__(self, cost_array, H, U, **kwargs):
        self.H_limit = H
        self.U_limit = U
        
        # Load system matrices from Problem 4 results to initialize parent MPC
        try:
            data = np.load("Results/Problem4/Problem_4_estimates_d.npz")
            A = data["A"]
            B = data["B"]
            C = data["C"]
            E = data["E"]
            Q = data["Q"]
            
            # Extract state-space matrices (removing disturbance states if present, based on Problem_10_4 usage)
            # Assuming A is (nx+nd)x(nx+nd) and we want nx x nx
            Ad = A[:-2, :-2]
            Bd = B[:-2, :]
            Cz = C[:, :-2]
            
            # Default parameters if not provided in kwargs
            N = kwargs.get('N', 30)
            us = kwargs.get('us', np.array([250, 325]))
            hs = kwargs.get('hs', np.array([10, 10]))
            
            # Remove them from kwargs to avoid duplicate argument error in super().__init__
            kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['N', 'us', 'hs']}
            
            # Default reference trajectories
            U_bar = np.zeros((N, 2))
            R_bar = np.zeros((N, 2))
            
            R_noise = np.eye(2) # Measurement noise covariance
            
            super().__init__(
                N=N,
                us=us,
                hs=hs,
                U_bar=U_bar,
                R_bar=R_bar,
                A=Ad,
                B=Bd,
                C=Cz,
                Q=Q,
                R=R_noise,
                E=E,
                problem="Problem 4",
                **kwargs_clean
            )
            
        except FileNotFoundError:
            print("Warning: Results/Problem4/Problem_4_estimates_d.npz not found.")
            print("Cannot initialize MPC parent class correctly without system matrices.")
            # Depending on usage, might want to raise error or continue if the user handles it
            pass
        except Exception as e:
            print(f"Error initializing MPC parent: {e}")
            raise e

        self.cost_array = cost_array
        self.current_step = 0 # Track current simulation step for cost array slicing
        
        # Store absolute bounds
        self.Umin_abs = self.compute_Umin(np.array([0, 0]))
        self.Umax_abs = self.compute_Umax(np.array([U, U]))
        self.Rmin_abs = self.compute_Rmin(np.array([H, H]))
        self.Rmax_abs = self.compute_Rmax(np.array([1e5, 1e5])) 
        
        # Initialize working bounds (will be updated in set_xk)
        # We manually call set_xk here to ensure bounds are correctly set relative to operating point
        # This is necessary because super().__init__ called set_xk before _abs variables were set
        self.set_xk()
        self.gs = 1e6*np.concatenate([[1/N**2 for _ in range(self.noutput)] for _ in range(self.N)])

        if self.Dmin is None:
            self.Dmin = -np.ones(self.N*self.ninput)*np.inf
        if self.Dmax is None:
            self.Dmax = np.ones(self.N*self.ninput)*np.inf

    def set_xk(self):
        # Same as parent: reset state
        self.xk = np.zeros(self.A.shape[0])
        self.xk_kf = np.zeros(self.A_kf.shape[0])

        # Reset bounds to absolute values before shifting
        # This prevents the "shrinking bounds" bug in the parent class
        if hasattr(self, 'Rmin_abs'):
            self.Rmin = self.Rmin_abs.copy()
            self.Rmax = self.Rmax_abs.copy()
            self.Umin = self.Umin_abs.copy()
            self.Umax = self.Umax_abs.copy()
        
        # Now apply the shift relative to operating point (same logic as parent)
        # Note: self.uadd and self.hs are set in _initialize/init
        if self.Rmin is not None:
            self.Rmin = self.Rmin - np.tile(self.hs, self.N)
        if self.Rmax is not None:
            self.Rmax = self.Rmax - np.tile(self.hs, self.N)
        if self.Umin is not None:
            self.Umin = self.Umin - np.tile(self.uadd, self.N)
        if self.Umax is not None:
            self.Umax = self.Umax - np.tile(self.uadd, self.N)
            
        # IMPORTANT: Do NOT append to predicted arrays in set_xk if this is just a bound reset!
        # Parent set_xk appends to predicted arrays, but we don't want to duplicate entries if called multiple times.
        # However, parent set_xk initializes them.
        # We can just clear them or let them be. 
        # But wait, parent set_xk is ONLY called in __init__. It is NOT called in update loop.
        # So it's safe to append here as it's initialization.
        if len(self.predicted_x_mpc) == 0: # Only append if empty (initialization)
             self.predicted_x_mpc.append(self.xk_kf)
             self.predicted_y_mpc.append(self.C_kf@self.xk_kf)
             self.predicted_Px.append(self.Pk)
             self.predicted_Py.append(self.C_kf@self.Pk@self.C_kf.T)
        
        self.wk = np.zeros(self.E.shape[1])
    
    def _compute_g(self):
        indices = np.arange(self.current_step, self.current_step + self.N)
        # Clip indices to max length of cost_array
        indices = np.clip(indices, 0, len(self.cost_array) - 1)
        
        current_costs = self.cost_array[indices]
        
        c_n_array = np.concatenate([[current_costs[i] for _ in range(self.ninput)] for i in range(self.N)])

        return np.concatenate([c_n_array, self.gs])

    def _compute_A(self):
        I = np.eye(self.N*self.noutput)
        Zero = np.zeros((self.N*self.noutput, self.N*self.noutput))
        A = np.block([
            [self.Gamma, I],
            [I, Zero],
            [-I, Zero],
            [Zero, I],
            [self.Lamb, Zero],
            [-self.Lamb, Zero],
        ])
        return A
    
    def _compute_b(self):
        bk = self.phi_x.T@self.xk + self.phi_w.T@self.wk

        Zero = np.zeros((self.N*self.noutput))
        b = np.block([
            self.Rmin - bk,
            self.Umin,
            -self.Umax,
            Zero,
            self.Dmin + self.I0@self.uk,
            -self.Dmax - self.I0@self.uk,
        ])
        return b

    def MPC_output_constraints(self):
        g = self._compute_g()
        
        # Dimensions for variables: x = [u (N*ninput), epsilon (N*noutput)]
        n_vars = self.N * self.ninput + self.N * self.noutput
        
        H = np.zeros((n_vars, n_vars))
        A = self._compute_A()
        b_l = self._compute_b()
        b_u = np.inf * np.ones_like(b_l)
        
        # Lower bounds for variables:
        # u (inputs) can be negative (deviation variables) -> -inf
        # epsilon (slacks) must be non-negative -> 0
        l_u = -np.inf * np.ones(self.N * self.ninput)
        l_eps = np.zeros(self.N * self.noutput)
        l = np.concatenate([l_u, l_eps])
        
        u = np.inf * np.ones(n_vars)
        ust_init = np.zeros(n_vars)

        ust_fin, info = qpsolver(H, g, l, u, A, b_l, b_u, ust_init)

        u_fin = ust_fin[0:self.ninput*self.N]

        return u_fin.reshape(-1, self.ninput).T, info["f"]
    
    def update(self, zk):
        
        # Call parent update
        u_applied = super().update(zk)
        
        # Increment step counter
        self.current_step += 1
        
        return u_applied

