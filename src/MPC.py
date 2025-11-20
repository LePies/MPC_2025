import numpy as np
from src.QPSolver import qpsolver
from src.KalmanFilterUpdate import KalmanFilterUpdate

class MPC:
    """
    Model Predictive Control (MPC) class for solving unconstrained MPC problems.
    
    This class implements an MPC controller that:
    - Uses a Kalman filter for state estimation
    - Solves an unconstrained QP problem at each time step
    - Handles reference tracking with input/output weighting
    """
    def __init__(
        self,
        x0, w0, N, U_bar, R_bar, G, A, B, C, Q, R,
        u0 = None,
        problem="Problem 5",
        Wz=None,
        Wu=None,
        Wdu=None,
        Umin = None,
        Umax = None,
        Dmin = None,
        Dmax = None,
        Rmax = None,
        Rmin = None,
    ) -> None:
        """
        Initialize the MPC controller.
        
        Parameters:
        -----------
        u0 : numpy.ndarray
            Initial input vector (nu x 1)
        x0 : numpy.ndarray
            Initial state vector (nx x 1)
        w0 : numpy.ndarray
            Initial disturbance vector (nw x 1)
        N : int
            Prediction horizon length
        U_bar : numpy.ndarray
            Input reference trajectory (N x nu)
        R_bar : numpy.ndarray
            Output reference trajectory (N x ny)
        G : numpy.ndarray
            Disturbance input matrix (nx x nw)
        A : numpy.ndarray
            State transition matrix (nx x nx)
        B : numpy.ndarray
            Input matrix (nx x nu)
        C : numpy.ndarray
            Output matrix (ny x nx)
        Q : numpy.ndarray
            Process noise covariance matrix (nx x nx)
        R : numpy.ndarray
            Measurement noise covariance matrix (ny x ny)
        problem : str
            Problem identifier ("Problem 4" or "Problem 5")
        Wz : numpy.ndarray, optional
            Output weighting matrix (ny x ny). Default: identity
        Wu : numpy.ndarray, optional
            Input weighting matrix (nu x nu). Default: identity
        Wdu : numpy.ndarray, optional
            Input rate weighting matrix (nu x nu). Default: identity
        """
        self.N = N
        self.U_bar = U_bar.flatten()  # Flatten to (N*nu,) vector
        self.R_bar = R_bar.flatten()  # Flatten to (N*ny,) vector
        self.wk = w0
        self.problem = problem
        self.Wz = Wz
        self.Wu = Wu
        self.Wdu = Wdu
        self.C = C
        self.A = A
        self.G = G
        self.B = B
        self.Q = Q
        self.R = R
        self.xk = x0
        if self.problem == "Problem 4":
            self.xk = self.xk[:4]
        self.noutput = C.shape[0]  # Number of outputs (ny)
        self.ninput = B.shape[1]   # Number of inputs (nu) - FIXED: Added to track input dimension shape of Uk and Umin, Umax
        self.uk = self.compute_uk(u0)
        self.Rmax = Rmax
        self.Rmin = Rmin
        
        # Pre-compute matrices that don't change during runtime
        self.Gamma = self.compute_gamma()  # Markov parameter matrix for output prediction
        self.phi_x = self.compute_phi_x()  # State-to-output prediction matrix
        self.phi_w = self.compute_phi_w()  # Disturbance-to-output prediction matrix
        self.Pk = self.compute_Pk()        # Initial covariance matrix for Kalman filter
        self.Umin = self.compute_Umin(Umin)
        self.Umax = self.compute_Umax(Umax)
        self.Dmin = self.compute_Dmin(Dmin)
        self.Dmax = self.compute_Dmax(Dmax)
        self.Rmax = self.compute_Rmax(Rmax)
        self.Rmin = self.compute_Rmin(Rmin)

    def set_umin(self, Umin):
        self.Umin = self.compute_Umin(Umin)
    def set_umax(self, Umax):
        self.Umax = self.compute_Umax(Umax)
    def set_dmin(self, Dmin):
        self.Dmin = self.compute_Dmin(Dmin)
    def set_dmax(self, Dmax):
        self.Dmax = self.compute_Dmax(Dmax)
    def set_rmax(self, Rmax):
        self.Rmax = self.compute_Rmax(Rmax)
    def set_rmin(self, Rmin):
        self.Rmin = self.compute_Rmin(Rmin)

    def compute_Rmax(self, Rmax):
        if Rmax is None:
            return None
        elif Rmax.shape[0] == self.noutput:
            return np.block([Rmax for i in range(self.N)])
        elif Rmax.shape[0] == self.N*self.noutput:
            return Rmax
        else:
            raise ValueError("Invalid Rmax shape")
    
    def compute_Rmin(self, Rmin):
        if Rmin is None:
            return None
        elif Rmin.shape[0] == self.noutput:
            return np.block([Rmin for i in range(self.N)])
        elif Rmin.shape[0] == self.N*self.noutput:
            return Rmin
        else:
            raise ValueError("Invalid Rmin shape")

    def compute_uk(self, u0):
        if u0 is None:
            return np.zeros(self.ninput)
        elif u0.shape[0] == self.ninput:
            return u0
        else:
            raise ValueError("Invalid u0 shape")
        
    def compute_Umin(self, Umin):
        if Umin is None:
            return None
        elif Umin.shape[0] == self.ninput:
            return np.block([Umin for i in range(self.N)])
        elif Umin.shape[0] == self.N*self.ninput:
            return Umin
        else:
            raise ValueError("Invalid Umin shape")

    def compute_Umax(self, Umax):
        if Umax is None:
            return None
        elif Umax.shape[0] == self.ninput:
            return np.block([Umax for i in range(self.N)])
        elif Umax.shape[0] == self.N*self.ninput:
            return Umax
        else:
            raise ValueError("Invalid Umax shape")
    
    def compute_Dmin(self, Dmin):
        if Dmin is None:
            return None
        elif Dmin.shape[0] == self.ninput:
            return np.block([Dmin for i in range(self.N)])
        elif Dmin.shape[0] == self.N*self.ninput:
            return Dmin
        else:
            raise ValueError("Invalid Dmin shape")
    
    def compute_Dmax(self, Dmax):
        if Dmax is None:
            return None
        elif Dmax.shape[0] == self.ninput:
            return np.block([Dmax for i in range(self.N)])
        elif Dmax.shape[0] == self.N*self.ninput:
            return Dmax
        else:
            raise ValueError("Invalid Dmax shape")

    def compute_Pk(self):
        """
        Get initial covariance matrix for Kalman filter.
        
        Returns:
        --------
        Pk : numpy.ndarray
            Initial covariance matrix (nx x nx)
        """
        Pk = 5*np.eye(self.A.shape[0])
        return Pk

    def reset_control(self):
        """Reset the Kalman filter covariance matrix to initial value."""
        self.Pk = self.compute_Pk()

    def compute_phi_x(self):
        """
        Compute the state-to-output prediction matrix phi_x.
        
        phi_x relates future states to outputs: y = phi_x^T * x
        Each block row corresponds to one step in the prediction horizon.
        
        Returns:
        --------
        phix : numpy.ndarray
            State-to-output matrix (nx x N*ny)
        """
        phix = np.block([(self.C@self.A**i).T for i in range(1, self.N+1)])
        return phix
    
    def compute_phi_w(self):
        """
        Compute the disturbance-to-output prediction matrix phi_w.
        
        phi_w relates future disturbances to outputs: y = phi_w^T * w
        Each block row corresponds to one step in the prediction horizon.
        
        Returns:
        --------
        phiw : numpy.ndarray
            Disturbance-to-output matrix (nw x N*ny)
        """
        phiw = np.block([(self.C@(self.A**i)@self.G).T for i in range(self.N)])
        return phiw
        
    def compute_gamma(self):
        """
        Get the Gamma matrix (Markov parameter matrix) for output prediction.
        
        Gamma relates future inputs to outputs: y = Gamma * u
        This matrix contains the system's impulse response (Markov parameters).
        
        Parameters:
        -----------
        N : int
            Prediction horizon length
        problem : str
            "Problem 5" or "Problem 4" - determines which data file to load
            
        Returns:
        --------
        Gamma : numpy.ndarray
            Markov parameter matrix (N*ny x N*nu)
        """
        # Load pre-computed Markov parameters from file
        if self.problem == "Problem 5":
            data = np.load("Results/Problem5/Problem_5_estimates.npz")
        elif self.problem == "Problem 4":
            data = np.load("Results/Problem4/Problem_4_estimates.npz")
        else:
            raise ValueError("Invalid problem")

        markov_mat = data["markov_mat"]

        # Reshape Markov parameters for easier indexing
        markov_mat_test = markov_mat[:, :, :self.N].reshape(2, -1)

        # Initialize Gamma matrix
        Gamma = np.zeros((2*self.N, 2*self.N))
        
        # Build Gamma matrix: each column block corresponds to one input time step
        # The structure creates a lower-triangular Toeplitz-like matrix
        for i in range(self.N):
            markov_input = markov_mat_test[:, :2*(self.N-i)]
            markov_input = np.block([
                [np.zeros((2, 2*i)), markov_input]
            ])
            Gamma[:, 2*i:2*i+2] = markov_input.T

        return Gamma

    def kron(self, W):
        """
        Create block-diagonal matrix by Kronecker product with identity.
        
        This creates a matrix where W is repeated N times along the diagonal.
        Used to create weighting matrices for the entire prediction horizon.
        
        Parameters:
        -----------
        W : numpy.ndarray
            Weighting matrix to repeat (typically ny x ny or nu x nu)
            
        Returns:
        --------
        Wbar : numpy.ndarray
            Block-diagonal matrix (N*size(W) x N*size(W))
        """
        I_N = np.eye(self.N)
        Wbar = np.kron(I_N, W)
        return Wbar
    
    def MPC_input_constraints(self):
        """
        Solve the unconstrained MPC optimization problem.
        
        The MPC problem minimizes:
            J = ||Wz*(y - r)||^2 + ||Wu*(u - u_ref)||^2 + ||Wdu*du||^2
        
        where:
            y: predicted outputs
            r: reference trajectory (R_bar)
            u: input sequence
            u_ref: input reference (U_bar)
            du: input rate (difference between consecutive inputs)
        
        Returns:
        --------
        ufin : numpy.ndarray
            Optimal input sequence (nu x N)
        cost : float
            Optimal cost value
        """
        # Set default weighting matrices if not provided
        if self.Wz is None:
            Wz = np.eye(self.noutput)  # Output weighting (ny x ny)
        else:
            Wz = self.Wz
        if self.Wu is None:
            # FIXED: Should be ninput, not noutput
            Wu = np.eye(self.ninput)   # Input weighting (nu x nu)
        else:
            Wu = self.Wu
        if self.Wdu is None:
            # FIXED: Should be ninput, not noutput
            Wdu = np.eye(self.ninput)  # Input rate weighting (nu x nu)
        else:
            Wdu = self.Wdu

        # Create block-diagonal weighting matrices for entire horizon
        Wz_bar = self.kron(Wz)   # (N*ny x N*ny)
        Wu_bar = self.kron(Wu)   # (N*nu x N*nu)
        Wdu_bar = self.kron(Wdu)  # (N*nu x N*nu)
        
        # I0 matrix: extracts first input from input sequence vector
        # Used to compute input rate: du[0] = u[0] - uk (current input)
        # FIXED: Should use ninput, not noutput
        I0 = np.block([np.eye(self.ninput) if i == 0 else np.zeros((self.ninput, self.ninput)) for i in range(self.N)]).T

        # Compute predicted output based on current state and disturbance
        # bk = predicted output without control action
        bk = self.phi_x.T@self.xk + self.phi_w.T@self.wk

        # Compute tracking error: difference between reference and predicted output
        ck = self.R_bar - bk

        # Lambda matrix: creates input rate (difference) operator
        # Input vector u is flattened as [u0_0, u0_1, u1_0, u1_1, ...] where:
        #   u0_0 is first input at time 0, u0_1 is second input at time 0, etc.
        # Lambda should compute: [u[0]-uk[0]; u[1]-uk[1]; u[2]-u[0]; u[3]-u[1]; u[4]-u[2]; ...]
        # That is, differences between corresponding inputs at consecutive time steps
        Lamb = np.zeros((self.N*self.ninput, self.N*self.ninput))
        for i in range(self.N):
            for j in range(self.ninput):
                row_idx = i*self.ninput + j
                if i == 0:
                    # First time step: constraint is u[0] - uk (handled via bounds adjustment)
                    Lamb[row_idx, row_idx] = 1.0
                else:
                    # Subsequent time steps: constraint is u[i] - u[i-1] for same input j
                    Lamb[row_idx, row_idx] = 1.0  # Current input
                    Lamb[row_idx, (i-1)*self.ninput + j] = -1.0  # Previous input (same input channel)

        # Build Hessian matrix H for QP: H = Hu + Hdu + Hz
        # Hu: penalty on input deviation from reference
        Hu = Wu_bar.T@Wu_bar
        
        # Hdu: penalty on input rate (smoothness)
        Hdu = Lamb.T@Wdu_bar.T@Wdu_bar@Lamb
        
        # Hz: penalty on output tracking error
        Hz = self.Gamma.T@Wz_bar.T@Wz_bar@self.Gamma
        H = Hu + Hdu + Hz

        # Build gradient vector g for QP
        # gu: gradient from input reference tracking
        gu = -Wu_bar.T@Wu_bar@self.U_bar
        
        # gdu: gradient from input rate penalty (depends on current input uk)
        gdu = -Lamb.T@Wdu_bar.T@Wdu_bar@I0@self.uk
        
        # gz: gradient from output tracking error
        # FIXED: Removed extra @self.Gamma term - should match Problem_8.py line 85
        gz = -(Wz_bar@self.Gamma).T@Wz_bar@ck 
        g = gu+gdu+gz

        # Compute constant term rho (doesn't affect optimization, but needed for cost)
        rhou = 1/2*self.U_bar.T@Wu_bar.T@Wu_bar@self.U_bar
        rhodu = 1/2*(Wdu_bar@I0@self.uk).T@(Wdu_bar@I0@self.uk)
        rhoz = 1/2*ck.T@Wz_bar@Wz_bar@ck
        rho = rhodu+rhou+rhoz

        # Initial guess for QP solver (zeros)
        u_init = np.zeros(H.shape[0])

        if self.Umin is not None:
            l = self.Umin
        else:
            l = None
        if self.Umax is not None:
            u = self.Umax
        else:
            u = None

        if self.Dmin is not None or self.Dmax is not None:
            # Constraint: Dmin <= Lamb * u <= Dmax
            # But we want: Dmin <= [u[0]-uk; u[1]-u[0]; ...] <= Dmax
            # However, Lamb[0:ninput, :] * u = u[0] (first ninput elements), not u[0] - uk
            # So we need to adjust bounds: Dmin <= u[0] - uk <= Dmax
            # Which means: Dmin + uk <= u[0] <= Dmax + uk for first ninput constraints
            # For subsequent constraints, Lamb[i, :] * u = u[i] - u[i-1], so bounds are just Dmin and Dmax
            bl = np.full(self.N * self.ninput, -np.inf) if self.Dmin is None else self.Dmin.copy()
            bu = np.full(self.N * self.ninput, np.inf) if self.Dmax is None else self.Dmax.copy()
            # Adjust first time step bounds to account for uk
            if self.Dmin is not None:
                bl[0:self.ninput] = self.Dmin[0:self.ninput] + self.uk
            if self.Dmax is not None:
                bu[0:self.ninput] = self.Dmax[0:self.ninput] + self.uk
        else:
            bl = None
            bu = None

        # Solve unconstrained QP: minimize 0.5*u^T*H*u + g^T*u
        # No constraints: l=-inf, u=inf, A=0 (no equality/inequality constraints)
        
        ufin, info = qpsolver(H, g, l, u, Lamb, bl, bu, u_init)

        # Reshape solution: from (N*nu,) to (nu x N) matrix
        # Each column is the input at one time step
        return ufin.reshape(-1, self.ninput).T, info["f"] + rho
    

    def MPC_output_constraints(self):
        pass

    def MPC_qp(self):
        if self.Rmax is not None and self.Rmin is not None:
            return self.MPC_output_constraints()
        else:
            return self.MPC_input_constraints()
    
    def KalmanFilterUpdate(self, zk):
        """
        Update state estimate using Kalman filter.
        
        Parameters:
        -----------
        zk : numpy.ndarray
            Current measurement (ny x 1)
            
        Returns:
        --------
        xk_new : numpy.ndarray
            Updated state estimate (nx x 1)
        """
        xk_new, Pk_new = KalmanFilterUpdate(
            xt=self.xk,      # Current state estimate
            dt=self.wk,      # Current disturbance estimate
            ut=self.uk,      # Current input
            yt=zk,           # Current measurement
            A=self.A,        # State transition matrix
            B=self.B,        # Input matrix
            C=self.C,        # Output matrix
            P=self.Pk,       # Current covariance
            Q=self.Q,        # Process noise covariance
            R=self.R,        # Measurement noise covariance
        )
        self.Pk = Pk_new  # Update covariance for next iteration
        return xk_new

    def update(self, zk):
        """
        Main update function: performs one MPC step.
        
        This function:
        1. Updates state estimate using Kalman filter
        2. Solves MPC optimization problem
        3. Applies first input from optimal sequence
        
        Parameters:
        -----------
        zk : numpy.ndarray
            Current measurement (ny x 1)
            
        Returns:
        --------
        uk : numpy.ndarray
            Optimal input to apply (nu x 1)
        """
        # Update state estimate with new measurement
        self.xk = self.KalmanFilterUpdate(zk)
        
        # Solve MPC optimization problem
        ufin, _ = self.MPC_qp()
        
        # Apply first input from optimal sequence (receding horizon)
        self.uk = ufin[:, 0]
        return self.uk