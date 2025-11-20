import numpy as np
from src.QPSolver import qpsolver
from src.KalmanFilterUpdate import KalmanFilterUpdate

class MPC:
    def __init__(self, u0, x0, w0,N, U_bar, R_bar, G, A, B, C, Q, R, problem="Problem 5", Wz=None, Wu=None, Wdu=None):
        self.N = N
        self.U_bar = U_bar.flatten()
        self.R_bar = R_bar.flatten()
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
        self.noutput = C.shape[0]
        self.uk = u0
        
        self.Gamma = self.get_gamma()
        self.phi_x = self.get_phi_x()
        self.phi_w = self.get_phi_w()
        self.Pk = self.get_Pk()

    def get_Pk(self):
        Pk = 5*np.eye(self.A.shape[0])
        return Pk

    def reset_control(self):
        self.Pk = self.get_Pk()

    def get_phi_x(self):
        phix = np.block([(self.C@self.A**i).T for i in range(1,self.N+1)])
        return phix
    
    def get_phi_w(self):
        phiw = np.block([(self.C@(self.A**i)@self.G).T for i in range(self.N)])
        return phiw
        
    def get_gamma(self):
        """
        Get the Gamma matrix for the given problem and number of steps

        Parameters:
            N: number of steps
            problem: "Problem 5" or "Problem 4"
        Returns:
            Gamma: Gamma matrix
        """
        if self.problem == "Problem 5":
            data = np.load("Results/Problem5/Problem_5_estimates.npz")
        elif self.problem == "Problem 4":
            data = np.load("Results/Problem4/Problem_4_estimates.npz")
        else:
            raise ValueError("Invalid problem")

        markov_mat = data["markov_mat"]

        markov_mat_test = markov_mat[:, :, :self.N].reshape(2, -1)

        Gamma = np.zeros((2*self.N, 2*self.N))
        
        for i in range(self.N):
            markov_input = markov_mat_test[:, :2*(self.N-i)]
            markov_input = np.block([
                [np.zeros((2, 2*i)), markov_input]
            ])
            Gamma[:, 2*i:2*i+2] = markov_input.T

        return Gamma

    def kron(self, W):
        I = np.eye(self.N)
        Wbar = np.kron(I,W)
        return Wbar

    def MPC_uncontrained(self, zk):

        if self.Wz is None:
            Wz = np.eye(self.noutput)
        else:
            Wz = self.Wz
        if self.Wu is None:
            Wu = np.eye(self.noutput)
        else:
            Wu = self.Wu
        if self.Wdu is None:
            Wdu = np.eye(self.noutput)
        else:
            Wdu = self.Wdu

        Wz_bar = self.kron(Wz)
        Wu_bar = self.kron(Wu)
        Wdu_bar = self.kron(Wdu)
        I0 = np.block([np.eye(self.noutput) if i == 0 else np.zeros((self.noutput, self.noutput)) for i in range(self.N)]).T

        bk = self.phi_x.T@self.xk + self.phi_w.T@self.wk

        ck = self.R_bar - bk

        Lamb = np.eye(self.N*self.noutput)
        Lamb += -1*np.eye(self.N*self.noutput,k=-1)

        Hu = Wu_bar.T@Wu_bar

        Hdu = Lamb.T@Wdu_bar.T@Wdu_bar@Lamb
        Hz = self.Gamma.T@Wz_bar.T@Wz_bar@self.Gamma
        H = Hu + Hdu + Hz

        gu = -Wu_bar.T@Wu_bar@self.U_bar
        gdu = -Lamb.T@Wdu_bar.T@Wdu_bar@I0@self.uk
        gz = -(Wz_bar@self.Gamma).T@Wz_bar@self.Gamma@ck 
        g = gu+gdu+gz

        rhou = 1/2*self.U_bar.T@Wu_bar.T@Wu_bar@self.U_bar
        rhodu = 1/2*(Wdu_bar@I0@self.uk).T@(Wdu_bar@I0@self.uk)
        rhoz = 1/2*ck.T@Wz_bar@Wz_bar@self.Gamma@ck
        rho = rhodu+rhou+rhoz

        u_init = np.zeros(H.shape[0])

        ufin, info = qpsolver(H, g, -np.inf, np.inf, H*0, -np.inf, np.inf, u_init)

        return ufin.reshape(-1, 2).T, info["f"] + rho
    
    def KalmanFilterUpdate(self, zk):
        xk_new, Pk_new = KalmanFilterUpdate(
            xt = self.xk,
            dt = self.wk,
            ut = self.uk,
            yt = zk,
            A = self.A,
            B = self.B,
            C = self.C,
            P = self.Pk,
            Q = self.Q,
            R = self.R,
        )
        self.Pk = Pk_new
        return xk_new

    def update(self, zk):
        self.xk = self.KalmanFilterUpdate(zk)
        ufin, _ = self.MPC_uncontrained(zk)
        self.uk = ufin[:,0]
        return self.uk