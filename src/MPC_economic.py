import numpy as np
from src.MPC import MPC

class MPC_economic(MPC):
    def __init__(self, cost_array, H, U, h_op, **kwargs):
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
            
            # Default parameters
            N = kwargs.pop("N", 30)
            
            # Default reference trajectories
            U_bar = np.ones((N, 2)) * 0
            R_bar = np.ones((N, 2)) * 0

            u_op = kwargs.pop("u_op", np.array([250, 325]))
            
            R_noise = np.eye(2) # Measurement noise covariance
            
            super().__init__(
                N=N,
                us=u_op,
                hs=h_op,
                U_bar=U_bar,
                R_bar=R_bar,
                A=Ad,
                B=Bd,
                C=Cz,
                Q=Q,
                R=R_noise,
                E=E,
                problem="Problem 4",
                **kwargs
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
        self.Rmax = np.array([H, H])
        self.Umin = np.array([0, 0])
        self.Umax = np.array([U, U])
        self.k = 0
        self.Wu = np.diag(self.cost_array)
        self._compute_Ws()

    def _compute_Ws(self):
        n_array = np.array([1/n for n in range(1, self.N)])
        self.Ws1 = 10*np.diag(n_array)
        self.Ws2 = 1e3*np.diag(n_array)
