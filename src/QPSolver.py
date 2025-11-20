import casadi as ca
import numpy as np
import sys
import os
from contextlib import contextmanager


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def qpsolver(H, g, l = None, u = None, A = None, bl = None, bu = None, xinit = None):
    """
    Solve a Quadratic Programming (QP) problem using CasADi.

    The QP problem is formulated as:
    minimize: (1/2) * x^T * H * x + g^T * x
    subject to: l <= x <= u
                bl <= A * x <= bu

    Parameters:
    -----------
    H : numpy.ndarray
        Hessian matrix (n x n) for the quadratic term
    g : numpy.ndarray
        Gradient vector (n x 1) for the linear term
    l : numpy.ndarray
        Lower bounds on variables (n x 1)
    u : numpy.ndarray
        Upper bounds on variables (n x 1)
    A : numpy.ndarray
        Constraint matrix (m x n)
    bl : numpy.ndarray
        Lower bounds on constraints (m x 1)
    bu : numpy.ndarray
        Upper bounds on constraints (m x 1)
    xinit : numpy.ndarray
        Initial guess for the solution (n x 1)

    Returns:
    --------
    x : numpy.ndarray
        Optimal solution vector (n x 1)
    info : dict
        Dictionary containing solver information:
        - 'f': Objective value at solution
        - 'lam_x': Dual variables for variable bounds
        - 'lam_g': Dual variables for constraint bounds
        - 'success': Boolean indicating if solver succeeded
        - 'stats': Solver statistics
    """
    if l is None:
        l = -np.inf*np.ones(H.shape[0])
    if u is None:
        u = np.inf*np.ones(H.shape[0])
    if A is None:
        A = np.zeros((0, H.shape[0]))
    if bl is None:
        bl = -np.inf*np.ones(A.shape[0])
    if bu is None:
        bu = np.inf*np.ones(A.shape[0])
    if xinit is None:
        xinit = np.zeros(H.shape[0])

    # Convert inputs to numpy arrays if needed
    H = np.array(H)
    g = np.array(g).flatten()
    lower_bound = np.array(l).flatten()  # Renamed to avoid ambiguity
    u = np.array(u).flatten()
    A = np.array(A)
    bl = np.array(bl).flatten()
    bu = np.array(bu).flatten()
    xinit = np.array(xinit).flatten()

    # Validate inputs
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("Hessian matrix H contains NaN or Inf values")
    if np.any(np.isnan(g)) or np.any(np.isinf(g)):
        raise ValueError("Gradient vector g contains NaN or Inf values")
    
    # Check if Hessian is positive definite (required for QP)
    try:
        eigenvals = np.linalg.eigvals(H)
        if np.any(eigenvals <= 0):
            # Make Hessian positive definite by adding small regularization
            min_eigenval = np.min(eigenvals)
            if min_eigenval <= 0:
                H = H + (abs(min_eigenval) + 1e-6) * np.eye(H.shape[0])
    except (np.linalg.LinAlgError, ValueError):
        pass  # If eigendecomposition fails, proceed anyway

    # Get dimensions
    n = H.shape[0]  # number of variables
    m = A.shape[0] if A.size > 0 else 0  # number of constraints
    
    # Validate constraint dimensions
    if m > 0:
        if bl.shape[0] != m or bu.shape[0] != m:
            raise ValueError(f"Constraint bounds dimension mismatch: A has {m} rows, but bl/bu have {bl.shape[0]}/{bu.shape[0]} elements")
        if A.shape[1] != n:
            raise ValueError(f"Constraint matrix dimension mismatch: A is {A.shape}, but H is {H.shape}")
        # Check for infeasible constraints
        if np.any(bl > bu):
            raise ValueError("Infeasible constraints: some lower bounds are greater than upper bounds")

    # Convert matrices to CasADi format
    H_ca = ca.DM(H)
    g_ca = ca.DM(g)
    if m > 0:
        A_ca = ca.DM(A)

    # Create symbolic variables
    x = ca.MX.sym('x', n)

    # Define the QP problem
    # Objective: (1/2) * x^T * H * x + g^T * x
    obj = 0.5 * ca.mtimes(ca.mtimes(x.T, H_ca), x) + ca.mtimes(g_ca.T, x)

    # Constraints: bl <= A * x <= bu
    if m > 0:
        g_constraints = ca.mtimes(A_ca, x)
    else:
        # No constraints case
        g_constraints = ca.MX()

    # Create QP structure
    qp = {
        'x': x,
        'f': obj,
        'g': g_constraints
    }

    # Prepare bounds
    # Variable bounds: l <= x <= u
    lbx = lower_bound
    ubx = u

    # Constraint bounds: bl <= A*x <= bu
    if m > 0:
        lbg = bl
        ubg = bu
    else:
        lbg = ca.DM()
        ubg = ca.DM()

    # Create QP solver and solve (suppress any output including license info)
    # Using 'qpoases' solver (other options: 'osqp', 'hpipm', etc.)
    try:
        with suppress_stdout_stderr():
            solver = ca.qpsol('solver', 'qpoases', qp, {'error_on_fail': False})
            sol = solver(
                x0=xinit,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg
            )
            
            # Check if solver succeeded
            if not solver.stats()['success']:
                # Try with OSQP as fallback if qpoases fails
                solver_osqp = ca.qpsol('solver_osqp', 'osqp', qp)
                sol = solver_osqp(
                    x0=xinit,
                    lbx=lbx,
                    ubx=ubx,
                    lbg=lbg,
                    ubg=ubg
                )
    except Exception as e:
        # If solver fails, provide more informative error
        error_msg = f"QP solver failed: {str(e)}\n"
        error_msg += f"Problem dimensions: n={n}, m={m}\n"
        error_msg += f"H shape: {H.shape}, H condition number: {np.linalg.cond(H):.2e}\n"
        if m > 0:
            error_msg += f"A shape: {A.shape}\n"
            error_msg += f"bl range: [{np.min(bl):.2e}, {np.max(bl):.2e}]\n"
            error_msg += f"bu range: [{np.min(bu):.2e}, {np.max(bu):.2e}]\n"
        raise RuntimeError(error_msg) from e

    # Extract solution
    x_opt = np.array(sol['x']).flatten()

    # Extract solver information
    info = {
        'f': float(sol['f']),  # Objective value
        # Dual variables for bounds
        'lam_x': (np.array(sol['lam_x']).flatten()
                  if 'lam_x' in sol else None),
        # Dual variables for constraints
        'lam_g': (np.array(sol['lam_g']).flatten()
                  if 'lam_g' in sol else None),
        'success': True,  # CasADi solvers typically don't fail silently
        'stats': solver.stats() if hasattr(solver, 'stats') else {}
    }

    return x_opt, info


if __name__ == "__main__":
    H = np.array([[2, 0], [0, 2]])
    g = np.array([-2, -5])
    l = np.array([0, 0])
    u = np.array([1.5, 2])
    A = np.array([[1, 1]])
    bl = np.array([-np.inf])
    bu = np.array([2.5])
    xinit = np.array([0, 0])
    for _ in range(10):
        x, info = qpsolver(H, g, l, u, A, bl, bu, xinit)
    print("Solution x:", x)
