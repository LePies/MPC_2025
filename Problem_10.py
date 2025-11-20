import numpy as np
from src.MPC import MPC
from src.FourTankSystem import FourTankSystem
from params.initialize import initialize
import matplotlib.pyplot as plt
import sys

def F3_func(t):
    if t < 250:
        return 100
    else:
        return 50
    return 100


def F4_func(t):
    if t < 250:
        return 120
    else:
        return 50
    return 120


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problem = "Problem " + sys.argv[1]
    else:
        problem = "Problem 5"
    
    print(f"Simulating: {problem}")
    
    x0, us, ds, p, R, R_d, delta_t = initialize()
    Model_Stochastic = FourTankSystem(R_s=R, R_d=R_d*0, p=p, delta_t=delta_t, F3=F3_func, F4=F4_func)
    x0 = np.concatenate((x0, ds))
    xs = Model_Stochastic.GetSteadyState(x0, us)
    data_prob5 = np.load(r"Results\Problem5\Problem_5_estimates.npz")
    Q = data_prob5["Q"]
    
    if problem == "Problem 5":
        data = data_prob5
        Ad, Bd, Ed, C, Cz = Model_Stochastic.LinearizeDiscreteTime(xs, ds, delta_t)
        G = np.block([
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.eye(2),
        ]).T
    elif problem == "Problem 4":
        data = np.load(r"Results\Problem4\Problem_4_estimates.npz")
        Ad = data["A"]
        Bd = data["B"]
        Cz = data["C"]
        G = np.block([
            np.zeros((2, 2)),
            np.eye(2),
        ]).T
        Q = Q[:4, :4]
    else:
        raise ValueError("Invalid problem")
    

    # Reference setpoint for controlled outputs (Tank 1 and Tank 2 heights)
    good_goal = np.array([111.05, 100.0])  # Height of Tank 1 and 2 [cm]

    # MPC parameters
    N_mpc = 15  # Prediction horizon - longer horizon improves long-term prediction
    N_t = 500   # Simulation time steps

    # Input reference: expected steady-state inputs for the reference setpoint
    U_bar = np.ones((N_mpc, 2)) * np.array([280, 300])  # [cm³/s]
    R_bar = np.ones((N_mpc, 2)) * good_goal  # Output reference trajectory

    # ============================================================================
    # INPUT CONSTRAINTS
    # ============================================================================
    Umin = np.array([0, 0])        # Minimum input flow [cm³/s] - physical limit
    Umax = np.array([3000, 3000])  # Maximum input flow [cm³/s] - physical limit
    Dmin = np.array([-5, -10])     # Minimum input rate [cm³/s] - smoothness constraint
    Dmax = np.array([10, 5])        # Maximum input rate [cm³/s] - smoothness constraint

    # ============================================================================
    # OUTPUT CONSTRAINTS (with slack variables for soft constraints)
    # ============================================================================
    # Design rationale:
    # - Constraints should allow normal operation around setpoint with safety margins
    # - Too tight: may cause infeasibility or excessive slack variable usage
    # - Too loose: constraints become ineffective
    # - Asymmetric constraints can account for different dynamics (e.g., faster rise vs fall)
    
    # Define constraint margins around setpoint [cm]
    # IMPORTANT: If constraints are too loose, the system never hits them,
    # slack variables stay zero, and behavior = Problem_9 (no constraints)
    # Tighten constraints to see the effect of output constraints
    
    # IMPORTANT: For constraints to have an effect, they must be TIGHTER than
    # the natural system response. If the setpoint is within bounds and tracking
    # is prioritized, slack variables stay zero = same as Problem_9.
    
    # Option 1: VERY tight constraints (will force different behavior)
    # These are tighter than typical response, so constraints will be active
    margin_low = np.array([10.0, 10.0])    # Very tight lower margin
    margin_high = np.array([10.0, 10.0])   # Very tight upper margin
    
    # Option 2: Moderate constraints (may or may not be active)
    # margin_low = np.array([5.0, 5.0])
    # margin_high = np.array([5.0, 5.0])
    
    # Option 3: Loose constraints (will match Problem_9 - constraints never hit)
    # margin_low = np.array([50.0, 50.0])
    # margin_high = np.array([50.0, 50.0])
    
    Rmin = good_goal - margin_low  # Lower output bounds [cm]
    Rmax = good_goal + margin_high  # Upper output bounds [cm]
    
    print("Output constraints:")
    print(f"  Setpoint:                 {good_goal} [cm]")
    print(f"  Constraint range: Tank 1: [{Rmin[0]:.2f}, {Rmax[0]:.2f}], "
          f"Tank 2: [{Rmin[1]:.2f}, {Rmax[1]:.2f}]")

    # ============================================================================
    # SLACK VARIABLE WEIGHTS
    # ============================================================================
    # Design rationale:
    # - Ws2, Wt2: Quadratic penalty - strongly penalizes constraint violations
    #   High values ensure constraints are respected when possible
    # - Ws1, Wt1: Linear penalty - provides soft constraint behavior
    #   Lower values allow small violations when necessary (e.g., during transients)
    # - Ratio Ws2/Ws1 determines how "hard" vs "soft" the constraint is
    
    slack_quad_weight = 1000.0  # Quadratic penalty - high to strongly discourage violations
    slack_lin_weight = 10.0      # Linear penalty - lower for soft constraint behavior
    
    Ws2 = np.eye(2) * slack_quad_weight  # Quadratic penalty on lower bound slack (s)
    Wt2 = np.eye(2) * slack_quad_weight  # Quadratic penalty on upper bound slack (t)
    Ws1 = np.eye(2) * slack_lin_weight   # Linear penalty on lower bound slack (s)
    Wt1 = np.eye(2) * slack_lin_weight   # Linear penalty on upper bound slack (t)

    # ============================================================================
    # TRACKING AND CONTROL WEIGHTS
    # ============================================================================
    # Wz: Output tracking weight - prioritizes reaching setpoint
    # Wu: Input deviation weight - penalizes deviation from input reference
    # Wdu: Input rate weight - penalizes rapid input changes (smoothness)
    
    Wz = np.eye(2) * 2.0      # Strong tracking priority
    Wu = np.eye(2) * 1e-3      # Low penalty - allow sufficient control effort
    Wdu = np.eye(2) * 0.5      # Moderate penalty - smooth but responsive

    # ============================================================================
    # MPC CONTROLLER INITIALIZATION
    # ============================================================================
    # Enable debug mode to check slack variable usage
    mpc_controller = MPC(
        N=N_mpc, 
        u0=np.ones(2),
        x0=xs,
        w0=ds,
        U_bar=U_bar,
        R_bar=R_bar,
        A=Ad, 
        B=Bd, 
        C=Cz, 
        G=G,
        Q=Q,
        R=R_d,
        problem=problem, 
        Wz=Wz,
        Wu=Wu,
        Wdu=Wdu,
        Umin=Umin,
        Umax=Umax,
        Dmin=Dmin,
        Dmax=Dmax,
        Rmax=Rmax,
        Rmin=Rmin,
        Ws2=Ws2,
        Wt2=Wt2,
        Ws1=Ws1,
        Wt1=Wt1,
    )
    
    # Enable debug mode to track slack variable usage
    mpc_controller._debug_slack = True

    # Run closed-loop simulation
    t, x, u, d, h = Model_Stochastic.ClosedLoop(np.array([0, N_t]), xs, mpc_controller)

    # ============================================================================
    # PLOTTING
    # ============================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Tank heights with constraints
    axes[0].plot(t, h[0, :], label='Height of Tank 1', color='dodgerblue', linewidth=2)
    axes[0].plot(t, h[1, :], label='Height of Tank 2', color='tomato', linewidth=2)
    axes[0].plot(t, h[2, :], label='Height of Tank 3', color='limegreen', linewidth=1.5, alpha=0.7)
    axes[0].plot(t, h[3, :], label='Height of Tank 4', color='orange', linewidth=1.5, alpha=0.7)
    
    # Plot setpoints
    axes[0].plot(t, t*0 + R_bar[0, 0], label='Setpoint Tank 1', 
                 color='dodgerblue', ls='--', linewidth=2)
    axes[0].plot(t, t*0 + R_bar[0, 1], label='Setpoint Tank 2', 
                 color='tomato', ls='--', linewidth=2)
    
    # Plot output constraints (only for controlled outputs - Tank 1 and 2)
    axes[0].fill_between(t, Rmin[0], Rmax[0], alpha=0.15, color='dodgerblue', 
                         label='Constraints Tank 1')
    axes[0].fill_between(t, Rmin[1], Rmax[1], alpha=0.15, color='tomato', 
                         label='Constraints Tank 2')
    axes[0].axhline(y=Rmin[0], color='dodgerblue', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=Rmax[0], color='dodgerblue', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=Rmin[1], color='tomato', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=Rmax[1], color='tomato', linestyle=':', linewidth=1.5, alpha=0.7)
    
    axes[0].legend(loc='best', ncol=2, fontsize=9)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Height [cm]')
    axes[0].set_title('Tank Heights with Output Constraints')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Input flows
    axes[1].plot(t, u[0, :], label='Flow Tank 1 (u₁)', color='dodgerblue', linewidth=2)
    axes[1].plot(t, u[1, :], label='Flow Tank 2 (u₂)', color='tomato', linewidth=2)
    
    # Plot input constraints
    axes[1].axhline(y=Umax[0], color='dodgerblue', linestyle=':', linewidth=1.5, 
                    alpha=0.5, label='Input limits')
    axes[1].axhline(y=Umin[0], color='dodgerblue', linestyle=':', linewidth=1.5, alpha=0.5)
    
    axes[1].legend(loc='best')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Flow [cm³/s]')
    axes[1].set_title('Control Inputs')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('figures/Problem10/Problem_10_Heights.png', dpi=150, bbox_inches='tight')
    plt.close()
