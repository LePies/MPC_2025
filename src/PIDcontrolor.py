import numpy as np

class PIDController:
    """
    Basic PID controller for SISO or MIMO systems.

    Usage:
        pid = PIDController(Kp, Ki, Kd, setpoint, dt, u_min=None, u_max=None)
        u = pid.update(measurement)
    """
    def __init__(self, Kp, Ki, Kd, setpoint, dt, u_min=None, u_max=None):
        """
        Args:
            Kp: Proportional gain (scalar or array)
            Ki: Integral gain (scalar or array)
            Kd: Derivative gain (scalar or array)
            setpoint: Desired setpoint (scalar or array)
            dt: Time step [s]
            u_min: Minimum control output (scalar or array, optional)
            u_max: Maximum control output (scalar or array, optional)
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.setpoint = np.array(setpoint)
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

        self.integral = np.zeros_like(self.setpoint, dtype=float)
        self.prev_error = np.zeros_like(self.setpoint, dtype=float)

    def update(self, measurement):
        """
        Compute PID control action.

        Args:
            measurement: Current process variable (scalar or array)

        Returns:
            u: Control output (same shape as setpoint)
        """
        error = self.setpoint - np.array(measurement)
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Anti-windup: clamp output and integral if needed
        if self.u_min is not None or self.u_max is not None:
            u_clamped = np.copy(u)
            if self.u_min is not None:
                u_clamped = np.maximum(u_clamped, self.u_min)
            if self.u_max is not None:
                u_clamped = np.minimum(u_clamped, self.u_max)
            # Optional: prevent integral windup
            # Only integrate if not saturated
            for i in range(np.size(u)):
                if (self.u_min is not None and u[i] <= self.u_min) or (self.u_max is not None and u[i] >= self.u_max):
                    self.integral.flat[i] -= error.flat[i] * self.dt  # Undo last integration step
            u = u_clamped

        self.prev_error = error
        return u

    def reset(self):
        """Reset the integral and previous error."""
        self.integral = np.zeros_like(self.setpoint, dtype=float)
        self.prev_error = np.zeros_like(self.setpoint, dtype=float)
