Model 3 (SDE with augmented disturbance states) - Patched MATLAB files

What changed:
- driftMFTS.m: deterministic 6x1 drift for x=[m1..m4,F3,F4]
- diffusionMFTS.m: 6x2 diffusion with noise only on F3,F4
- driverMFTSCasadiCollocationClosedLoop.m: x0 fixed to 6x1 (removed duplicated d0)
- parametersMFTS.m: nx=6, nw=2, nd=2 template and noise settings
- eulerMaruyamaStep.m: helper for SDE simulation (Euler–Maruyama)

Important:
- If your project builds CasADi Jacobians (theta.dfdxFun), regenerate them for nx=6.
- NMPC/collocation should use drift only; stochasticity enters via simulation and EKF covariance (Sigma*Qw*Sigma').
