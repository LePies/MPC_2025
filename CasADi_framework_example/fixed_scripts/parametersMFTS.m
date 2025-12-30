function [ theta ] = parametersMFTS()

%% Sizes
theta.nx = 6;
theta.nu = 2;
theta.nd = 2;
theta.nw = 2;
theta.ny = 2;
theta.nz = 2;

%% Physical parameters
theta.a = 1.2272 * ones(4,1);
theta.A = 380.1327 * ones(4,1);
theta.gamma = [0.45; 0.40];
theta.g   = 981;
theta.rho = 1.0;

%% Disturbance dynamics
theta.ad = 0.02;

%% Diffusion intensity (std scaling) for disturbance states
theta.sigmaD = 1.0;

%% Measurement noise covariance (must be PD)
theta.R = 1e-4 * eye(theta.ny);

%% (Optional) Wiener intensity (if you use Gn*Qw*Gn')
theta.Qw = eye(theta.nw);

%% CasADi derivatives
import casadi.*
tCas = MX.sym('tCas', 1, 1);
xCas = MX.sym('xCas', theta.nx, 1);
uCas = MX.sym('uCas', theta.nu, 1);
dCas = MX.sym('dCas', theta.nd, 1);

% Drift and Jacobian
fExp = driftMFTS(tCas, xCas, uCas, dCas, theta);
dfdxTmp = jacobian(fExp, xCas);
theta.dfdxFun = Function('dfdxFun', {tCas, xCas, uCas, dCas}, {dfdxTmp});

% Measurement and Jacobian
yExp = measurementMFTS(tCas, xCas, theta);     % returns y only when called with 1 output
dydxTmp = jacobian(yExp, xCas);
theta.dydxFun = Function('dydxFun', {tCas, xCas, uCas, dCas}, {dydxTmp});

end
