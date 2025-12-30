function [ f, dfdx ] = driftMFTS( t, x, u, d, theta )
%--------------------------------------------------------------------------
% Drift term for Modified Four Tank System (Model 3: augmented with disturbance states)
%
% State: x = [m1 m2 m3 m4 F3 F4]'  (masses + disturbance inflow states)
% Input: u = [F1 F2]'
% Disturbance parameter: d = [F3bar F4bar]' (mean/nominal disturbance inflows)
%
% NOTE:
%   - Must be deterministic for CasADi / NMPC / EKF linearization.
%   - Stochasticity enters via diffusionMFTS + Euler-Maruyama in simulation,
%     and via Sigma*Sigma' in the CDEKF covariance ODE.
%--------------------------------------------------------------------------

%% Un-pack Parameters and Variables

% States
m  = x(1:4);     % masses in tanks
F3 = x(5);       % disturbance inflow state to tank 3
F4 = x(6);       % disturbance inflow state to tank 4

% Inputs
F  = u(1:2);

% Disturbance "means"/setpoints (for mean-reversion dynamics)
F3bar = d(1);
F4bar = d(2);

% Parameters
a      = theta.a;       % outlet areas (4x1)
A      = theta.A;       % tank areas   (4x1)
gamma  = theta.gamma;   % split ratios (2x1)
g      = theta.g;
rho    = theta.rho;

% Mean reversion rates for disturbance states (scalar or 2x1)
if isfield(theta,'ad')
    ad = theta.ad;
elseif isfield(theta,'a_d')
    ad = theta.a_d;
elseif isfield(theta,'a_f')
    ad = theta.a_f;
else
    ad = 0.05; % fallback
end
if isscalar(ad)
    ad3 = ad; ad4 = ad;
else
    ad3 = ad(1); ad4 = ad(2);
end

%% Mass subsystem drift

% heights from masses
h    = m./( rho*A );

% outflows
qOut = a.*sqrt( 2*g*h );

% inflows (disturbance inflows are states F3,F4)
qIn = [          gamma(1) *F(1) + qOut(3) ;     % tank 1
                 gamma(2) *F(2) + qOut(4) ;     % tank 2
           ( 1 - gamma(2) )*F(2) + F3     ;     % tank 3
           ( 1 - gamma(1) )*F(1) + F4     ];    % tank 4

% mass derivatives
fs = rho.*( qIn - qOut );   % 4x1

%% Disturbance-state subsystem drift (mean reversion)
fd = [ ad3*(F3bar - F3) ;
       ad4*(F4bar - F4) ];  % 2x1

%% Full augmented drift (6x1)
f = [ fs ; fd ];

%% x-derivative
if nargout > 1
    % If theta provides a CasADi jacobian function, use it
    if isfield(theta,'dfdxFun')
        dfdx = full(theta.dfdxFun( t, x, u, d ));
    else
        dfdx = [];
    end
end

end
