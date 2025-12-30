function [ f, dfdx ] = driftMFTS( t, x, u, d, theta )

% States
m  = x(1:4);     % masses
F3 = x(5);       % disturbance state 1
F4 = x(6);       % disturbance state 2

% Inputs
F = u(1:2);

% Disturbance means (these are the "bars" in your notes)
F3bar = d(1);
F4bar = d(2);

% Parameters
a     = theta.a;
A     = theta.A;
gamma = theta.gamma;
g     = theta.g;
rho   = theta.rho;

% --- Mass subsystem ---
h    = m./(rho*A);
qOut = a.*sqrt(2*g*h);

qIn = [          gamma(1)*F(1) + qOut(3) ;
                 gamma(2)*F(2) + qOut(4) ;
           (1-gamma(2))*F(2) + F3       ;
           (1-gamma(1))*F(1) + F4       ];

fs = rho.*(qIn - qOut);  % 4x1

% --- Disturbance subsystem (mean reversion) ---
ad = theta.ad;           % add this parameter
fd = [ ad*(F3bar - F3) ;
       ad*(F4bar - F4) ]; % 2x1

% Full drift (6x1)
f = [fs; fd];

% Jacobian if requested
if nargout > 1
    dfdx = full(theta.dfdxFun(t, x, u, d));
end
end
