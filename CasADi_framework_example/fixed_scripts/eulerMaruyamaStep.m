function xNext = eulerMaruyamaStep(t, x, u, d, dt, driftFun, diffusionFun, theta)
%--------------------------------------------------------------------------
% Single Euler-Maruyama step for SDE:
%   dx = f(x,u,d)*dt + Sigma(x,u,d)*dW
% where dW ~ N(0, dt*I)
%--------------------------------------------------------------------------

f = driftFun(t, x, u, d, theta);
Sigma = diffusionFun(t, x, u, d, theta);

nw = size(Sigma,2);
dW = sqrt(dt)*randn(nw,1);

xNext = x + f*dt + Sigma*dW;

end
