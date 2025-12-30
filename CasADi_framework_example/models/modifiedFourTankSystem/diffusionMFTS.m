function [ sigma ] = diffusionMFTS( t, x, u, d, theta )
% Model 3: noise only on disturbance states (x5,x6)

if isfield(theta,'sigmaD')
    s = theta.sigmaD;
elseif isfield(theta,'sigma_d')
    s = theta.sigma_d;
else
    s = 1; % default std scaling
end

if isscalar(s)
    S = s*eye(2);
else
    S = diag(s(:));
end

sigma = [ zeros(4,2);
          S ];

end
