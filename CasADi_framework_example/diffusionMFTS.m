function [ sigma ] = diffusionMFTS( t, x, u, d, theta )
%--------------------------------------------------------------------------
% Diffusion term for Modified Four Tank System (Model 3)
%
% Only disturbance states (x5,x6) are driven by Wiener noise.
% sigma is (nx x nw) = 6 x 2
%--------------------------------------------------------------------------

% Noise intensities (optional)
if isfield(theta,'sigmaD')
    s = theta.sigmaD;
    if isscalar(s)
        S = s*eye(2);
    else
        S = diag(s(:));
    end
elseif isfield(theta,'sigma_d')
    s = theta.sigma_d;
    if isscalar(s)
        S = s*eye(2);
    else
        S = diag(s(:));
    end
else
    S = eye(2);
end

sigma = [ zeros(4,2) ;
          S         ];

end
