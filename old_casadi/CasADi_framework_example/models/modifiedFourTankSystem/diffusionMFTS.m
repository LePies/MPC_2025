function [ sigma ] = diffusionMFTS( t, x, u, d, theta )
%--------------------------------------------------------------------------
%   Author(s):
%       Morten Ryberg Wahlgreen
%
%   Email:
%       morwa@dtu.dk
%
%--------------------------------------------------------------------------
%   Call:
%       [ f ] = diffusionMFTS( t, x, u, d, theta )
%
%   Description:
%       evalutates a diffusion term for the modified four tank system 
%       (MFTS), which models variations in the inlet flow streams.
%
%   Inputs:
%       t       :     time
%       x       :     masses of tank contents
%       u       :     manipulated liquid in-flows
%       d       :     disturbance in-flows
%       theta   :     parameters
%
%   Outputs:
%       sigma   :     diffusion term
%
%--------------------------------------------------------------------------

%% Diffusion Function

% Multiplicative noise on inlet flows
sigma = 20*[   0.0      , 0.0	                    ;   
                    0.0                  , 0.0            ;    
                    0.0                  ,0.0    ;  
                  0.0, 0.0                     ;
                  1,0.0;
                  0.0,1];     
% ...

end