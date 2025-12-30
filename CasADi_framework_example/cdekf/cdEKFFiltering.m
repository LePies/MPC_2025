function [ xkk, Pkk ] = cdEKFFiltering( tk, yk, xkkm1, Pkkm1, ...
                                        measurementFun, parametersModel )
%--------------------------------------------------------------------------
% Continuous-discrete EKF measurement update (robust)
% - Regularizes innovation covariance
% - Uses Joseph stabilized covariance update
% - Adds guards against NaN/Inf and loss of PSD
%--------------------------------------------------------------------------

%% Unpack
nx = length(xkkm1);
R  = parametersModel.R;
I  = eye(nx);

% Basic sanity checks (prevents silent NaN propagation)
if any(~isfinite(xkkm1)) || any(~isfinite(Pkkm1(:))) || any(~isfinite(yk))
    warning('cdEKFFiltering:NonFiniteInput', ...
            'Non-finite input detected. Skipping measurement update.');
    xkk = xkkm1;
    Pkk = Pkkm1;
    return;
end

%% Measurement prediction
[ ykkm1, Ck ] = measurementFun( tk, xkkm1, parametersModel );

if any(~isfinite(ykkm1)) || any(~isfinite(Ck(:)))
    warning('cdEKFFiltering:NonFiniteMeasurementModel', ...
            'Non-finite measurement model output. Skipping update.');
    xkk = xkkm1;
    Pkk = Pkkm1;
    return;
end

%% Innovation
ek = yk - ykkm1;

%% Innovation covariance
Sk = Ck*Pkkm1*Ck' + R;

% Adaptive regularization: increase jitter until Sk is usable
epsS = 1e-9;
maxTries = 8;
tries = 0;

while (any(~isfinite(Sk(:))) || rcond(Sk) < 1e-12) && tries < maxTries
    Sk = Ck*Pkkm1*Ck' + R + epsS*eye(size(Sk));
    epsS = epsS * 10;
    tries = tries + 1;
end

if any(~isfinite(Sk(:))) || rcond(Sk) < 1e-14
    warning('cdEKFFiltering:SingularSk', ...
            'Innovation covariance ill-conditioned even after regularization. Skipping update.');
    xkk = xkkm1;
    Pkk = Pkkm1;
    return;
end

%% Kalman gain (solve, no inverse)
% K = P*C' / S  is equivalent to  (S'\(C*P') )' but / is fine once Sk is regular
Kk = (Pkkm1*Ck') / Sk;

if any(~isfinite(Kk(:)))
    warning('cdEKFFiltering:NonFiniteGain', ...
            'Kalman gain became non-finite. Skipping update.');
    xkk = xkkm1;
    Pkk = Pkkm1;
    return;
end

%% State update
xkk = xkkm1 + Kk*ek;

% Optional: prevent clearly unphysical negative masses from destabilizing drift sqrt()
% Uncomment if you keep seeing sqrt/NaN issues:
% xkk(1:4) = max(xkk(1:4), 0);

%% Covariance update (Joseph form)
Pkk = (I - Kk*Ck)*Pkkm1*(I - Kk*Ck)' + Kk*R*Kk';

% Enforce symmetry
Pkk = 0.5*(Pkk + Pkk');

% PSD guard: if Pkk lost PSD due to numerics, project to nearest PSD (small floor)
[eV, eD] = eig(Pkk);
d = diag(eD);
if any(~isfinite(d))
    warning('cdEKFFiltering:NonFiniteCov', ...
            'Covariance became non-finite. Reverting to prior covariance.');
    xkk = xkkm1;
    Pkk = Pkkm1;
    return;
end
d(d < 1e-12) = 1e-12;
Pkk = eV*diag(d)*eV';
Pkk = 0.5*(Pkk + Pkk');

end
