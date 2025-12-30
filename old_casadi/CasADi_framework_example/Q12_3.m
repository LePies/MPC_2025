close all
clear
clc

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

addpath(genpath( 'casadiOCPInterface' ));
addpath(genpath( 'models'             ));
addpath(         'util'                );
addpath(         'cdekf'               );

%% Save Plot/Data
savePlot = 0;
name    = 'MFTSCollocationClosedLoop';
figPath = './figures/';

%% Simulation Info
t0    = 0;                      % [s]
tf    = 60*20;                  % [s]
Ts    = 1.0;                    % [s]
Nsim  = 10;                     % EM steps per sample
N     = 30;                     % control horizon
Ns    = (tf-t0)/Ts;             % number of sampling intervals
M     = 10;                     % collocation steps in OCP

% Initial condition
u0 = [250.0; 325.0];
d0 = [100.0; 120.0];
x0 = [21362.92718669; 19317.4122346; 11195.65652256; 8530.21545886; d0];

% Bounds
uMin = [0.0;   0.0];
uMax = [500.0; 500.0];
dMin = [0.0;   0.0];
dMax = [100.0; 100.0];

%% Model
pSim              = parametersMFTS();
driftFunSim       = @driftMFTS;
diffusionFunSim   = @diffusionMFTS;
measurementFunSim = @measurementMFTS;

pCtrl          = parametersMFTS();
driftFun       = @driftMFTS;
diffusionFun   = @diffusionMFTS;
outputFun      = @outputMFTS;
measurementFun = @measurementMFTS;

nx = pSim.nx; nz = pSim.nz; nd = pSim.nd;
nu = pSim.nu; ny = pSim.ny; nw = pSim.nw;

%% Input and Disturbance Initialisation
D = repmat(d0, 1, Ns+N);  % known disturbance trajectory used in sim & EKF

%% Setpoint
zBar = zeros(nz, Ns+N*M);
zBar(1, 1:round(Ns/2) ) = 50.0;
zBar(2, 1:round(Ns/2) ) = 30.0;
zBar(1, round(Ns/2)+1:end ) = 35.0;
zBar(2, round(Ns/2)+1:end ) = 45.0;

%% Solve OCP (kept as in your file; not used in EKF-only test below)
model = @(x,u,d,p) driftFun(0,x,u,d,pCtrl);

X0 = repmat(x0, 1, M*N+1);
U0 = repmat(u0, 1, N);

[opti, x, u, d, x0Cas] = casadiCollocation(model, X0, U0, Ts, M, x0, D(:,1:N), pCtrl);

z       = outputFun(0, x, pCtrl);
zbar    = opti.parameter(nz, size(z,2));
ukm1Cas = opti.parameter(nu, 1);

alpha = 0.995;
phiz  = sum((reshape(z, nz*(M*N+1), 1) - reshape(zbar, nz*(M*N+1), 1)).^2);
phidu = sum((u(:,1) - ukm1Cas).^2) + ...
        sum((reshape(u(:,2:end), nu*(N-1), 1) - reshape(u(:,1:end-1), nu*(N-1), 1)).^2);

opti.minimize(alpha*phiz + (1-alpha)*phidu);
opti.subject_to(uMin(1) <= u(1,:) <= uMax(1));
opti.subject_to(uMin(2) <= u(2,:) <= uMax(2));
opti.subject_to(0.0 <= x(:));

p_opts = struct;
s_opts = struct;
s_opts.acceptable_tol             = 1.0e-7;
s_opts.acceptable_dual_inf_tol    = 1.0e-7;
s_opts.acceptable_constr_viol_tol = 1.0e-7;
s_opts.acceptable_compl_inf_tol   = 1.0e-7;
s_opts.acceptable_obj_change_tol  = 1.0e-7;
s_opts.acceptable_iter            = 10;
s_opts.tol                        = 1.0e-7;
s_opts.dual_inf_tol               = 1.0e-7;
s_opts.constr_viol_tol            = 1.0e-7;
s_opts.max_iter                   = 5000;
s_opts.hessian_approximation      = 'exact';
s_opts.print_level                = 0;
p_opts.print_time                 = 0;

opti.solver('ipopt', p_opts, s_opts);

%% Setup noise
seed = 5;
rng(seed);

Rv = pCtrl.R;
try
    L = chol(Rv,'lower');
catch
    L = zeros(size(Rv));
end
v  = L*randn(ny, Ns+1);

dt = Ts/Nsim;
dW = sqrt(dt) * randn(nw, Ts/dt, Ns);

%% Closed-loop (EKF only, no NMPC) — FIXED INDEXING

% Sample time grid
t_samp = (t0:Ts:tf)';              % length Ns+1

% Full sim storage (Euler-Maruyama internal grid)
T = zeros(Ns*Nsim+1, 1);
X = zeros(nx, Ns*Nsim+1);
T(1)   = t0;
X(:,1) = x0;

% Sampled signals for ID
U = zeros(nu, Ns);                 % u_k applied over [t_k, t_{k+1}]
Y = zeros(ny, Ns+1);               % y_k at t_k
Xhat = zeros(nx, Ns+1);            % EKF filtered estimate at t_k
Phat = zeros(nx, nx, Ns+1);        % EKF covariance at t_k

% Initialize plant
xk = x0;

% Initialize EKF prior at t0
xk_k  = x0;
Pk_k  = 5*eye(nx);

useStepInput = false;

for k = 1:Ns
    tk = t_samp(k);

    fprintf("Iteration %4d of %4d...\n", k, Ns);

    % ---- Measurement at t_k and filtering at t_k ----
    yk = measurementFunSim(tk, xk, pSim) + v(:,k);
    Y(:,k) = yk;

    % Filter update at t_k using prior (xk_k,Pk_k)
    [xk_k, Pk_k] = cdEKFFiltering(tk, yk, xk_k, Pk_k, measurementFun, pCtrl);

    Xhat(:,k)    = xk_k;
    Phat(:,:,k)  = Pk_k;

    % ---- Choose input to apply over [t_k, t_{k+1}] ----
    if ~useStepInput
        uk = u0;
    else
        if k <= floor(Ns/2)
            uk = u0;
        else
            uk = 1.2*u0;
        end
    end

    if k > 300
        uk = [300.0; 200.0];
    end
    if k > 800
        uk = [500.0; 100.0];
    end

    U(:,k) = uk;

    % ---- Known/used disturbance over [t_k, t_{k+1}] ----
    dk = D(:,k);

    % ---- EKF prediction from t_k to t_{k+1} (PRIOR for next sample) ----
    tspanPred = [tk, t_samp(k+1)];
    [xkp1_k, Pkp1_k] = cdEKFPredictionERK1( ...
        driftFun, diffusionFun, tspanPred, dt, xk_k, Pk_k, uk, dk, pCtrl );

    % ---- Simulate true plant from t_k to t_{k+1} ----
    tspanSim = [tk, t_samp(k+1)];
    [Tsim, Xsim] = eulerMaruyama( driftFunSim, diffusionFunSim, ...
        tspanSim, dt, xk, uk, dk, pSim, dW(:,:,k) );

    % Store the internal trajectory for plotting
    idx = (2:Nsim+1) + (k-1)*Nsim;
    T(idx)   = Tsim(2:end);
    X(:,idx) = Xsim(:,2:end);

    % Advance plant state
    xk = Xsim(:,end);

    % Carry EKF prior to next sample (this is x_{k+1|k})
    xk_k = xkp1_k;
    Pk_k = Pkp1_k;

    fprintf("Done.\n\n");
end

% ---- Final measurement at t_{Ns+1}=tf and final filtering ----
tN = t_samp(end);
yN = measurementFunSim(tN, xk, pSim) + v(:,end);
Y(:,end) = yN;

[xN_N, PN_N] = cdEKFFiltering(tN, yN, xk_k, Pk_k, measurementFun, pCtrl);
Xhat(:,end)   = xN_N;
Phat(:,:,end) = PN_N;

%% PEM (Problem 12.3) — consistent with data indexing above

assert(size(Y,2) == Ns+1, 'Y must be ny x (Ns+1)');
assert(size(U,2) == Ns,   'U must be nu x Ns');
assert(size(D,2) >= Ns,   'D must be nd x Ns (or longer)');

theta0  = pem_defaultTheta0(pCtrl);
lb      = 0.2*theta0;
ub      = 2.0*theta0;
lb(theta0==0) = -1;
ub(theta0==0) = 1;

pemOpts = struct();
pemOpts.dt        = dt;
pemOpts.xhat0     = Xhat(:,1);      % filtered at t0 (now meaningful)
pemOpts.P0        = Phat(:,:,1);
pemOpts.useLogDet = true;
pemOpts.reg       = 0.0;

obj = @(th) pem_objectiveMFTS(th, Y, U, D, t_samp, ...
                             driftFun, diffusionFun, measurementFun, ...
                             pCtrl, pemOpts);

fprintf('\n=== Running PEM identification (fmincon) ===\n');

problem = struct();
problem.objective = obj;
problem.x0        = theta0(:);
problem.lb        = lb(:);
problem.ub        = ub(:);
problem.solver    = 'fmincon';

opts = optimoptions('fmincon', ...
    'Display','iter', ...
    'Algorithm','sqp', ...
    'MaxFunctionEvaluations', 5e4, ...
    'MaxIterations', 500, ...
    'StepTolerance', 1e-10, ...
    'OptimalityTolerance', 1e-7);

[thetaHat, Jhat] = fmincon(problem, opts);
fprintf('\nPEM done. Final objective J = %.6g\n', Jhat);

pHat = pem_applyThetaToParams(pCtrl, thetaHat);
pem_printParamComparison(pSim, pHat);

fprintf('\n=== Comparing one-step-ahead predictions ===\n');

[predTrue, innovTrue] = pem_runPredictorMFTS( ...
    pem_defaultTheta0(pSim), Y, U, D, t_samp, ...
    driftFun, diffusionFun, measurementFun, pSim, pemOpts);

[predHat, innovHat] = pem_runPredictorMFTS( ...
    thetaHat, Y, U, D, t_samp, ...
    driftFun, diffusionFun, measurementFun, pCtrl, pemOpts);

eTrue = innovTrue(:,2:end);
eHat  = innovHat(:,2:end);

rmseTrue = sqrt(mean(eTrue.^2,2));
rmseHat  = sqrt(mean(eHat.^2,2));

fprintf('RMSE innovation (true model):       %s\n', mat2str(rmseTrue', 6));
fprintf('RMSE innovation (identified model): %s\n', mat2str(rmseHat',  6));

tmin = t_samp/60;

figure('units','normalized','outerposition',[0.1 0.1 0.9 0.8]);

subplot(2,1,1);
plot(tmin, Y(1,:), '.', 'DisplayName','$y_1$'); hold on;
plot(tmin, predTrue(1,:), '-',  'LineWidth',2, 'DisplayName','$\hat y_{1|k-1}$ (true)');
plot(tmin, predHat(1,:),  '--', 'LineWidth',2, 'DisplayName','$\hat y_{1|k-1}$ (id)');
grid on; xlabel('Time [min]'); ylabel('Output 1'); legend('Location','best');

subplot(2,1,2);
if size(Y,1) >= 2
    plot(tmin, Y(2,:), '.', 'DisplayName','$y_2$'); hold on;
    plot(tmin, predTrue(2,:), '-',  'LineWidth',2, 'DisplayName','$\hat y_{2|k-1}$ (true)');
    plot(tmin, predHat(2,:),  '--', 'LineWidth',2, 'DisplayName','$\hat y_{2|k-1}$ (id)');
    grid on; xlabel('Time [min]'); ylabel('Output 2'); legend('Location','best');
else
    plot(tmin, innovHat(1,:), 'LineWidth',2);
    grid on; xlabel('Time [min]'); ylabel('Innovation');
end

%% ========= Local Functions =========

function J = pem_objectiveMFTS(theta, Y, U, D, t_samp, ...
                               driftFun, diffusionFun, measurementFun, ...
                               pBase, opts)
    p = pem_applyThetaToParams(pBase, theta);

    ny = p.ny;
    nx = p.nx;
    Ns = size(U,2);

    dt    = opts.dt;
    xk_k  = opts.xhat0;   % filtered state at t1 (here t0)
    Pk_k  = opts.P0;

    J = 0.0;
    epsS = 1e-9;

    for k = 1:Ns
        tk   = t_samp(k);
        tkp1 = t_samp(k+1);

        uk = U(:,k);     % applied over [tk, tkp1]
        dk = D(:,k);

        % Predict to tkp1
        [xkp1_k, Pkp1_k] = cdEKFPredictionERK1( ...
            driftFun, diffusionFun, [tk, tkp1], dt, xk_k, Pk_k, uk, dk, p);

        % One-step-ahead output prediction at tkp1
        yhat = measurementFun(tkp1, xkp1_k, p);
        ek   = Y(:,k+1) - yhat;

        % Innovation covariance
        C = pem_fdJacobian(@(x) measurementFun(tkp1, x, p), xkp1_k, ny, nx);
        S = C*Pkp1_k*C' + p.R;
        S = 0.5*(S+S') + epsS*eye(ny);

        [L,flag] = chol(S,'lower');
        if flag ~= 0
            Sinv = pinv(S);
            quad = ek' * Sinv * ek;
            if opts.useLogDet
                logdetS = log(max(det(S), epsS));
            else
                logdetS = 0;
            end
        else
            v = L\ek;
            quad = v'*v;
            if opts.useLogDet
                logdetS = 2*sum(log(diag(L)));
            else
                logdetS = 0;
            end
        end

        J = J + quad + logdetS;

        if isfield(opts,'reg') && opts.reg > 0
            J = J + opts.reg*(theta(:)'*theta(:));
        end

        % Filter at tkp1 to keep innovations-form predictor stable
        [xk_k, Pk_k] = cdEKFFiltering(tkp1, Y(:,k+1), xkp1_k, Pkp1_k, measurementFun, p);
    end
end

function [yPred, innov] = pem_runPredictorMFTS(theta, Y, U, D, t_samp, ...
                                               driftFun, diffusionFun, measurementFun, ...
                                               pBase, opts)
    p = pem_applyThetaToParams(pBase, theta);

    ny = p.ny;
    Ns = size(U,2);

    yPred = zeros(ny, Ns+1);
    innov = zeros(ny, Ns+1);

    dt    = opts.dt;
    xk_k  = opts.xhat0;
    Pk_k  = opts.P0;

    yPred(:,1) = Y(:,1);
    innov(:,1) = zeros(ny,1);

    for k = 1:Ns
        tk   = t_samp(k);
        tkp1 = t_samp(k+1);

        uk = U(:,k);
        dk = D(:,k);

        [xkp1_k, Pkp1_k] = cdEKFPredictionERK1( ...
            driftFun, diffusionFun, [tk, tkp1], dt, xk_k, Pk_k, uk, dk, p);

        yhat = measurementFun(tkp1, xkp1_k, p);

        yPred(:,k+1) = yhat;
        innov(:,k+1) = Y(:,k+1) - yhat;

        [xk_k, Pk_k] = cdEKFFiltering(tkp1, Y(:,k+1), xkp1_k, Pkp1_k, measurementFun, p);
    end
end

function J = pem_fdJacobian(fun, x, ny, nx)
    fx = fun(x);
    if nargin < 3, ny = numel(fx); end
    if nargin < 4, nx = numel(x);  end

    J = zeros(ny, nx);
    h = 1e-6;

    for i = 1:nx
        dx = zeros(nx,1);
        dx(i) = h*(1+abs(x(i)));
        fp = fun(x + dx);
        fm = fun(x - dx);
        J(:,i) = (fp - fm) / (2*dx(i));
    end
end

function theta0 = pem_defaultTheta0(p)
    theta0 = [];

    if isfield(p,'k1'),      theta0 = [theta0; p.k1]; end
    if isfield(p,'k2'),      theta0 = [theta0; p.k2]; end
    if isfield(p,'gamma1'),  theta0 = [theta0; p.gamma1]; end
    if isfield(p,'gamma2'),  theta0 = [theta0; p.gamma2]; end
    if isfield(p,'a1'),      theta0 = [theta0; p.a1]; end
    if isfield(p,'a2'),      theta0 = [theta0; p.a2]; end
    if isfield(p,'a3'),      theta0 = [theta0; p.a3]; end
    if isfield(p,'a4'),      theta0 = [theta0; p.a4]; end

    if isempty(theta0)
        error('pem_defaultTheta0: no fields found. Edit this function to match parametersMFTS().');
    end
end

function p = pem_applyThetaToParams(pBase, theta)
% IMPORTANT: EDIT THIS MAPPING TO MATCH YOUR parametersMFTS() STRUCT.
% The code runs only if these fields exist and the order matches pem_defaultTheta0.

    p = pBase;
    th = theta(:);

    if isfield(p,'k1'),      p.k1      = th(1); end
    if isfield(p,'k2'),      p.k2      = th(2); end

    if isfield(p,'gamma1'),  p.gamma1  = min(max(th(3),0),1); end
    if isfield(p,'gamma2'),  p.gamma2  = min(max(th(4),0),1); end

    if isfield(p,'a1'),      p.a1      = max(th(5), 1e-9); end
    if isfield(p,'a2'),      p.a2      = max(th(6), 1e-9); end
    if isfield(p,'a3'),      p.a3      = max(th(7), 1e-9); end
    if isfield(p,'a4'),      p.a4      = max(th(8), 1e-9); end
end

function pem_printParamComparison(pTrue, pHat)
    fprintf('\n=== Parameter comparison (true vs identified) ===\n');

    fields = intersect(fieldnames(pTrue), fieldnames(pHat));
    keep = false(size(fields));
    for i = 1:numel(fields)
        a = pTrue.(fields{i});
        b = pHat.(fields{i});
        keep(i) = isnumeric(a) && isnumeric(b) && isscalar(a) && isscalar(b);
    end
    fields = fields(keep);

    for i = 1:numel(fields)
        f = fields{i};
        at = pTrue.(f);
        ah = pHat.(f);
        rel = (ah-at)/(abs(at)+1e-12);
        fprintf('%-12s  true=% .6g   id=% .6g   rel.err=% .3g\n', f, at, ah, rel);
    end
end
function [theta0, names] = pem_defaultTheta0(p)
% Builds theta0 automatically from numeric scalar fields in p.
% Returns:
%   theta0 : vector of initial parameter values
%   names  : corresponding field names in p

    names = pem_getIdentifiableFieldNames(p);

    theta0 = zeros(numel(names),1);
    for i = 1:numel(names)
        theta0(i) = p.(names{i});
    end

    if isempty(theta0)
        error('pem_defaultTheta0: no numeric scalar fields found after exclusions. Update pem_getIdentifiableFieldNames().');
    end
end

function p = pem_applyThetaToParams(pBase, theta)
% Applies theta values back into pBase using the same auto-selected fields.

    p = pBase;
    names = pem_getIdentifiableFieldNames(p);

    th = theta(:);
    if numel(th) ~= numel(names)
        error('pem_applyThetaToParams: theta length (%d) does not match number of selected fields (%d).', numel(th), numel(names));
    end

    for i = 1:numel(names)
        p.(names{i}) = th(i);
    end
end

function names = pem_getIdentifiableFieldNames(p)
% Select which fields in parameters struct are "identifiable parameters".
% We take numeric scalar fields and exclude dimensions/covariances/flags/etc.

    fn = fieldnames(p);
    names = {};

    % Exclude common non-parameter fields (edit this list if needed)
    exclude = { ...
        'nx','ny','nz','nu','nd','nw', ...
        'Q','R','P0','x0', ...
        'Ts','dt','N','Ns','M','Nsim', ...
        'g','rho','pi', ...
        'name','label' ...
    };

    for k = 1:numel(fn)
        f = fn{k};

        if any(strcmp(f, exclude))
            continue
        end

        v = p.(f);

        % Keep numeric scalar real values
        if isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v)
            names{end+1,1} = f; %#ok<AGROW>
        end
    end

    % Optional: sort for stable ordering
    names = sort(names);

    % Safety: if this selects too many parameters, reduce the set here.
    % For example, keep only fields containing 'a' or 'k':
    % names = names(contains(names,'a') | contains(names,'k'));
end

