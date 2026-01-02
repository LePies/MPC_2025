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


% Set casadiPath
casadiPath = 'C:\casadi';
addpath( genpath(casadiPath) );

%% Save Plot/Data

% Save booleans
savePlot = 0;

% Names and paths
name      = 'MFTSCollocationClosedLoop';
figPath   = './figures/';

%% Simulation Info

% Simulation period and sampling time
t0    = 0;                      % [s]       :   initial time
tf    = 60*20;                  % [s]       :   final time
Ts    = 1.0;                   % [s]       :   sampling time
Nsim  = 10;                     % []        :   euler-maruyama steps in simulation
N     = 30;               % []        :   discrete control horizon 
Ns    = (tf-t0)/Ts;             % []        :   sampling intervals
M     = 10;                     % []        :   euler steps in OCP

% Initial condition
u0 = [  250.0 ;  325.0 ];
d0 = [   100.0 ;   120.0 ];
x0 = [21362.92718669; 19317.4122346; 11195.65652256;  8530.21545886;d0];

% Boundaries
uMin = [   0.0;   0.0 ];      	% [cm^3/s]	:   input lower bound
uMax = [ 500.0; 500.0 ];        % [cm^3/s] 	:   input upper bound
dMin = [   0.0;   0.0 ];        % [cm^3/s]	:   disturbance lower bound
dMax = [ 100.0; 100.0 ];        % [cm^3/s]	:   disturbance lower bound

%% Model

% Simulation model 
pSim                = parametersMFTS();
driftFunSim         = @driftMFTS;
diffusionFunSim     = @diffusionMFTS;
measurementFunSim   = @measurementMFTS;

% Controller model (can be different from the simulation model)
pCtrl           = parametersMFTS();
driftFun        = @driftMFTS;
diffusionFun    = @diffusionMFTS;
outputFun       = @outputMFTS;
measurementFun  = @measurementMFTS;

% Dimensions (note: these can differ from simulation to controller model)
nx = pSim.nx;
nz = pSim.nz;
nd = pSim.nd;
nu = pSim.nu;
ny = pSim.ny;
nw = pSim.nw;

%% Input and Disturbance Initialisation

% Initialize
D = repmat( d0, 1, Ns+N );

%% Setpoint


% Define setpoints
zBar = zeros( nz, Ns+N*M );
zBar(1, 1:round(Ns/2) ) = 50.0;
zBar(2, 1:round(Ns/2) ) = 30.0;

zBar(1, round(Ns/2)+1:end ) = 35.0;
zBar(2, round(Ns/2)+1:end ) = 45.0;

%% Solve OCP

% Define autonomous function definition
model = @(x, u, d, p) driftFun(0, x, u, d, pCtrl); 

% Initial guess for optimization
X0 = repmat( x0, 1, M*N+1 );
U0 = repmat( u0, 1, N     );

% Setup multiple shooting constraints
[opti, x, u, d, x0Cas] = casadiCollocation(model, X0, U0, Ts, M, x0, ...
                                                        D(:,1:N), pCtrl);
% Output function
z       = outputFun(0, x, pCtrl);           % Compute symbolic output
zbar    = opti.parameter(nz, size(z,2));    % Make output target symbolic

% 
ukm1Cas = opti.parameter(nu, 1);            % Previous input

% Objectives (output target tracking and input rate of movement penalty)
alpha = 0.995;
phiz  = sum( ( reshape( z           , nz*(M*N+1), 1 ) - ...
               reshape( zbar        , nz*(M*N+1), 1 ) ).^2 );
phidu = sum((u(:, 1) - ukm1Cas).^2) + ...
            sum( ( reshape( u(:,2:end)  , nu*(N-1), 1 ) - ...
               reshape( u(:,1:end-1), nu*(N-1), 1 ) ).^2 );

% Define optimal control problem (OCP)
% ...
% objective
opti.minimize( alpha*phiz + ( 1 - alpha )*phidu );
% constraints
opti.subject_to( uMin(1) <= u(1,:) <= uMax(1) );
opti.subject_to( uMin(2) <= u(2,:) <= uMax(2) );
opti.subject_to( 0.0     <= x(:)              );

% options
p_opts = struct;
s_opts = struct;
% acceptable tolerance
s_opts.acceptable_tol             = 1.0e-7;
s_opts.acceptable_dual_inf_tol    = 1.0e-7;
s_opts.acceptable_constr_viol_tol = 1.0e-7;
s_opts.acceptable_compl_inf_tol   = 1.0e-7;
s_opts.acceptable_obj_change_tol  = 1.0e-7;
s_opts.acceptable_iter            = 10;
% convergence tolerance
s_opts.tol                        = 1.0e-7;
s_opts.dual_inf_tol               = 1.0e-7;
s_opts.constr_viol_tol            = 1.0e-7;
s_opts.max_iter                   = 5000;
% misc options
s_opts.hessian_approximation      = 'exact';    %'limited-memory';

% make silent
s_opts.print_level                = 0;
p_opts.print_time                 = 0;

% Casadi solver
opti.solver( 'ipopt', p_opts, s_opts );


%% Setup noise
% Seed
seed = 5;
rng(seed);

% Measurement noise
Rv = pCtrl.R;
try 
    L = chol(Rv, 'lower');
catch
    L = zeros(size(Rv));
end
v = L*randn( ny, Ns+1 );

% Process noise
dt = Ts/Nsim; % Nsim euler-maruyama in simulation (and euler steps in CDEKF)
dW = sqrt(dt) * randn(nw, Ts/dt, Ns);

%% Closed-loop (EKF only, no NMPC)

% Save for plotting
T = zeros(Ns*Nsim+1, 1);
T(1) = t0;

X = zeros(nx, Ns*Nsim+1);
X(:, 1) = x0;

% EKF estimates at sampling instants (k = 1..Ns+1)
Xhat = zeros(nx, Ns+1);
Xhat(:,1) = x0;

U = zeros(nu, Ns);
Y = zeros(ny, Ns+1);

% Simulation data
tk = t0;
xk = x0;

% CDEKF states
xkm1km1 = x0;
Pkm1km1 = 5*eye(nx);

% Previous inputs and disturbances (for EKF prediction)
ukm1 = u0;
dkm1 = d0;

% Choose an input strategy for "EKF-only" testing
% Option A: constant input
useStepInput = false;

Phat = zeros(nx, nx, Ns+1);
Phat(:,:,1) = Pkm1km1;


% Closed-loop
for k = 1:Ns

    fprintf("Iteration %4d of %4d... \n", k, Ns);

    % ------------------- Measurement ------------------- %
    yk = measurementFunSim(tk, xk, pSim) + v(:, k);

    % ------------------- CDEKF -------------------------- %
    tspanCDEKF = [tk-Ts, tk];

    [xkkm1, Pkkm1] = cdEKFPredictionERK1( driftFun, diffusionFun, ...
        tspanCDEKF, dt, xkm1km1, Pkm1km1, ukm1, dkm1, pCtrl );

    [xkk, Pkk] = cdEKFFiltering( tk, yk, xkkm1, Pkkm1, ...
        measurementFun, pCtrl );

    Phat(:,:,k) = Pkk;

    % Save estimate (sampling instant k)
    Xhat(:, k) = xkk;

    % Prepare for next iteration (EKF memory)
    xkm1km1 = xkk;
    Pkm1km1 = Pkk;

    % ------------------- Input to plant ----------------- %
    if ~useStepInput
        uk = u0;   % constant input
    else
        % simple step test (edit as you like)
        if k <= floor(Ns/2)
            uk = u0;
        else
            uk = 1.2*u0;
        end
    end
    
    if k>300
        uk = [  300.0 ;  200.0 ];
    end

    if k>800
        uk = [  500.0 ;  100.0 ];
    end

    % Save as "previous applied input" for next EKF prediction
    ukm1 = uk;

    % ------------------- Simulation --------------------- %
    % Current disturbances (true plant)
    dk = D(:, k);

    % Save as "previous disturbance" for next EKF prediction
    dkm1 = dk;

    % Simulation time span
    tspan = [tk, tk+Ts];

    % Simulate system
    [Tsim, Xsim] = eulerMaruyama( driftFunSim, diffusionFunSim, ...
        tspan, dt, xk, uk, dk, pSim, dW(:,:,k) );

    % Save data (same indexing style you already had)
    T(   (2:Nsim+1) + (k-1)*Nsim ) = Tsim(2:end);
    X(:, (2:Nsim+1) + (k-1)*Nsim ) = Xsim(:,2:end);

    U(:, k) = uk;
    Y(:, k) = yk;

    % Update time and state for next iteration
    tk = Tsim(end);
    xk = Xsim(:, end);

    fprintf("Done.\n\n");
end

Phat(:,:,end) = Pkm1km1;

% Last measurement
Y(:, end) = measurementFunSim(tk, xk, pSim) + v(:, end);

% Optional: last EKF estimate (store a final update at k = Ns+1)
% If you want Xhat to have Ns+1 meaningful entries, do a last filter update:
tspanCDEKF = [tk-Ts, tk];
[xkkm1, Pkkm1] = cdEKFPredictionERK1( driftFun, diffusionFun, ...
    tspanCDEKF, dt, xkm1km1, Pkm1km1, ukm1, dkm1, pCtrl );
[xkk, ~] = cdEKFFiltering( tk, Y(:, end), xkkm1, Pkkm1, measurementFun, pCtrl );
Xhat(:, end) = xkk;


%% Plot 

%% Plot (Problem 12.2: CDEKF performance)

% True output along the full simulated trajectory
Z = outputFun(0, X, pSim);

% EKF-estimated output at sampling instants (based on EKF state estimate)
Zhat = outputFun(0, Xhat, pCtrl);

% Sampling time vector
t = t0:Ts:tf;                 % length Ns+1
tMin = t/60;

% Indices in T/X corresponding to sampling instants
idx = 1:Nsim:(Ns*Nsim+1);
Tsamp = T(idx)/60;

% True states/outputs sampled at measurement instants
Xsamp = X(:, idx);
Zsamp = Z(:, idx);

% Fontsize and linewidth
fs = 24;
lw = 6;
ms = 10;

% Limits
tLim = [T(1), T(end)]/60;
zLim = [0.0, 1.50*max(max(Z))];
xLim = [0.0e+0, 1.50*max(max(X))]/1000;
uLim = [uMin(1), 1.25*uMax(1)];
dLim = [dMin(1), 1.25*dMax(1)];

% Figure
fig = figure('units','normalized','outerposition',[0.5 0 1.0 1.0]);

%% (1) Outputs: true z vs estimated zhat vs measurements y
subplot(2,2,1);

% True outputs at sampling instants (clean comparison with EKF & measurements)
plot(Tsamp, Zsamp(1,:), 'linewidth', lw, ...
    'DisplayName', '$z_1$ (true, sampled)', 'linestyle','-');
hold on
plot(Tsamp, Zsamp(2,:), 'linewidth', lw, ...
    'DisplayName', '$z_2$ (true, sampled)', 'linestyle','-');

% EKF estimated outputs
plot(tMin, Zhat(1,:), 'linewidth', lw, ...
    'DisplayName', '$\hat z_1$ (EKF)', 'linestyle','--');
plot(tMin, Zhat(2,:), 'linewidth', lw, ...
    'DisplayName', '$\hat z_2$ (EKF)', 'linestyle','--');

% Measurements
plot(tMin, Y(1,:), '.', 'markersize', ms, ...
    'displayname', '$y_1$ (meas)');
plot(tMin, Y(2,:), '.', 'markersize', ms, ...
    'displayname', '$y_2$ (meas)');

grid on
ylabel('Heights [cm]')
xlim(tLim)
ylim(zLim)
legend('location','north','orientation','horizontal','numcolumns',2)
set(gca,'fontsize',fs)
hold off

%% (2) States: true sampled vs EKF estimate
subplot(2,2,2);

plot(Tsamp, Xsamp(1,:)/1000, 'linewidth', lw, 'displayname', '$x_1$ (true)');
hold on
plot(Tsamp, Xsamp(2,:)/1000, 'linewidth', lw, 'displayname', '$x_2$ (true)');
plot(Tsamp, Xsamp(3,:)/1000, 'linewidth', lw, 'displayname', '$x_3$ (true)');
plot(Tsamp, Xsamp(4,:)/1000, 'linewidth', lw, 'displayname', '$x_4$ (true)');

plot(tMin, Xhat(1,:)/1000, 'linewidth', lw, 'linestyle','--', 'displayname', '$\hat x_1$ (EKF)');
plot(tMin, Xhat(2,:)/1000, 'linewidth', lw, 'linestyle','--', 'displayname', '$\hat x_2$ (EKF)');
plot(tMin, Xhat(3,:)/1000, 'linewidth', lw, 'linestyle','--', 'displayname', '$\hat x_3$ (EKF)');
plot(tMin, Xhat(4,:)/1000, 'linewidth', lw, 'linestyle','--', 'displayname', '$\hat x_4$ (EKF)');

grid on
ylabel('Masses [kg]')
xlim(tLim)
ylim(xLim)
legend('location','north','orientation','horizontal','numcolumns',2)
set(gca,'fontsize',fs)
hold off

%% (3) Inputs (optional but useful)
subplot(2,2,3);

stairs(tMin, [U(1,:) U(1,end)], 'linewidth', lw, 'displayname', '$u_1$');
hold on
stairs(tMin, [U(2,:) U(2,end)], 'linewidth', lw, 'displayname', '$u_2$');

plot([t0 tf]/60, [uMin(1) uMin(1)], 'k--', 'linewidth', ceil(lw/2), 'displayname', 'bounds');
p = plot([t0 tf]/60, [uMax(1) uMax(1)], 'k--', 'linewidth', ceil(lw/2));
p.Annotation.LegendInformation.IconDisplayStyle = 'off';

grid on
ylabel('Inlet flow rates [cm$^3$/s]')
xlabel('Time [min]')
xlim(tLim)
ylim(uLim)
legend('location','north','orientation','horizontal','numcolumns',3)
set(gca,'fontsize',fs)
hold off

%% (4) Disturbances (optional)
subplot(2,2,4);

stairs(tMin, D(1,1:Ns+1), 'linewidth', lw, 'displayname', '$d_1$');
hold on
stairs(tMin, D(2,1:Ns+1), 'linewidth', lw, 'displayname', '$d_2$');

plot([t0 tf]/60, [dMin(1) dMin(1)], 'k--', 'linewidth', ceil(lw/2), 'displayname', 'bounds');
p = plot([t0 tf]/60, [dMax(1) dMax(1)], 'k--', 'linewidth', ceil(lw/2));
p.Annotation.LegendInformation.IconDisplayStyle = 'off';

grid on
ylabel('Disturbance flow rates [cm$^3$/s]')
xlabel('Time [min]')
xlim(tLim)
ylim(dLim)
legend('location','north','orientation','horizontal','numcolumns',3)
set(gca,'fontsize',fs)
hold off


%% Export data to Excel (Problem 12.2)

filename = 'Problem12_2_EKF_Data.xlsx';

% ---------- Sampling grid ----------
t_samp = (t0:Ts:tf)';           % Ns+1 x 1
t_samp_min = t_samp/60;

% Indices for sampling instants in continuous trajectory
idx = 1:Nsim:(Ns*Nsim+1);

% True states & outputs sampled at EKF instants
Xsamp = X(:, idx);
Zsamp = outputFun(0, Xsamp, pSim);

% EKF estimated outputs
Zhat = outputFun(0, Xhat, pCtrl);

% ---------- Sheet 1: EKF & Measurements ----------
EKF_table = table( ...
    t_samp_min, ...
    Y(1,:)', Y(2,:)', ...
    Zsamp(1,:)', Zsamp(2,:)', ...
    Zhat(1,:)', Zhat(2,:)', ...
    Xhat(1,:)', Xhat(2,:)', ...
    Xhat(3,:)', Xhat(4,:)', ...
    'VariableNames', { ...
        'time_min', ...
        'y1_meas', 'y2_meas', ...
        'z1_true', 'z2_true', ...
        'z1_hat', 'z2_hat', ...
        'x1_hat_kg', 'x2_hat_kg', ...
        'x3_hat_kg', 'x4_hat_kg' } );

writetable(EKF_table, filename, 'Sheet', 'EKF_Data');

% ---------- Sheet 2: True states at sampling instants ----------
TrueStates_table = table( ...
    t_samp_min, ...
    Xsamp(1,:)', Xsamp(2,:)', ...
    Xsamp(3,:)', Xsamp(4,:)', ...
    'VariableNames', { ...
        'time_min', ...
        'x1_true_kg', 'x2_true_kg', ...
        'x3_true_kg', 'x4_true_kg' } );

writetable(TrueStates_table, filename, 'Sheet', 'True_States_Sampled');

% ---------- Sheet 3: Inputs ----------
Inputs_table = table( ...
    t_samp_min(1:end-1), ...
    U(1,:)', U(2,:)', ...
    'VariableNames', { 'time_min', 'u1', 'u2' } );

writetable(Inputs_table, filename, 'Sheet', 'Inputs');

% ---------- Sheet 4: Disturbances ----------
Dist_table = table( ...
    t_samp_min, ...
    D(1,1:Ns+1)', D(2,1:Ns+1)', ...
    'VariableNames', { 'time_min', 'd1', 'd2' } );

writetable(Dist_table, filename, 'Sheet', 'Disturbances');

% ---------- Sheet 5 (optional): Continuous trajectory ----------
Zfull = outputFun(0, X, pSim);

Cont_table = table( ...
    T/60, ...
    X(1,:)', X(2,:)', ...
    X(3,:)', X(4,:)', ...
    Zfull(1,:)', Zfull(2,:)', ...
    'VariableNames', { ...
        'time_min', ...
        'x1_kg', 'x2_kg', 'x3_kg', 'x4_kg', ...
        'z1', 'z2' } );

writetable(Cont_table, filename, 'Sheet', 'Continuous_Trajectory');

% Standard deviations (sqrt of diagonal)
stdx = zeros(nx, Ns+1);
for k = 1:Ns+1
    stdx(:,k) = sqrt(diag(Phat(:,:,k)));
end

Std_table = table( ...
    t_samp_min, ...
    stdx(1,:)', stdx(2,:)', stdx(3,:)', stdx(4,:)', ...
    'VariableNames', {'time_min','std_x1','std_x2','std_x3','std_x4'} );

writetable(Std_table, filename, 'Sheet', 'EKF_Std');



fprintf('EKF data exported to %s\n', filename);


