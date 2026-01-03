# MPC 2025 - Model Predictive Control Exam Project

This repository contains the implementation of Model Predictive Control (MPC) algorithms applied to a **Four Tank System**. The project covers system identification, state estimation, and various MPC formulations including constrained MPC and economic MPC.

---

## Project Structure

```
MPC_2025/
├── src/                          # Core modules and implementations
│   ├── FourTankSystem.py         # Four tank system dynamics and simulation
│   ├── MPC.py                    # Model Predictive Controller implementation
│   ├── MPC_economic.py           # Economic MPC implementation
│   ├── KalmanFilterUpdate.py     # Discrete-time Kalman filter
│   ├── ExtendedKalmanFilterUpdate.py  # Extended Kalman filter for nonlinear systems
│   ├── PIDcontrolor.py           # PID controller implementation
│   ├── HankelSystem.py           # Hankel matrix-based system identification
│   ├── PEM.py                    # Prediction Error Method for system ID
│   ├── QPSolver.py               # Quadratic programming solver
│   ├── PlotMPC_sim.py            # MPC simulation plotting utilities
│   └── PlotSimulation.py         # General plotting utilities
│
├── params/                       # System parameters and initialization
│   ├── parameters_tank.py        # Physical parameters of the four tank system
│   ├── initialize.py             # Initial conditions and operating points
│   └── pid_params.py             # PID controller tuning parameters
│
├── figures/                      # Output figures organized by problem
│   ├── Problem2/                 # Open-loop simulation plots
│   ├── Problem3/                 # PID control plots
│   ├── Problem4/                 # System identification plots
│   ├── Problem5/                 # Markov parameters and Hankel analysis
│   ├── Problem6/                 # Kalman filter estimation plots
│   ├── Problem8/                 # Unconstrained MPC plots
│   ├── Problem9/                 # Constrained MPC plots
│   ├── Problem10/                # Soft-constrained MPC plots
│   ├── Problem12/                # NMPC/EKF plots
│   └── Problem13/                # Economic MPC plots
│
├── Results/                      # Saved data and intermediate results
│   ├── Problem4/                 # System identification results
│   └── Problem5/                 # Estimated system matrices
│
├── Problem_*.py                  # Main problem scripts (see below)
└── CasADi_framework_example/     # MATLAB reference examples using CasADi
```

---

## Problem Scripts Overview

### Problem 2: Open-Loop Simulation (`Problem_2.py`)
Simulates the four tank system in open-loop under three different model formulations:
1. **Deterministic model** - No process or measurement noise
2. **Piecewise constant noise model** - Disturbances modeled as piecewise constant signals
3. **SDE model** - Disturbances modeled as stochastic differential equations (Ornstein-Uhlenbeck process)

Computes steady-state conditions and visualizes tank heights and flow rates over time.

---

### Problem 3: PID Control (`Problem_3.py`)
Implements and tunes a **PID controller** for the four tank system:
- Runs closed-loop simulations with different gain combinations
- Tests proportional-only (`Kp`), PI (`Kp + Ki`), and full PID (`Kp + Ki + Kd`) control
- Evaluates setpoint tracking performance for tank heights h₁ and h₂
- Command line arguments: `p`, `pi`, `pid`, `sim`, `test`

---

### Problem 4: System Identification - Step Response (`Problem_4.py`, `Problem_4_2nd.py`, `Problem_4_3rd.py`)
Performs **step response experiments** for system identification:
- Applies step changes to inputs (u₁, u₂) and disturbances (d₁, d₂)
- Extracts steady-state gains and time constants
- Fits **2nd-order transfer function models** using curve fitting
- Tests robustness to different noise levels (0, 0.01, 0.1, 0.5)
- Saves identified parameters (K, τ₁, τ₂) to CSV files

---

### Problem 5: Markov Parameters & Hankel Matrix (`Problem_5.py`)
Analyzes the linearized discrete-time system:
- Computes **Markov parameters** (impulse response coefficients)
- Constructs **Hankel matrices** for system realization
- Performs **SVD decomposition** to determine system order
- Extracts A, B, C matrices from Hankel decomposition
- Computes transfer function zeros and poles for SISO subsystems
- Saves estimated matrices to `Results/Problem5/`

---

### Problem 6: Kalman Filter State Estimation (`Problem_6.py`)
Implements **discrete-time Kalman filtering** for state and disturbance estimation:
- Augments the state space to include disturbance states (d₁, d₂)
- Runs filter on both linear and nonlinear system simulations
- Handles step changes in disturbances
- Compares true states vs. Kalman estimates
- Generates estimation error plots

---

### Problem 8: Unconstrained MPC (`Problem_8_4.py`, `Problem_8_5.py`)
Implements **unconstrained linear MPC**:
- **Problem_8_4.py**: Uses identified model from Problem 4 (step response ID)
- **Problem_8_5.py**: Uses linearized model from Problem 5 (Hankel realization)
- Tracks setpoint changes in tank heights (h₁, h₂)
- Tunable weighting matrices: Wz (output), Wu (input), Wdu (input rate)
- Prediction horizon N = 30 steps

---

### Problem 9: Constrained MPC (`Problem_9_4.py`, `Problem_9_5.py`)
Extends MPC with **hard constraints**:
- **Input constraints**: Umin = [0, 0], Umax = [3000, 3000] cm³/s
- **Input rate constraints**: Dmin, Dmax for smooth control
- **Problem_9_4.py**: Uses Problem 4 model
- **Problem_9_5.py**: Uses Problem 5 model
- Longer prediction horizon (N = 150 steps for Problem_9_4)

---

### Problem 10: Soft-Constrained MPC (`Problem_10_4.py`, `Problem_10_5.py`)
Implements **soft output constraints** using slack variables:
- Adds output constraints with margins: Rmin, Rmax
- Slack variable penalties:
  - Quadratic: Ws2, Wt2 (weight = 1000)
  - Linear: Ws1, Wt1 (weight = 10)
- Allows temporary constraint violations with penalty
- **Problem_10_4.py**: Uses Problem 4 model
- **Problem_10_5.py**: Uses Problem 5 model

---

### Problem 12: Nonlinear MPC with EKF (`Problem_12.py`, `Problem_12_4.py`, `Problm_12.py`)
Implements **Nonlinear MPC (NMPC)** combined with **Extended Kalman Filter**:
- Uses nonlinear four tank system dynamics
- State estimation via continuous-discrete EKF
- Nonlinear optimization for control computation

---

### Problem 13: Economic MPC (`problem_13_1.py`, `Problem_13_2.py`)
Implements **Economic MPC** with time-varying energy costs:
- Cost function: c(t) = 200 - 190·exp(-0.01·t)
- Objective: Minimize total energy cost c·(u₁ + u₂)
- Maintains minimum tank height constraint (H = 40 cm)
- **problem_13_1.py**: QP-based economic MPC
- **Problem_13_2.py**: LP-based economic MPC with rate constraints

---

## Core Modules

### `src/FourTankSystem.py`
The main system class implementing:
- Nonlinear ODEs for tank dynamics
- Stochastic disturbance modeling (Ornstein-Uhlenbeck)
- Open-loop and closed-loop simulation
- Continuous and discrete-time linearization
- Steady-state computation

### `src/MPC.py`
The MPC controller class implementing:
- Prediction matrices (Gamma, Phi, Psi)
- QP problem formulation
- Input and output constraints
- Soft constraints with slack variables
- Real-time control update

### `src/KalmanFilterUpdate.py`
Discrete-time Kalman filter for linear state estimation with optional stationary (time-invariant) mode.

### `src/ExtendedKalmanFilterUpdate.py`
Extended Kalman filter for nonlinear state estimation using continuous-discrete formulation.

---

## Usage

Run individual problem scripts from the project root:

```bash
# Open-loop simulation
python Problem_2.py

# PID control (choose mode: p, pi, pid, sim, test)
python Problem_3.py pi

# System identification
python Problem_4.py

# MPC simulations
python Problem_8_4.py
python Problem_9_4.py
python Problem_10_5.py

# Economic MPC
python problem_13_1.py
```

---

## Dependencies

- NumPy
- SciPy
- Matplotlib
- tqdm
- pandas

---

## System Description

The **Four Tank System** is a benchmark control problem consisting of:
- 4 interconnected water tanks (Tank 1-4)
- 2 controllable pump inputs (F₁, F₂)
- 2 measured disturbances (F₃, F₄)
- 2 controlled outputs (h₁, h₂ - heights of lower tanks)

The tanks are arranged such that:
- Tank 1 receives flow from pump 1 (γ₁·F₁) and drains from Tank 3
- Tank 2 receives flow from pump 2 (γ₂·F₂) and drains from Tank 4
- Tank 3 receives flow from pump 2 ((1-γ₂)·F₂) and disturbance F₃
- Tank 4 receives flow from pump 1 ((1-γ₁)·F₁) and disturbance F₄

The splitting ratios γ₁ and γ₂ determine whether the system operates in minimum-phase or non-minimum-phase configuration.
