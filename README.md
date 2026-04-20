# Extended State Estimation of a Two-Area Power System

> CentraleSupélec 3A — Information & Communication Engineering — 2025/2026  
> Authors: Lucie Peyrot, Alexis Peters, Marin Dagron

---

## Overview

This project investigates **Extended State Estimation** applied to a minimal two-node electrical power system. Unlike conventional static state estimation (which assumes steady-state conditions), this approach reconstructs the **internal dynamic states** of generators — rotor angles, angular velocities, mechanical torques, and secondary control signals — from noisy pseudo-measurements produced by a network state estimator.

The pipeline consists of three main stages:

1. **System Modeling** — Nonlinear electromechanical model linearized around an operating point.
2. **Observability Analysis** — Symbolic and numerical verification that the chosen state-output pair is fully observable.
3. **Kalman Filtering + RTS Smoothing** — Recursive state estimation under stochastic load variations and discrete topology changes (line trip).

---

## Repository Structure

```
Extended-State-Estimation/
│
├── Model and Observability/
│   ├── model_theta1_in_Y.py       # Observable model with θ₁ added to the output vector
│   ├── non_observable_model.py    # Original non-observable model (for comparison)
│   ├── observability.py           # Numerical observability matrix rank computation
│   └── numerical_values.json      # System parameters and equilibrium values
│
└── Filtre_de_Kalman/
    ├── Kalman_PowerSystem.ipynb          # Kalman filter under Gauss-Markov load disturbances
    ├── Kalman_PowerSystem_LineTrip.ipynb # Time-varying Kalman filter for line trip scenario
    └── Per_Window_RTS_Smoother.ipynb     # Per-window RTS smoother implementation
```

---

## System Description

The system is a two-node power network with two synchronous generators, two loads, and a tie-line. The state vector (dimension 9 after augmentation) is:

$$X = [\theta_1,\ \omega_1,\ T_{m1},\ \theta_2,\ \omega_2,\ T_{m2},\ N,\ P_{L1},\ P_{L2}]^\top$$

where $\theta_i$ are rotor angles, $\omega_i$ angular velocities, $T_{mi}$ mechanical torques, $N$ the secondary control signal, and $P_{Li}$ the (unknown) load disturbances included as augmented states.

Key physical parameters:

| Symbol | Description | Value |
|--------|-------------|-------|
| $\omega_0$ | Nominal frequency | $2\pi \times 50$ rad/s |
| $K_L$ | Tie-line synchronising coefficient | 3064 MW/rad |
| $K_s$ | Secondary control gain | 0.05 |
| $J_1$, $J_2$ | Generator inertia constants | 0.4, 0.1 kg·m² |
| $P_{0,1}$, $P_{0,2}$ | Nominal power setpoints | 600, 400 MW |

---

## Observability

The observability matrix $\mathcal{O} = [C^\top,\ (CA)^\top,\ \ldots,\ (CA^{n-1})^\top]^\top$ is computed both symbolically (via SymPy) and numerically (via NumPy).

The original measurement set $Y = [P_{G1},\ P_{G2},\ F_{12}]^\top$ is **not fully observable** (rank < 7) due to the rotational symmetry of grid equations, which only depend on the angle difference $\theta_1 - \theta_2$.

Adding $\theta_1$ (measurable via PMU) and $\theta_2$, and augmenting with $P_{L1}$, $P_{L2}$ as states, the final output vector is:

$$Y = [\theta_1,\ \theta_2,\ F_{12},\ P_{G1},\ P_{G2},\ P_{L1},\ P_{L2}]^\top$$

Symbolic and numerical analysis confirms **rank = 9** (full observability).

---

## Kalman Filter

A **discrete linear Kalman filter** operates on deviations from the equilibrium point. The continuous matrices are discretized via matrix exponential at $\Delta t_K = 10^{-4}$ s.

**Prediction:**
$$\Delta\hat{x}_{k|k-1} = A_d\,\Delta\hat{x}_{k-1|k-1} + B_d\,\Delta u_k$$
$$P_{k|k-1} = A_d\,P_{k-1|k-1}\,A_d^\top + Q$$

**Correction:**
$$K_k = P_{k|k-1}\,C^\top\,(C\,P_{k|k-1}\,C^\top + R)^{-1}$$
$$\Delta\hat{x}_{k|k} = \Delta\hat{x}_{k|k-1} + K_k\,\tilde{y}_k$$

**Noise tuning:**
- Measurement noise $R$: diagonal, $\sigma_i = 1\%$ of mean signal amplitude.
- Process noise $Q$: small values ($10^{-6}$ to $10^{-4}$) for physical states; elevated ($10^{-4}$) for augmented load states to account for their unknown dynamics.

### Scenario 1 — Gauss-Markov Load Disturbances

Loads follow an AR(1) process with $\sigma = 25$ MW and $\tau = 50$ s. The filter reduces the relative estimation error on loads from ~1% (raw measurement noise) down to **0.17% on $P_{L1}$** and **0.12% on $P_{L2}$**.

### Scenario 2 — Line Trip (Time-Varying Kalman Filter)

At $t = 2$ s, one tie-line conductor trips, halving $K_L$. The filter is extended to a **Linear Time-Varying (LTV)** formulation: at the fault instant, system matrices, equilibrium point, and the error covariance (boosted by ×50) are all reset. The filter successfully tracks post-fault transients with relative errors below **0.7%** on power flows and speeds.

---

## Per-Window RTS Smoother

In the real application, state estimator outputs are available every $T_g = 1$ s while the filter runs at $\Delta t = 10^{-4}$ s. The classical full-batch RTS smoother is unsuitable for near-real-time use.

The implemented **Per-Window RTS** smoother divides the trajectory into successive $N_w = T_g/\Delta t = 10\,000$-step windows. Each window is anchored by a real measurement (lower noise $\sigma = 0.1\%$); intermediate steps use interpolated measurements ($\sigma = 1\%$). A single RTS backward sweep is triggered at the end of each window, introducing a fixed latency of exactly $T_g = 1$ s.

**RTS backward pass:**
$$G_k = P_{k|k}\,A_d^\top\,(P_{k+1|k})^{-1}$$
$$\Delta\hat{x}^s_k = \Delta\hat{x}_{k|k} + G_k\left(\Delta\hat{x}^s_{k+1} - \Delta\hat{x}_{k+1|k}\right)$$

**Smoother gains (vs. Kalman alone):**

| Signal | Kalman error | Smoothed error | Gain |
|--------|-------------|----------------|------|
| $P_{L1}$ | 0.179% | 0.123% | +31% |
| $P_{L2}$ | 0.109% | 0.081% | +26% |
| $\theta_1$, $\theta_2$ | ~0.113% | ~0.082% | +29% |
| $P_{G1}$ | 0.300% | 0.264% | +12% |

The smoother is most beneficial for slowly-varying channels (loads, angles) where forward-pass drift accumulates between anchor points. It provides no benefit — and slightly degrades — fast-varying states like $T_{m1}$ and $T_{m2}$.

---

## Dependencies

```bash
pip install numpy scipy sympy matplotlib jupyterlab
```

---

## Usage

```bash
# 1. Observability analysis
cd "Model and Observability"
python observability.py

# 2. Kalman filter — random load scenario
jupyter notebook "Filtre_de_Kalman/Kalman_PowerSystem.ipynb"

# 3. Kalman filter — line trip scenario
jupyter notebook "Filtre_de_Kalman/Kalman_PowerSystem_LineTrip.ipynb"

# 4. Per-window RTS smoother
jupyter notebook "Filtre_de_Kalman/Per_Window_RTS_Smoother.ipynb"
```

---

## Results Summary

| Scenario | Signal | Relative Error |
|----------|--------|---------------|
| Gauss-Markov loads | $P_{L1}$, $P_{L2}$ | 0.17%, 0.12% |
| Gauss-Markov loads | $P_{G1}$, $P_{G2}$ | 0.31%, 0.47% |
| Line Trip | $P_{G1}$, $P_{G2}$ | 0.20%, 0.30% |
| Line Trip | $\omega_1$, $\omega_2$ | 0.0002%, 0.0004% |
| Smoother | $P_{L1}$ | 0.12% (−31% vs Kalman) |

---

## References

1. V. Basetti et al. — *Power System Dynamic State Estimation using Prediction-Based Evolutionary Technique*, Energy, 2016.
2. M. A. Mahmud et al. — *Power System Dynamic State Estimation Using Extended and Unscented Kalman Filters*, 2020.
3. M. Taimah et al. — *Power System Dynamic State Estimation Based on Kalman Filter*, IJCA, 2016.
4. J. Zhang et al. — *A Two-Stage Kalman Filter Approach for Robust and Real-Time Power System State Estimation*, IEEE Trans. Sustainable Energy, 2014.
