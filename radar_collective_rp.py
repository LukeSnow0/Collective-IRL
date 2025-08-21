#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Luke Snow, August 21 2025

"""
Collective Spectral Revealed Preferences – Radar Network
=======================================================

This script generates *assignable* per-agent consumption bundles for a
multi-agent system under a common "price/probe" vector at each time step,
then reconstructs each agent's utility via Afriat's Theorem (linear program)
*separately per agent*. Because we observe assignable quantities (each agent's
own response), agents are identified without label ambiguity.

Pipeline
--------
1) **Data generation (simulation):**
   - For each time t = 1..N, draw a probe vector a_t (think of "prices").
   - Solve a small nonlinear program to allocate resources across M agents,
     maximizing a weighted sum of known utility shapes (used only for simulation).
   - Constraint: sum_j a_t^T b_{j,t} <= y_t  (budget/resource cap).
   - Save each agent's bundle b_{j,t}. These are *assignable* observations.

2) **Reconstruction (Afriat per agent):**
   - For each agent j, solve an LP consistent with Afriat's inequalities to
     obtain levels u_{j,t} and multipliers λ_{j,t} that rationalize the observed
     choices { (a_t, b_{j,t}, y_t) }_t.
   - This yields a piecewise-linear concave utility representation for each agent.

3) **Diagnostics & plots:**
   - Report Afriat max violation (should be ~<= 0 up to numerical tolerance).
   - Plot reconstructed utility surfaces vs the "true" (simulated) shapes.

Why trust-constr?
-----------------
The generator utilities involve sqrt terms, which are non-differentiable at 0.
We smooth as sqrt(x + EPS) and enforce strictly positive bounds.
We use `trust-constr` with an analytic gradient and BFGS Hessian update
for stable convergence, with a fallback to SLSQP.

Dependencies
------------
- numpy
- scipy (optimize: trust-constr, BFGS, linprog)
- matplotlib

Conventions
-----------
- Shapes:
    * B: (2, M, N)    -> 2 goods, M agents, N time steps
    * A_mat: (2, N)   -> price/probe vectors per time step
- Budget constraint (MATLAB RPcon2): c = sum(a * x) - y <= 0
  -> SciPy "ineq" uses fun(z) >= 0, so we pass: y - sum(a @ x) >= 0.
- We never permute agent indices. Arrays are indexed by the *true* agent id.

Notes
-----
- EPS_ANCH adds a vanishingly small per-agent quadratic term to break numerical
  ties across agents during simulation. It does *not* change the economic
  interpretation; it only helps the optimizer pick a stable labeling.
"""

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize, Bounds, NonlinearConstraint, linprog, BFGS
import matplotlib.pyplot as plt

# ========= Config =========
RNG_SEED = 123
rng = default_rng(RNG_SEED)

contour_res = 40          # number of contour levels in plots
N = 50                    # number of time steps / observations
M = 3                     # number of agents
y_total = 1.0             # total available "budget" each period
EPS = 1e-8                # sqrt smoothing & strictly positive lower bound
EPS_ANCH = 1e-12          # tiny tie-breaker across agents in objective (harmless)

# ========= Storage =========
# B[:, j, t] stores the 2D bundle for agent j at time t (assignable)
B = np.zeros((2, M, N))   # b_{dim, agent, t}
SB = np.zeros((2, N))     # sum over agents at each t (aggregate)
A_mat = np.zeros((2, N))  # probe/price a_t per time step
y_vec = np.zeros(N)       # resource cap per time step

# ========= Utility weights (for data generation only) =========
# These weights weight the simulated agents' utilities in the generator.
# They do not affect the identification logic in reconstruction.
if M == 3:
    mu = np.array([1/3, 1/3, 1/3], dtype=float)
else:
    mu = np.array([0.5, 0.5], dtype=float)


# ========= Generator objective & gradient =========
def f_obj_and_grad(z, mu, M, eps=EPS):
    """
    Compute the *negative* of the weighted sum of simulated utilities and its
    gradient with respect to z = vec(x) where x has shape (2, M).

    For M == 3 (matches active MATLAB branch):
        Agent 1: U1(b) = (b1 * b2)^2
        Agent 2: U2(b) = b1 * sqrt(b2)
        Agent 3: U3(b) = sqrt(b1) * b2
    The solver *minimizes* f, hence we supply the negative of the sum.

    Smoothing:
        sqrt(x) is replaced by sqrt(x + eps) to avoid kinks at zero.

    Tie-breaker (EPS_ANCH):
        Adds a vanishingly small, agent-specific quadratic term that prevents
        numerical label ties. This does not change the economic solution.

    Parameters
    ----------
    z : np.ndarray, shape (2*M,)
        Flattened decision variable, x.reshape(-1).
    mu : np.ndarray, shape (M,)
        Per-agent weights used only for data generation.
    M : int
        Number of agents.
    eps : float
        Smoothing / positivity parameter.

    Returns
    -------
    f : float
        Objective value (to minimize).
    g_flat : np.ndarray, shape (2*M,)
        Gradient vector.
    """
    x = z.reshape(2, M)
    g = np.zeros_like(x)

    if M == 3:
        # aliases (columns are agents)
        x11, x21 = x[0, 0], x[1, 0]
        x12, x22 = x[0, 1], x[1, 1]
        x13, x23 = x[0, 2], x[1, 2]

        s22 = np.sqrt(x22 + eps)
        s13 = np.sqrt(x13 + eps)

        # objective (negative of weighted sum of utilities)
        f = -(
            mu[0] * (x11**2) * (x21**2) +
            mu[1] * (x12 * s22) +
            mu[2] * (s13 * x23)
        )

        # gradients (d/dx of the *negative* utility sum)
        g[0, 0] = -(mu[0] * 2.0 * x11 * (x21**2))                 # d/dx11
        g[1, 0] = -(mu[0] * 2.0 * x21 * (x11**2))                 # d/dx21
        g[0, 1] = -(mu[1] * s22)                                  # d/dx12
        g[1, 1] = -(mu[1] * x12 * (0.5 / max(s22, eps)))          # d/dx22
        g[0, 2] = -(mu[2] * x23 * (0.5 / max(s13, eps)))          # d/dx13
        g[1, 2] = -(mu[2] * s13)                                  # d/dx23

    else:
        # Simple two-agent variant (not used in the main experiments)
        f = -(mu[0] * x[0, 0] * x[1, 0] + mu[1] * x[0, 1] * x[1, 1])
        g[0, 0] = -(mu[0] * x[1, 0])
        g[1, 0] = -(mu[0] * x[0, 0])
        g[0, 1] = -(mu[1] * x[1, 1])
        g[1, 1] = -(mu[1] * x[0, 1])

    # Tiny per-agent anchor to avoid numerical label ties (does not change econ)
    if EPS_ANCH > 0:
        # Agent 1 anchor
        f -= EPS_ANCH * (1.0 * np.sum(x[:, 0]**2))
        g[:, 0] -= EPS_ANCH * 2.0 * x[:, 0]
        # Agent 2 anchor
        if M >= 2:
            f -= EPS_ANCH * (2.0 * np.sum(x[:, 1]**2))
            g[:, 1] -= EPS_ANCH * 4.0 * x[:, 1]
        # Agent 3 anchor
        if M >= 3:
            f -= EPS_ANCH * (3.0 * np.sum(x[:, 2]**2))
            g[:, 2] -= EPS_ANCH * 6.0 * x[:, 2]

    return f, g.reshape(-1)


# ========= RPcon2: y - sum(a @ x_j) >= 0 (SciPy 'ineq') =========
def make_budget_constraint(a, y, M):
    """
    Build the NonlinearConstraint for the budget/resource cap:

        sum_j a^T x_j <= y

    SciPy's "ineq" convention is fun(z) >= 0, so we supply:

        fun(z) = y - sum_j a^T x_j  >= 0.

    We also supply the (constant) Jacobian for faster convergence.

    Parameters
    ----------
    a : np.ndarray, shape (2,)
        Probe/price vector at a given time step.
    y : float
        Resource cap at this time step.
    M : int
        Number of agents.

    Returns
    -------
    nlc : scipy.optimize.NonlinearConstraint
        Inequality constraint suitable for trust-constr.
    """
    def fun(z):
        x = z.reshape(2, M)
        return y - float(np.sum(a @ x))  # >= 0  <=> sum(a*x) <= y

    def jac(z):
        # Derivative of -sum_j a^T x_j w.r.t. x = [-a[0], ..., -a[0], -a[1], ..., -a[1]]
        J = np.empty((1, 2 * M))
        J[0, :M] = -a[0]
        J[0, M:] = -a[1]
        return J

    return NonlinearConstraint(fun, lb=0.0, ub=np.inf, jac=jac)


# ========= Solve one observation's generator program =========
def solve_one_obs(alph, y_, mu, M):
    """
    Solve the per-time-step resource allocation given probe vector `alph`
    and resource cap `y_`, producing the assignable bundles x[:, j] per agent.

    We minimize the negative of the weighted sum of (simulated) agent utilities,
    subject to the budget constraint sum_j alph^T x_j <= y_ and x >= EPS.

    Numerical stability:
    - Start from an interior point that spends about 80% of the budget evenly.
    - Enforce strictly positive lower bounds (EPS).
    - Use trust-constr with analytic gradient and BFGS Hessian update.
    - Fallback to SLSQP if needed.

    Parameters
    ----------
    alph : np.ndarray, shape (2,)
        Probe/price vector at this time step.
    y_ : float
        Resource cap.
    mu : np.ndarray, shape (M,)
        Weights used only for *simulation* of the generator objective.
    M : int
        Number of agents.

    Returns
    -------
    x_opt : np.ndarray, shape (2, M)
        Optimal assignable bundles for all agents at this time step.
    """
    # Interior start: ~80% of budget, split across dims & agents
    x0 = np.zeros((2, M))
    x0[0, :] = 0.8 * y_ / (2.0 * alph[0] * M)
    x0[1, :] = 0.8 * y_ / (2.0 * alph[1] * M)
    z0 = np.clip(x0.reshape(-1), EPS, None)  # ensure positivity

    bounds = Bounds(lb=np.full(2 * M, EPS), ub=np.full(2 * M, np.inf))
    nlc = make_budget_constraint(alph, y_, M)

    def fun(z):
        f, _ = f_obj_and_grad(z, mu, M, EPS)
        return f

    def jac(z):
        _, g = f_obj_and_grad(z, mu, M, EPS)
        return g

    # Primary solver: trust-constr with BFGS Hessian update
    res = minimize(
        fun, z0, method="trust-constr",
        jac=jac, hess=BFGS(),
        bounds=bounds, constraints=[nlc],
        options={"maxiter": 500, "gtol": 1e-8, "verbose": 0}
    )

    if not res.success:
        # Fallback: SLSQP with same interior start & bounds
        res = minimize(
            fun, z0, method="SLSQP", jac=jac,
            bounds=list(zip(np.full(2 * M, EPS), np.full(2 * M, np.inf))),
            constraints=[{
                "type": "ineq",
                "fun": lambda z, a=alph, y=y_, M=M: y - float(np.sum(a @ z.reshape(2, M)))
            }],
            options={"ftol": 1e-9, "maxiter": 300}
        )
        if not res.success:
            raise RuntimeError(f"Nonlinear solve failed: {res.message}")

    return res.x.reshape(2, M)


# ========= Generate dataset (assignable per-agent bundles) =========
for t in range(N):
    # Draw a probe/price vector a_t with each component >= 0.1
    alph = np.array([rng.random() + 0.1, rng.random() + 0.1], dtype=float)
    A_mat[:, t] = alph
    y_ = y_total

    # Solve resource allocation to produce per-agent bundles at time t
    x_opt = solve_one_obs(alph, y_, mu, M)  # shape (2, M)
    B[:, :, t] = x_opt
    SB[:, t] = np.sum(x_opt, axis=1)       # aggregate spend (not used in LP)
    y_vec[t] = y_

# Freeze B for reconstruction (exactly as observed; do NOT reorder agents)
b_sol_ = B.copy()


# ========= Afriat reconstruction per agent (no permutations) =========
# For each agent j, we solve:
#     maximize  phi_j
#     subject to: for all t, s in {1..N}:
#         u_{j,t} <= u_{j,s} + λ_{j,s} * a_s^T (b_{j,t} - b_{j,s}) - phi_j
# with bounds λ_{j,s} >= 1, u_{j,t} >= 1, phi_j >= 0.
# We implement as a standard LP: minimize -phi_j subject to A_ub z <= b_ub.

lambda_sol = np.zeros((N, M))  # λ_{j,t}
u_sol = np.zeros((N, M))       # u_{j,t}

for agent in range(M):
    # Decision vector per agent: z = [λ_0..λ_{N-1}, u_0..u_{N-1}, phi]
    num_vars = 2 * N + 1
    c = np.zeros(num_vars)
    c[-1] = -1.0  # minimize -phi -> maximize phi

    # Build A_ub z <= b_ub from Afriat inequalities
    A_ub = np.zeros((N * N, num_vars))
    b_ub = np.zeros(N * N)

    row = 0
    for t in range(N):
        b_t = b_sol_[:, agent, t]
        for s in range(N):
            b_s = b_sol_[:, agent, s]
            c_st = float(A_mat[:, s] @ (b_t - b_s))  # a_s^T(b_t - b_s)

            # u_t - u_s - λ_s * c_st + phi <= 0
            A_ub[row, N + t] = 1.0       # +u_t
            A_ub[row, N + s] = -1.0      # -u_s
            A_ub[row, s] = -c_st         # -λ_s * c_st
            A_ub[row, -1] = 1.0          # +phi
            b_ub[row] = 0.0
            row += 1

    # Bounds: λ >= 1, u >= 1, phi >= 0
    bounds = [(1.0, None)] * N + [(1.0, None)] * N + [(0.0, None)]

    # Solve LP with HiGHS
    lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not lp.success:
        raise RuntimeError(f"LP failed for agent {agent}: {lp.message}")

    z = lp.x
    lambda_sol[:, agent] = z[:N]
    u_sol[:, agent] = z[N:2*N]
    # phi_j = z[-1] if you want to inspect it


# ========= Sanity: Afriat max violation per agent (should be ~<= 0) =========
def afriat_max_violation(A_mat, b_agent, u_j, lam_j):
    """
    Compute the maximum Afriat slack for a single agent:

        max_{t,s} [ u_t - u_s - λ_s * a_s^T (b_t - b_s) ]

    For a perfectly rationalizable dataset (and exact arithmetic), this
    maximum should be <= 0. In practice with finite precision, expect values
    on the order of 1e-9 to 1e-12.

    Parameters
    ----------
    A_mat : np.ndarray, shape (2, N)
        Probe vectors per time step.
    b_agent : np.ndarray, shape (2, N)
        Agent's assignable bundles over time.
    u_j : np.ndarray, shape (N,)
        Recovered utility levels at each time.
    lam_j : np.ndarray, shape (N,)
        Recovered multipliers at each time.

    Returns
    -------
    float
        Maximum violation (should be small and <= ~1e-9).
    """
    Nloc = b_agent.shape[1]
    maxvio = -np.inf
    for t in range(Nloc):
        for s in range(Nloc):
            lhs = u_j[t] - u_j[s] - lam_j[s] * (A_mat[:, s] @ (b_agent[:, t] - b_agent[:, s]))
            if lhs > maxvio:
                maxvio = lhs
    return float(maxvio)


# Print the violation per agent; large positive values indicate an issue.
for j in range(M):
    vio = afriat_max_violation(A_mat, b_sol_[:, j, :], u_sol[:, j], lambda_sol[:, j])
    print(f"Agent {j+1}: max Afriat violation = {vio:.3e}")


# ========= Plot reconstructed vs true utilities (fixed agent labels) =========
# Build a grid of candidate bundles (b1, b2) for visualization.
betaspace = np.linspace(0.0, 2.0, 20)
B1, B2 = np.meshgrid(betaspace, betaspace, indexing="ij")

# "True" utilities by agent (the generator's shapes; used for comparison only).
U_true_per_agent = []
if M == 3:
    U_true_per_agent.append((B1 * B2) ** 2)     # Agent 1 true utility
    U_true_per_agent.append(B1 * np.sqrt(B2))   # Agent 2 true utility
    U_true_per_agent.append(np.sqrt(B1) * B2)   # Agent 3 true utility
else:
    # For M != 3, adjust this block according to your generator choices.
    for _ in range(M):
        U_true_per_agent.append(B1 * B2)

# Reconstructed per agent (NO reordering; fixed agent labels).
for agent in range(M):
    # Evaluate the reconstructed piecewise-linear concave utility at grid points:
    # U(b) = min_t [ u_{agent,t} + λ_{agent,t} * a_t^T (b - b_{agent,t}) ]
    U_rec = np.empty_like(B1)
    for i1 in range(B1.shape[0]):
        for i2 in range(B2.shape[1]):
            b_vec = np.array([B1[i1, i2], B2[i1, i2]])
            vals = np.empty(N)
            for t in range(N):
                vals[t] = (
                    u_sol[t, agent]
                    + lambda_sol[t, agent] * (A_mat[:, t] @ (b_vec - b_sol_[:, agent, t]))
                )
            U_rec[i1, i2] = np.min(vals)

    # Plot reconstructed utility surface
    plt.figure()
    cs = plt.contourf(betaspace, betaspace, U_rec, levels=contour_res)
    plt.colorbar(cs)
    plt.title(f"Agent {agent+1} Reconstructed Utility")
    plt.xlabel("b1")
    plt.ylabel("b2")
    plt.tight_layout()

    # Plot true utility surface (for visual comparison)
    plt.figure()
    cs = plt.contourf(betaspace, betaspace, U_true_per_agent[agent], levels=contour_res)
    plt.colorbar(cs)
    plt.title(f"Agent {agent+1} True Utility")
    plt.xlabel("b1")
    plt.ylabel("b2")
    plt.tight_layout()

plt.show()
