#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Luke Snow, August 21 2025

"""
Created on Mon Aug 18 10:35:28 2025

@author: lukesnow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inverse Multi-Objective Optimization — Algorithm 1 (Python)
Translation of the provided MATLAB code.

Dependencies:
    pip install numpy scipy matplotlib
"""

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize, linprog, LinearConstraint, Bounds
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
def ecdf(values):
    """Empirical CDF: returns (F(x), x) like MATLAB's [f, x] = ecdf(M_)."""
    x = np.sort(values)
    n = x.size
    f = np.arange(1, n + 1, dtype=float) / n
    return f, x


# ----------------------------
# Config and storage
# ----------------------------
rng = default_rng(12345)

N = 10                           # number of probes/responses per run
NV = 40                          # number of noise variance bins
NM = 0.1                         # max noise variance
noise_v = np.linspace(0.001, NM, NV)

statistic = np.zeros(NV)         # per-variance statistic for current tt
fval_store = np.zeros(NV)

stat = np.zeros((2, NV))         # [coordinating; non-coordinating]

MC = 300                         # Monte Carlo runs
statmc = np.zeros_like(stat)     # accumulates across MC
statmc_c = np.zeros((MC, NV))    # per-MC, coordinating
statmc_nc = np.zeros((MC, NV))   # per-MC, non-coordinating

# Initialize alpha probes for all t
alph = np.zeros((N, 2))
for k in range(N):
    alph[k, :] = [0.1 + rng.random(), 0.1 + rng.random()]  # U[0.1, 1.1]


# ----------------------------
# Main loops
# ----------------------------
for mc in range(MC):
    for tt in range(2):  # 0: coordinating (tt==1 in MATLAB), 1: non-coordinating
        for n in range(NV):
            # ----------------------------
            # Gaussian noise characteristics for this variance level
            # ----------------------------
            var = noise_v[n]
            Al = np.zeros((N, 2))     # records probes
            Bet = np.zeros((N, 2))    # records responses
            if tt == 0:
                Bet_store = np.zeros_like(Bet)

            # ----------------------------
            # Compute empirical distribution of M
            # M = max_{t≠s} a_t^T (eps_t - eps_s)
            # ----------------------------
            L = 500
            eps = np.zeros((2, N))
            M_samples = np.zeros(L)
            for l in range(L):
                for t in range(N):
                    Al[t, :] = alph[t, :]                   # record probe
                    eps[:, t] = rng.normal(0.0, np.sqrt(var), size=2)  # noise ~ N(0, var I)

                # compute max over t≠s of a_t^T (eps_t - eps_s)
                M = -np.inf
                for t in range(N):
                    for s in range(N):
                        if t == s:
                            continue
                        test = Al[t, :].dot(eps[:, t] - eps[:, s])
                        if test > M:
                            M = test
                M_samples[l] = M

            ecdff, ecdfx = ecdf(M_samples)

            # ----------------------------
            # Generate probe/response dataset
            # ----------------------------
            # Constraint form: alph[k]·x <= 1,  x >= 0
            for k in range(N):
                if tt == 0:
                    # Coordinating: solve (convex) scalarization of 3-objective utility, then add noise
                    A = alph[k, :].copy()
                    B_budget = 1.0

                    def fun(x):
                        # fun1 = x1*x2; fun2 = x1 + x2; fun3 = sqrt(x1)*x2
                        # Minimize negative weighted sum (1/3 each)
                        x1, x2 = x
                        return -( (x1 * x2) + (x1 + x2) + (np.sqrt(max(x1, 0.0)) * x2) ) / 3.0

                    # Linear inequality: A x <= B_budget
                    A_ub = A.reshape(1, 2)
                    b_ub = np.array([B_budget], dtype=float)
                    lin_con = LinearConstraint(A_ub, -np.inf, b_ub)

                    # Bounds x >= 0
                    bnds = Bounds([0.0, 0.0], [np.inf, np.inf])

                    # Start at zero (on boundary); SLSQP can handle it for this problem
                    res = minimize(fun, x0=np.zeros(2), method="SLSQP",
                                   bounds=bnds, constraints=[lin_con],
                                   options={"ftol": 1e-9, "maxiter": 200})
                    if not res.success:
                        raise RuntimeError(f"fmincon-equivalent failed at mc={mc}, n={n}, k={k}: {res.message}")

                    # Noisy "coordinating" response
                    prn = res.x + rng.normal(0.0, np.sqrt(var), size=2)

                    Bet[k, :] = prn
                    Bet_store[k, :] = prn  # keep for possible inspection
                else:
                    # Non-coordinating: random (scaled to typical A'x<=B range) + noise
                    B_budget = 1.0
                    Bet[k, 0] = 2 * B_budget * rng.random() + rng.normal(0.0, np.sqrt(var))
                    Bet[k, 1] = 2 * B_budget * rng.random() + rng.normal(0.0, np.sqrt(var))

            # ----------------------------
            # Linear program (Afriat-like statistic)
            # Minimize phi s.t. for all s≠t:
            #   u_s - u_t - lam_t * (a_t^T (Bet_s - Bet_t)) - phi <= 0
            # with u >= 1, lam >= 1, phi >= 0
            # ----------------------------
            # Variables z = [u_1..u_N, lam_1..lam_N, phi]  (length 2N+1)
            num_vars = 2 * N + 1
            c = np.zeros(num_vars)
            c[-1] = 1.0  # minimize phi

            A_ub_lp = np.zeros((N * N - N, num_vars))  # exclude t==s
            b_ub_lp = np.zeros(N * N - N)
            row = 0
            for s in range(N):
                for t in range(N):
                    if t == s:
                        continue
                    # c_st = a_t^T (Bet_s - Bet_t)
                    c_st = float(Al[t, :].dot(Bet[s, :] - Bet[t, :]))

                    # u_s - u_t - lam_t * c_st - phi <= 0
                    A_ub_lp[row, s] = 1.0             # +u_s
                    A_ub_lp[row, t] = -1.0            # -u_t
                    A_ub_lp[row, N + t] = -c_st       # -lam_t * c_st
                    A_ub_lp[row, 2 * N] = -1.0        # -phi
                    b_ub_lp[row] = 0.0
                    row += 1

            bounds = [(1.0, None)] * N + [(1.0, None)] * N + [(0.0, None)]  # u>=1, lam>=1, phi>=0

            lp = linprog(c, A_ub=A_ub_lp, b_ub=b_ub_lp, bounds=bounds, method="highs")
            if not lp.success:
                raise RuntimeError(f"linprog failed at mc={mc}, n={n}, tt={tt}: {lp.message}")

            fval = lp.x[-1]  # optimal phi
            fval_store[n] = fval

            # Evaluate ECDF at fval: find nearest ecdfx point
            ind = np.argmin(np.abs(ecdfx - fval))
            statistic[n] = 1.0 - ecdff[ind]

        # Save stats for this regime
        stat[tt, :] = statistic

    # Accumulate across MC and store per-MC traces
    statmc += stat
    statmc_c[mc, :] = stat[0, :]
    statmc_nc[mc, :] = stat[1, :]

    print(f"MC: {mc+1}/{MC}")

# Average over MC
statmc /= MC

# ----------------------------
# Plots
# ----------------------------
plt.figure()
plt.plot(noise_v, statmc[0, :], 'b', label='coordinating network')
plt.plot(noise_v, statmc[1, :], 'r', label='non-coordinating network')
plt.xlabel('Noise variance σ²')
plt.ylabel('1 - F̂_Ψ(Φ*)')
plt.legend(loc='best')
plt.title('Average statistic vs noise variance')
plt.tight_layout()

# Errorbar plot (using variance)
mean_c = statmc_c.mean(axis=0)
var_c = statmc_c.var(axis=0, ddof=1)  # sample variance
mean_nc = statmc_nc.mean(axis=0)
var_nc = statmc_nc.var(axis=0, ddof=1)

plt.figure()
plt.errorbar(noise_v, mean_c, yerr=var_c, fmt='-ob', label='coordinating (mean ± var)')
plt.errorbar(noise_v, mean_nc, yerr=var_nc, fmt='-or', label='non-coordinating (mean ± var)')
plt.xlabel('Noise variance σ²')
plt.ylabel('Statistic (mean ± var)')
plt.legend(loc='best')
plt.title('Error bars with variance')
plt.tight_layout()

# Shaded "error bars" (mean ± var)
plt.figure()
plt.plot(noise_v, mean_c, 'b', label='coordinating (mean)')
plt.fill_between(noise_v, mean_c - var_c, mean_c + var_c, color='b', alpha=0.33)
plt.plot(noise_v, mean_nc, 'r', label='non-coordinating (mean)')
plt.fill_between(noise_v, mean_nc - var_nc, mean_nc + var_nc, color='r', alpha=0.33)
plt.xlabel('Noise variance σ²')
plt.ylabel('Statistic (mean ± var)')
plt.legend(loc='best')
plt.title('Shaded variance bands')
plt.tight_layout()

plt.show()
