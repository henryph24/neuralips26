"""Bootstrap confidence interval for routing signal ratio correlation.

Addresses reviewer concern about small n=7 for Spearman rho=-0.96.
Reports 95% CI via bootstrap resampling.

Usage:
    python scripts/bootstrap_correlation.py
"""

import numpy as np
from scipy import stats

# Data from Figure 3: (routing_signal_ratio R, RR-MoA improvement delta%)
# 9 datasets, computed from analyze_routing_signal_ratio.py
data = [
    ("Solar",        0.061, -35.0),
    ("Traffic",      0.135,  +2.9),
    ("Electricity",  0.242, -26.8),
    ("ETTh1",        0.566, -43.2),
    ("ETTm1",        0.671, -51.1),
    ("Weather",      0.703, -44.6),
    ("ETTh2",        1.163, -71.0),
    ("ETTm2",        1.241, -77.2),
    ("Exchange",     2.174, -68.5),
]

names = [d[0] for d in data]
R = np.array([d[1] for d in data])
delta = np.array([d[2] for d in data])

# Point estimate
rho, p = stats.spearmanr(R, delta)
print("=== Point Estimate ===")
print("Spearman rho = %.4f, p = %.6f (n=%d)" % (rho, p, len(R)))

# Bootstrap CI
n_bootstrap = 10000
np.random.seed(42)
rhos = []
for _ in range(n_bootstrap):
    idx = np.random.choice(len(R), size=len(R), replace=True)
    if len(set(idx)) < 3:
        continue  # skip degenerate resamples
    r, _ = stats.spearmanr(R[idx], delta[idx])
    if not np.isnan(r):
        rhos.append(r)

rhos = np.array(rhos)
ci_lower = np.percentile(rhos, 2.5)
ci_upper = np.percentile(rhos, 97.5)

print("\n=== Bootstrap (n=%d) ===" % n_bootstrap)
print("95%% CI: [%.4f, %.4f]" % (ci_lower, ci_upper))
print("Mean rho: %.4f" % np.mean(rhos))
print("Std rho: %.4f" % np.std(rhos))

# Permutation test (more robust than parametric p-value for small n)
n_perm = 100000
np.random.seed(42)
perm_rhos = []
for _ in range(n_perm):
    perm_delta = np.random.permutation(delta)
    r, _ = stats.spearmanr(R, perm_delta)
    perm_rhos.append(r)

perm_rhos = np.array(perm_rhos)
perm_p = np.mean(np.abs(perm_rhos) >= np.abs(rho))

print("\n=== Permutation Test (n=%d) ===" % n_perm)
print("Two-sided p = %.6f" % perm_p)
print("(Fraction of permutations with |rho| >= |%.4f|)" % rho)

# Leave-one-out stability
print("\n=== Leave-One-Out Stability ===")
for i in range(len(R)):
    R_loo = np.delete(R, i)
    d_loo = np.delete(delta, i)
    r_loo, _ = stats.spearmanr(R_loo, d_loo)
    print("  Drop %-12s: rho = %.4f  (n=%d)" % (names[i], r_loo, len(R_loo)))

print("\n=== LaTeX for paper ===")
print("Spearman $\\rho = %.2f$ ($p < %.3f$; 95\\%% bootstrap CI: $[%.2f, %.2f]$, $n{=}%d$; permutation $p = %.4f$)" % (
    rho, p, ci_lower, ci_upper, len(R), perm_p))
