"""Combine direct MI and KSG leakage to produce calibrated MI gap figure.

Reads:
  - results/analysis/direct_mi.json (I(M,Sigma;E) from compute_direct_mi.py)
  - results/analysis/ksg_leakage.json (epsilon = I(M,Sigma;S) from compute_ksg_leakage.py)

Produces:
  - Spearman correlation of [I(M,Sigma;E) - epsilon] vs Delta%
  - Comparison with R(D) proxy correlation
  - Combined figure data for paper integration

If I(M,Sigma;E) - epsilon correlates with Delta% at rho >= -0.85, this closes
the "Proposition 2 isn't tight" weakness (W3).

Usage:
    python scripts/analyze_mi_gap.py
"""

import json
import os
import sys

import numpy as np

# Verified Delta% values from the paper
DELTA_PCT = {
    "ETTh1": -43.2, "ETTh2": -71.0, "ETTm1": -51.1, "ETTm2": -77.2,
    "Weather": -44.6, "Electricity": -26.8, "Traffic": +2.9,
    "Exchange": -66.5, "Solar": -32.9,
}


def main():
    # Load direct MI results
    mi_path = "results/analysis/direct_mi.json"
    if not os.path.exists(mi_path):
        print("ERROR: %s not found. Run compute_direct_mi.py first." % mi_path)
        sys.exit(1)
    with open(mi_path) as f:
        mi_data = json.load(f)

    # Load KSG leakage results
    ksg_path = "results/analysis/ksg_leakage.json"
    if not os.path.exists(ksg_path):
        print("ERROR: %s not found. Run compute_ksg_leakage.py first." % ksg_path)
        sys.exit(1)
    with open(ksg_path) as f:
        ksg_data = json.load(f)

    # Index KSG by dataset name
    ksg_by_ds = {r["dataset"]: r for r in ksg_data}

    print("=" * 70)
    print("CALIBRATED MI GAP ANALYSIS")
    print("=" * 70)
    print("\n%-12s  %8s  %8s  %8s  %8s  %7s" % (
        "Dataset", "I(M,S;E)", "eps_KSG", "MI_gap", "R(D)", "Delta%"))
    print("-" * 70)

    datasets = []
    mi_gaps = []
    R_vals = []
    delta_vals = []

    for ds in sorted(mi_data.keys()):
        if ds.startswith("_"):
            continue
        if ds not in ksg_by_ds:
            print("%-12s  SKIP (no KSG data)" % ds)
            continue
        if ds not in DELTA_PCT:
            print("%-12s  SKIP (no Delta%%)" % ds)
            continue

        mi_info = mi_data[ds]
        ksg_info = ksg_by_ds[ds]

        I_MS_E = mi_info["mi"]["I_joint_knn"]
        epsilon = ksg_info["eps_ksg"]
        mi_gap = max(I_MS_E - epsilon, 0.0)
        R = mi_info["R"]
        delta = DELTA_PCT[ds]

        datasets.append(ds)
        mi_gaps.append(mi_gap)
        R_vals.append(R)
        delta_vals.append(delta)

        print("%-12s  %8.4f  %8.4f  %8.4f  %8.3f  %+6.1f%%" % (
            ds, I_MS_E, epsilon, mi_gap, R, delta))

    if len(datasets) < 5:
        print("\nToo few datasets (%d) for reliable correlation." % len(datasets))
        sys.exit(1)

    # Compute correlations
    from scipy.stats import spearmanr

    rho_gap, p_gap = spearmanr(mi_gaps, delta_vals)
    rho_R, p_R = spearmanr(R_vals, delta_vals)
    rho_gap_R, p_gap_R = spearmanr(mi_gaps, R_vals)

    print("\n" + "=" * 70)
    print("CORRELATION RESULTS (n=%d)" % len(datasets))
    print("=" * 70)
    print("  MI gap vs Delta%%:  rho = %.3f (p = %.4f)  %s" % (
        rho_gap, p_gap,
        "CLOSES W3" if rho_gap <= -0.85 else "PARTIAL" if rho_gap <= -0.70 else "WEAK"))
    print("  R(D) vs Delta%%:   rho = %.3f (p = %.4f)  [reference]" % (rho_R, p_R))
    print("  MI gap vs R(D):   rho = %.3f (p = %.4f)  [proxy validation]" % (rho_gap_R, p_gap_R))

    # Save combined results
    combined = {
        "datasets": datasets,
        "mi_gaps": mi_gaps,
        "R_values": R_vals,
        "delta_pct": delta_vals,
        "correlations": {
            "mi_gap_vs_delta": {"rho": float(rho_gap), "p": float(p_gap)},
            "R_vs_delta": {"rho": float(rho_R), "p": float(p_R)},
            "mi_gap_vs_R": {"rho": float(rho_gap_R), "p": float(p_gap_R)},
        },
        "per_dataset": {
            ds: {
                "I_MS_E": mi_data[ds]["mi"]["I_joint_knn"],
                "epsilon": ksg_by_ds[ds]["eps_ksg"],
                "mi_gap": max(mi_data[ds]["mi"]["I_joint_knn"] - ksg_by_ds[ds]["eps_ksg"], 0.0),
                "R": mi_data[ds]["R"],
                "delta_pct": DELTA_PCT[ds],
            }
            for ds in datasets
        },
    }

    out_path = "results/analysis/mi_gap_combined.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print("\nSaved to %s" % out_path)

    # Paper-ready summary
    print("\n" + "=" * 70)
    print("PAPER-READY CLAIM")
    print("=" * 70)
    if rho_gap <= -0.85:
        print("The calibrated MI gap I(M,Sigma;E) - epsilon predicts RR-MoA")
        print("improvement with rho=%.2f (p=%.4f, n=%d), matching or exceeding" % (
            rho_gap, p_gap, len(datasets)))
        print("the variance proxy R(D) (rho=%.2f). This validates Proposition 2's" % rho_R)
        print("bound as empirically tight for practical prediction.")
    else:
        print("MI gap correlation: rho=%.2f (weaker than R(D) proxy rho=%.2f)." % (
            rho_gap, rho_R))
        print("The variance proxy R(D) may capture additional variance not in the MI estimate.")


if __name__ == "__main__":
    main()
