"""Plot MI tightness analysis for Proposition 2.

Generates two figures:
1. H(E|S) [exact MI loss] vs Delta% scatter with correlation annotation
2. Stacked bar: lower bound + gap = exact MI loss per dataset

Usage:
    python scripts/plot_mi_tightness.py
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

LOG_K = math.log(5)

# Paper-quality defaults
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_results():
    path = "results/analysis/exact_mi_loss.json"
    with open(path) as f:
        data = json.load(f)
    # Filter out meta keys
    datasets = {k: v for k, v in data.items() if not k.startswith("_")}
    correlations = data.get("_correlations", {})
    return datasets, correlations


def plot_exact_vs_delta(datasets, correlations, outpath):
    """Scatter: H(E|S) vs Delta% across datasets."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ds_names = sorted(datasets.keys())
    x = [datasets[d]["exact_mi_loss_mean"] for d in ds_names]
    y = [datasets[d]["delta_pct"] for d in ds_names]
    xerr = [datasets[d]["exact_mi_loss_std"] for d in ds_names]

    ax.errorbar(x, y, xerr=xerr, fmt="o", markersize=5, color="#2563eb",
                ecolor="#93c5fd", capsize=3, linewidth=1)

    for i, d in enumerate(ds_names):
        offset = (5, 5) if y[i] < -40 else (5, -10)
        ax.annotate(d, (x[i], y[i]), textcoords="offset points",
                    xytext=offset, fontsize=7, color="#4b5563")

    rho = correlations.get("rho_exact_delta", float("nan"))
    p = correlations.get("p_exact_delta", float("nan"))
    ax.text(0.05, 0.95, r"$\rho = %.2f$ ($p = %.3f$)" % (rho, p),
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", alpha=0.8))

    ax.set_xlabel(r"$H(E \mid S)$ [exact MI loss, nats]")
    ax.set_ylabel(r"$\Delta\%$ (RR-MoA vs AdaMix)")
    ax.axhline(0, color="#d1d5db", linewidth=0.5, linestyle="--")
    ax.set_title("Exact MI Loss Predicts Improvement")

    fig.savefig(outpath)
    plt.close(fig)
    print("Saved %s" % outpath)


def plot_stacked_bar(datasets, outpath):
    """Stacked bar: LB (blue) + gap (red) = exact (total)."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    ds_names = sorted(datasets.keys())
    x_pos = np.arange(len(ds_names))

    lb = [datasets[d]["lower_bound_mean"] for d in ds_names]
    gap = [datasets[d]["I_MS_S_given_E_mean"] for d in ds_names]
    exact = [datasets[d]["exact_mi_loss_mean"] for d in ds_names]

    ax.bar(x_pos, lb, color="#3b82f6", label="Lower bound", width=0.6)
    ax.bar(x_pos, gap, bottom=lb, color="#ef4444", alpha=0.7,
           label=r"Gap $I(M,\Sigma; S \mid E)$", width=0.6)

    # Mark exact value
    ax.scatter(x_pos, exact, marker="_", color="black", s=80, zorder=5,
               label=r"$H(E \mid S)$ [exact]", linewidths=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MI loss (nats)")
    ax.axhline(LOG_K, color="#9ca3af", linewidth=0.5, linestyle=":",
               label=r"$\log K = %.2f$" % LOG_K)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Bound Tightness Decomposition")

    fig.savefig(outpath)
    plt.close(fig)
    print("Saved %s" % outpath)


def plot_synthetic(outpath):
    """Plot synthetic sweep: exact loss and lower bound vs rho_MS."""
    path = "results/analysis/synthetic_mi_tightness.json"
    if not os.path.exists(path):
        print("No synthetic results at %s — skipping" % path)
        return

    with open(path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    rho = [r["rho_MS"] for r in results]
    exact = [r["exact_loss_mean"] for r in results]
    exact_std = [r["exact_loss_std"] for r in results]
    lb = [r["lower_bound_mean"] for r in results]
    lb_std = [r["lower_bound_std"] for r in results]

    ax.errorbar(rho, exact, yerr=exact_std, fmt="o-", markersize=4,
                color="#2563eb", label=r"$H(E \mid S)$ [exact]", linewidth=1.2)
    ax.errorbar(rho, lb, yerr=lb_std, fmt="s--", markersize=4,
                color="#ef4444", label="Lower bound (ii)", linewidth=1.2)

    # Shade the gap
    ax.fill_between(rho, lb, exact, alpha=0.15, color="#ef4444")

    ax.set_xlabel(r"$\rho_{MS}$ (shape-statistics correlation)")
    ax.set_ylabel("MI loss (nats)")
    ax.legend(fontsize=8)
    ax.set_title("Synthetic: Bound Tightness vs Correlation")

    fig.savefig(outpath)
    plt.close(fig)
    print("Saved %s" % outpath)


def main():
    os.makedirs("figures", exist_ok=True)

    datasets, correlations = load_results()

    plot_exact_vs_delta(datasets, correlations,
                        "figures/mi_exact_vs_delta.pdf")
    plot_stacked_bar(datasets,
                     "figures/mi_tightness_decomposition.pdf")
    plot_synthetic("figures/synthetic_mi_tightness.pdf")


if __name__ == "__main__":
    main()
