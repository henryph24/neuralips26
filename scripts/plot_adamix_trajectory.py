"""T2.A visualization: plot per-step routing / gradient trajectories
produced by ``run_adamix.py --trajectory ...``.

Reads two JSONL files (an "unfrozen" collapse case and a "frozen" control)
and produces a single two-panel figure:

    Panel (a): routing entropy over optimizer steps, one line per setting.
    Panel (b): per-expert gradient L2 norm over steps, collapse case only
               (to show that one expert's gradient dominates and the others
               rapidly starve). A dashed line gives the sum of the other 4
               experts for reference.

Usage::

    python scripts/plot_adamix_trajectory.py \
        --collapse results/adamix/trajectory_ETTh1_last4_42.jsonl \
        --control  results/adamix/trajectory_ETTh1_frozen_42.jsonl \
        --out figures/adamix_trajectory.pdf

The output PDF is intended to be dropped into ``main.tex`` as a standalone
figure between Tables 4 and 5, upgrading the gradient co-adaptation claim
from correlational (Table 4 entropies at epoch end) to mechanistic.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _smooth(y, window=5):
    """Centered moving average; preserves length, ignores zeros at endpoints."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    y_padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(y_padded, kernel, mode="valid")[: len(y)]


def plot_trajectory(collapse_path, control_path, out_path, K=5):
    collapse = load_jsonl(collapse_path)
    control = load_jsonl(control_path) if control_path else None

    log_max = math.log(K)

    # Match LaTeX paper text: serif family, sized for NeurIPS column width.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), dpi=150)

    # --- Panel (a): routing entropy over steps ---
    ax = axes[0]
    steps_c = [r["step"] for r in collapse]
    ent_c = [r["routing_entropy"] for r in collapse]
    ax.plot(steps_c, ent_c, color="#d62728", linewidth=2.0,
            label="Last-4 unfrozen (collapses)")
    if control is not None:
        steps_f = [r["step"] for r in control]
        ent_f = [r["routing_entropy"] for r in control]
        ax.plot(steps_f, ent_f, color="#1f77b4", linewidth=2.0,
                label="Strictly frozen (control)")
    ax.axhline(log_max, color="gray", linestyle=":", linewidth=1.0)
    ax.text(395, log_max - 0.04, r"$\log K = 1.609$",
            fontsize=8, color="gray", ha="right", va="top", style="italic")
    # Annotate the collapse event
    ax.annotate("collapses\n(step ~40)", xy=(40, 0.02), xytext=(120, 0.35),
                fontsize=8.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Router entropy (nats)")
    ax.set_ylim(-0.05, log_max + 0.15)
    ax.set_xlim(0, max(steps_c))
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="gray", fancybox=False)
    ax.set_title("(a) Router entropy", fontsize=10, loc="left", pad=6)

    # --- Panel (b): grad norm of dominant vs starved experts ---
    ax = axes[1]
    labels = ["Mean", "Last", "Max", "Attn", "Conv1d"]
    grad_matrix = np.array([r["expert_grad_norms"] for r in collapse])  # (T, K)
    weights_matrix = np.array([r["mean_routing_weights"] for r in collapse])  # (T, K)
    # The eventually-dominant expert is the argmax of routing weight at the final step.
    dom_idx = int(np.argmax(weights_matrix[-1]))
    other_idx = [i for i in range(K) if i != dom_idx]

    # Floor to avoid log(0); below this is numerical noise (Adam's epsilon).
    floor = 1e-3
    dom = np.maximum(grad_matrix[:, dom_idx], floor)
    others = np.maximum(grad_matrix[:, other_idx], floor)
    others_mean = others.mean(axis=1)

    dom_smooth = _smooth(dom, window=5)
    others_smooth = _smooth(others_mean, window=5)

    # Shade the band of individual non-dominant experts to convey the spread
    # without the visual noise of 4 dashed lines.
    others_lo = others.min(axis=1)
    others_hi = others.max(axis=1)
    ax.fill_between(steps_c, np.maximum(others_lo, floor), others_hi,
                    color="#1f77b4", alpha=0.12,
                    label="Other experts (range)")
    ax.plot(steps_c, others_smooth, color="#1f77b4", linewidth=1.8,
            label=f"Other experts (mean of 4)")
    ax.plot(steps_c, dom_smooth, color="#ff7f0e", linewidth=2.4,
            label=f"Dominant: {labels[dom_idx]}")

    # Annotate the gradient starvation
    starve_step = int(np.argmax(others_mean[10:] < 0.01)) + 10
    if 10 < starve_step < len(steps_c) - 10:
        ax.annotate("others starve",
                    xy=(starve_step, others_mean[starve_step] + floor),
                    xytext=(starve_step + 80, 0.05),
                    fontsize=8.5, color="#1f77b4",
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"$\|\nabla_{\phi_k}\mathcal{L}\|_2$")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, max(grad_matrix.max(), 10) * 2)
    ax.set_xlim(0, max(steps_c))
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, which="major")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="gray", fancybox=False)
    ax.set_title("(b) Per-expert gradient norm (log)", fontsize=10, loc="left", pad=6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    pdf_path = out_path
    if not out_path.endswith(".pdf"):
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    print("Saved: %s (dominant expert: %s, idx=%d)" % (out_path, labels[dom_idx], dom_idx))
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collapse", required=True,
                   help="JSONL trajectory for an unfrozen / collapse condition")
    p.add_argument("--control", default=None,
                   help="JSONL trajectory for the frozen / control condition")
    p.add_argument("--out", default="figures/adamix_trajectory.pdf")
    p.add_argument("--K", type=int, default=5)
    args = p.parse_args()

    plot_trajectory(args.collapse, args.control, args.out, K=args.K)


if __name__ == "__main__":
    main()
