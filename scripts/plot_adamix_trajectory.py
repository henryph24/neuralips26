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


def plot_trajectory(collapse_path, control_path, out_path, K=5):
    collapse = load_jsonl(collapse_path)
    control = load_jsonl(control_path) if control_path else None

    max_steps = max(len(collapse), len(control) if control else 0)
    log_max = math.log(K)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), dpi=150)

    # --- Panel (a): routing entropy over steps ---
    ax = axes[0]
    steps_c = [r["step"] for r in collapse]
    ent_c = [r["routing_entropy"] for r in collapse]
    ax.plot(steps_c, ent_c, color="tab:red", linewidth=2,
            label="AdaMix, last-4 unfrozen (ETTh1)")
    if control is not None:
        steps_f = [r["step"] for r in control]
        ent_f = [r["routing_entropy"] for r in control]
        ax.plot(steps_f, ent_f, color="tab:blue", linewidth=2,
                label="AdaMix, strictly frozen (ETTh1, control)")
    ax.axhline(log_max, color="gray", linestyle=":", linewidth=1,
               label=r"$\log K = \log 5$ (uniform max)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Router entropy (nats)")
    ax.set_title("(a) Router entropy trajectory")
    ax.set_ylim(-0.05, log_max + 0.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=7, frameon=True)

    # --- Panel (b): per-expert grad norm for the collapse case ---
    ax = axes[1]
    grad_matrix = np.array([r["expert_grad_norms"] for r in collapse])  # (T, K)
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
    labels = ["Mean", "Last", "Max", "Attn", "Conv1d"]
    # Order by peak gradient norm -> the dominant expert becomes the red line
    peak_idx = np.argsort(-grad_matrix.max(axis=0))
    for rank, k in enumerate(peak_idx):
        style = "-" if rank == 0 else "--"
        lw = 2.0 if rank == 0 else 1.0
        alpha = 1.0 if rank == 0 else 0.7
        ax.plot(steps_c, grad_matrix[:, k], color=colors[rank % len(colors)],
                linestyle=style, linewidth=lw, alpha=alpha,
                label=("%s (dominant)" % labels[k]) if rank == 0 else labels[k])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"$\|\nabla_{\phi_k}\mathcal{L}\|_2$")
    ax.set_title("(b) Per-expert gradient norm (last-4 unfrozen)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=7, frameon=True, ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    pdf_path = out_path
    if not out_path.endswith(".pdf"):
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    print("Saved: %s" % out_path)
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
