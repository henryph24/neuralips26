"""Plot the causal contrast figure: MOMENT vs Timer-XL entropy trajectories.

4 lines on one panel:
  - MOMENT frozen (blue, dashed) — control, entropy stays moderate
  - MOMENT last-4 (red, solid) — collapse, entropy → 0
  - Timer-XL frozen (green, dashed) — control, entropy stays at max
  - Timer-XL last-4 (orange, solid) — NO collapse, entropy stays at max

Usage:
    python scripts/plot_causal_contrast.py --out figures/causal_contrast.pdf
"""

import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--moment-frozen", default="results/adamix/trajectory_ETTh1_frozen_42.jsonl")
    p.add_argument("--moment-last4", default="results/adamix/trajectory_ETTh1_last4_42.jsonl")
    p.add_argument("--timer-frozen", default="results/adamix_timer/trajectory_ETTh1_frozen_42_timer.jsonl")
    p.add_argument("--timer-last4", default="results/adamix_timer/trajectory_ETTh1_last4_42_timer.jsonl")
    p.add_argument("--out", default="figures/causal_contrast.pdf")
    p.add_argument("--K", type=int, default=5)
    args = p.parse_args()

    log_max = math.log(args.K)

    # Match LaTeX paper text: serif family, NeurIPS-style sizing.
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

    # Cool teal for Timer-XL so red and orange are not both warm hues.
    trajectories = {
        "MOMENT frozen": (args.moment_frozen, "#1f77b4", "--", 1.5),       # blue
        "MOMENT last-4 (RevIN)": (args.moment_last4, "#d62728", "-", 2.5),  # red
        "Timer-XL frozen": (args.timer_frozen, "#17becf", "--", 1.5),       # cyan-teal
        "Timer-XL last-4 (no RevIN)": (args.timer_last4, "#2ca02c", "-", 2.5),  # green
    }

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.6), dpi=150)

    for label, (path, color, ls, lw) in trajectories.items():
        data = load_jsonl(path)
        steps = [r["step"] for r in data]
        ent = [r["routing_entropy"] for r in data]
        ax.plot(steps, ent, color=color, linestyle=ls, linewidth=lw, label=label)

    ax.axhline(log_max, color="gray", linestyle=":", linewidth=0.9)
    # Inline label for the log K ceiling, anchored to the right edge.
    ax.text(395, log_max + 0.02, r"$\log K = 1.609$ (max entropy)",
            ha="right", va="bottom", fontsize=7.5, color="gray", style="italic")

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Router entropy (nats)")
    ax.set_ylim(-0.05, log_max + 0.22)
    ax.set_xlim(0, 400)
    ax.grid(alpha=0.30, which="major")
    ax.set_axisbelow(True)
    ax.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.92,
              edgecolor="gray")

    # Pull the "Collapse" annotation away from the curve so the arrow tip
    # doesn't visually touch the steep red drop.
    ax.annotate("Collapse\n(step ~40)",
                xy=(45, 0.05), xytext=(135, 0.55),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728",
                                lw=1.2, shrinkA=2, shrinkB=4))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("Saved: %s" % args.out)


if __name__ == "__main__":
    main()
