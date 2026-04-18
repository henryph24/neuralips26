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

    trajectories = {
        "MOMENT frozen": (args.moment_frozen, "tab:blue", "--", 1.5),
        "MOMENT last-4 (RevIN)": (args.moment_last4, "tab:red", "-", 2.5),
        "Timer-XL frozen": (args.timer_frozen, "tab:cyan", "--", 1.5),
        "Timer-XL last-4 (no RevIN)": (args.timer_last4, "tab:orange", "-", 2.5),
    }

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5), dpi=150)

    for label, (path, color, ls, lw) in trajectories.items():
        data = load_jsonl(path)
        steps = [r["step"] for r in data]
        ent = [r["routing_entropy"] for r in data]
        ax.plot(steps, ent, color=color, linestyle=ls, linewidth=lw, label=label)

    ax.axhline(log_max, color="gray", linestyle=":", linewidth=0.8,
               label=r"$\log K$ (max entropy)")
    ax.set_xlabel("Optimizer step", fontsize=11)
    ax.set_ylabel("Router entropy (nats)", fontsize=11)
    ax.set_ylim(-0.05, log_max + 0.15)
    ax.set_xlim(0, 400)
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.9)

    # Add annotation for the collapse point
    ax.annotate("Collapse\n(step ~40)",
                xy=(40, 0.0), xytext=(120, 0.35),
                fontsize=8, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("Saved: %s" % args.out)


if __name__ == "__main__":
    main()
