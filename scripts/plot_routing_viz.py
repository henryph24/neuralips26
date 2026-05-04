"""
Generate a two-panel scatter plot of routing decisions colored by top-1 expert.
Left: amplitude vs volatility. Right: mean_level vs amplitude.
Output: figures/routing_viz.pdf
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "results", "routing_analysis", "ETTh1_42.json")
OUT_DIR = os.path.join(ROOT, "figures")
OUT_PATH = os.path.join(OUT_DIR, "routing_viz.pdf")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

amplitude = np.array(data["amplitude"])
volatility = np.array(data["volatility"])
mean_level = np.array(data["mean_level"])
top1 = np.array(data["top1_expert"])
expert_names = data["expert_names"]
n_experts = len(expert_names)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Distinct colorblind-safe palette (Wong palette: orange/sky-blue/bluish-green/vermillion/yellow).
# Order matches expert index; only experts present in data are drawn.
WONG = {
    0: "#E69F00",   # mean: orange
    1: "#56B4E9",   # last: sky blue
    2: "#009E73",   # max: bluish green
    3: "#0072B2",   # attention: dark blue (distinct from mean's orange)
    4: "#CC79A7",   # conv1d: reddish purple
}

# Honest legend: count usage per expert and only display experts with >=1%.
counts = Counter(top1)
N = len(top1)
PRESENT_THRESH = 0.01  # 1% of routing decisions
present_ids = [eid for eid in range(n_experts) if counts[eid] / N >= PRESENT_THRESH]
absent_ids = [eid for eid in range(n_experts) if counts[eid] / N < PRESENT_THRESH]

# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

panels = [
    (amplitude, volatility, "Amplitude", "Volatility", axes[0]),
    (mean_level, amplitude, "Mean level", "Amplitude", axes[1]),
]

# Plot in count order (largest cluster first, drawn underneath; smaller on top so they remain visible).
plot_order = sorted(present_ids, key=lambda eid: -counts[eid])

for x_data, y_data, xlabel, ylabel, ax in panels:
    for eid in plot_order:
        mask = top1 == eid
        ax.scatter(
            x_data[mask], y_data[mask],
            c=WONG[eid], s=8, alpha=0.45,
            label=f"{expert_names[eid]} ({100*counts[eid]/N:.0f}%)",
            rasterized=True, edgecolors="none",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)

# Build a single shared legend below the panels (only experts present in data).
handles, labels = axes[0].get_legend_handles_labels()
absent_note = ""
if absent_ids:
    absent_note = "  (" + ", ".join(expert_names[i] for i in absent_ids) + r"$\,{<}1\%$)"
fig.legend(
    handles, labels,
    loc="lower center", ncol=len(present_ids),
    fontsize=8, frameon=True, framealpha=0.92, edgecolor="gray",
    bbox_to_anchor=(0.5, -0.04),
    markerscale=2.0,
    handletextpad=0.5, columnspacing=1.4,
    title=f"Top-1 expert (N={N:,} test windows){absent_note}",
    title_fontsize=8,
)

# Annotate the semantic story: attention takes the high-amp/high-vol windows.
attn_id = expert_names.index("attention") if "attention" in expert_names else None
mean_id = expert_names.index("mean") if "mean" in expert_names else None
max_id = expert_names.index("max") if "max" in expert_names else None

if attn_id is not None and counts.get(attn_id, 0) > 100:
    attn_mask = top1 == attn_id
    cx, cy = np.percentile(amplitude[attn_mask], 80), np.percentile(volatility[attn_mask], 80)
    axes[0].annotate(
        "attention $\\to$\nhigh amp+vol",
        xy=(cx, cy), xytext=(cx - 1.8, cy + 0.08),
        fontsize=7.5, color=WONG[attn_id],
        arrowprops=dict(arrowstyle="->", color=WONG[attn_id], lw=0.7, alpha=0.7),
        ha="center", va="bottom",
    )

if max_id is not None and counts.get(max_id, 0) > 100:
    max_mask = top1 == max_id
    cx, cy = np.median(amplitude[max_mask]), np.median(volatility[max_mask])
    axes[0].annotate(
        "max $\\to$ low",
        xy=(cx, cy), xytext=(cx + 1.4, cy - 0.04),
        fontsize=7.5, color=WONG[max_id],
        arrowprops=dict(arrowstyle="->", color=WONG[max_id], lw=0.7, alpha=0.7),
        ha="center", va="top",
    )

plt.tight_layout(rect=[0, 0.10, 1, 1.0])
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
print(f"Present experts (>=1%): {[expert_names[i] for i in present_ids]}")
print(f"Absent experts (<1%): {[expert_names[i] for i in absent_ids]}")
