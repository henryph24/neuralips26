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
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# Colorblind-friendly palette (tab10 first 5)
cmap = plt.cm.tab10
colors = [cmap(i) for i in range(n_experts)]

# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6, 2.7))

panels = [
    (amplitude, volatility, "Amplitude", "Volatility", axes[0]),
    (mean_level, amplitude, "Mean level", "Amplitude", axes[1]),
]

for x_data, y_data, xlabel, ylabel, ax in panels:
    for eid in range(n_experts):
        mask = top1 == eid
        ax.scatter(
            x_data[mask], y_data[mask],
            c=[colors[eid]], s=3.5, alpha=0.25,
            label=expert_names[eid], rasterized=True,
            edgecolors="none",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in")

# Single shared legend, BELOW the panels (avoids float-detachment in LaTeX)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", ncol=n_experts,
    fontsize=7, frameon=False,
    bbox_to_anchor=(0.5, -0.02),
    markerscale=2.5,
    handletextpad=0.4, columnspacing=1.2,
)

# ── Cluster annotation (if one expert dominates a region) ──────────────
# Check left panel: attention expert tends to cluster at high amplitude
counts = Counter(top1)
dominant_id = counts.most_common(1)[0][0]
dom_mask = top1 == dominant_id
dom_amp = amplitude[dom_mask]
dom_vol = volatility[dom_mask]
# Annotate centroid of the dominant cluster in left panel
cx, cy = np.median(dom_amp), np.median(dom_vol)
axes[0].annotate(
    f"{expert_names[dominant_id]} cluster",
    xy=(cx, cy), fontsize=6, fontstyle="italic",
    color=colors[dominant_id],
    ha="center", va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
)

plt.tight_layout(rect=[0, 0.08, 1, 1.0])
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
