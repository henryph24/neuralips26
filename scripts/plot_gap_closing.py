#!/usr/bin/env python3
"""Generate the dual-stream gap-closing bar chart (R2 figure).

Produces figures/gap_closing_bars.pdf — a grouped bar chart showing
DLinear, RR-MoA, and Dual-stream MSE for 3 datasets, with the Weather
bar highlighted where Dual-stream beats DLinear.
"""
import json, glob, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

# --- Load data ---
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

gc = defaultdict(lambda: defaultdict(list))
for f in glob.glob("results/gap_closing/*.json"):
    with open(f) as fh:
        d = json.load(fh)
    gc[d['dataset']][d['variant']].append(d['mse'])

rrmoa = defaultdict(list)
for f in glob.glob("results/rr_moa/*_H96_*_frozen_*.json"):
    if any(s in f for s in ['bb-', 'pool-', 'no_revin', 'batchnorm', 'groupnorm', 'router-', 'ep50']):
        continue
    with open(f) as fh:
        d = json.load(fh)
    if d['seed'] in [42, 43, 44]:
        rrmoa[d['dataset']].append(d['rr_moa']['mse'])

dlinear = defaultdict(list)
for f in glob.glob("results/dlinear/*.json"):
    with open(f) as fh:
        d = json.load(fh)
    if d.get('horizon', 96) == 96 and d.get('seed', 42) in [42, 43, 44]:
        dlinear[d['dataset']].append(d['dlinear_mse'])

datasets = ['ETTh1', 'ETTm1', 'Weather']
methods = ['DLinear', 'RR-MoA (frozen)', 'Dual-stream']

means = []
stds = []
for ds in datasets:
    dl_m, dl_s = np.mean(dlinear[ds]), np.std(dlinear[ds])
    rr_m, rr_s = np.mean(rrmoa[ds]), np.std(rrmoa[ds])
    ds_m, ds_s = np.mean(gc[ds]['dual-stream']), np.std(gc[ds]['dual-stream'])
    means.append([dl_m, rr_m, ds_m])
    stds.append([dl_s, rr_s, ds_s])

means = np.array(means)
stds = np.array(stds)

# --- Plot ---
fig, ax = plt.subplots(figsize=(5.5, 3.2))

x = np.arange(len(datasets))
width = 0.22
colors = ['#4A90D9', '#E8634A', '#2EAD6D']
edge_colors = ['#2C5F9E', '#B84232', '#1E7A4C']
hatches = ['', '///', '']

for i, method in enumerate(methods):
    bars = ax.bar(x + (i - 1) * width, means[:, i], width,
                  yerr=stds[:, i], capsize=3,
                  color=colors[i], edgecolor=edge_colors[i],
                  linewidth=0.8, label=method, zorder=3,
                  error_kw={'linewidth': 0.8, 'zorder': 4})

    # Highlight Weather dual-stream bar (beats DLinear)
    if method == 'Dual-stream':
        # Add star on Weather bar
        weather_idx = 2
        ax.annotate('*',
                    xy=(x[weather_idx] + (i - 1) * width, means[weather_idx, i] + stds[weather_idx, i] + 0.008),
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color='#1E7A4C')

# Add gap annotations above each dataset group
for j, ds in enumerate(datasets):
    dl = means[j, 0]
    rr = means[j, 1]
    ds_val = means[j, 2]
    rr_gap = (rr - dl) / dl * 100
    ds_gap = (ds_val - dl) / dl * 100

    # Show gap reduction
    max_bar = max(means[j, :]) + max(stds[j, :])
    gap_text = f'+{rr_gap:.0f}% $\\to$ {"+" if ds_gap >= 0 else ""}{ds_gap:.1f}%'
    ax.text(x[j], max_bar + 0.025, gap_text, ha='center', va='bottom',
            fontsize=7.5, color='#333333', style='italic')

ax.set_ylabel('Test MSE')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, max(means.max(axis=0)) + 0.15)
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')
ax.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/gap_closing_bars.pdf', bbox_inches='tight')
plt.savefig('figures/gap_closing_bars.png', bbox_inches='tight', dpi=200)
print("Saved figures/gap_closing_bars.pdf and .png")
