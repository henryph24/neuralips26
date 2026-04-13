#!/usr/bin/env python3
"""Generate the cross-backbone R4 figure.

Produces figures/cross_backbone_bars.pdf — a bar chart showing per-backbone
RR-MoA improvement (delta% over best-fixed baseline), with Chronos as the
negative control.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data from paper-verified tables + computed Chronos values
backbones = ['MOMENT\nsmall', 'MOMENT\nlarge', 'Moirai', 'Moirai\nMoE', 'Chronos']
deltas = [46.3, 42.0, 19.0, 60.6, -14.3]
normalization = ['RevIN', 'RevIN', 'LayerNorm', 'LayerNorm', 'None']

# Colors by normalization type
color_map = {
    'RevIN': '#3A7ABF',      # blue — aggressive instance norm
    'LayerNorm': '#5BAE6D',  # green — mild normalization
    'None': '#D9534F',       # red — no instance norm (negative control)
}
edge_map = {
    'RevIN': '#2B5D8F',
    'LayerNorm': '#3D7A4A',
    'None': '#A63D3A',
}

colors = [color_map[n] for n in normalization]
edges = [edge_map[n] for n in normalization]

fig, ax = plt.subplots(figsize=(5.5, 3.2))

x = np.arange(len(backbones))
bars = ax.bar(x, deltas, width=0.55, color=colors, edgecolor=edges, linewidth=0.8, zorder=3)

# Zero line
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', zorder=2)

# Annotations on each bar
for i, (delta, bb) in enumerate(zip(deltas, backbones)):
    va = 'bottom' if delta >= 0 else 'top'
    offset = 1.5 if delta >= 0 else -1.5
    label = f'{delta:+.1f}%'
    if 'MoE' in bb:
        label += '$^*$'
    ax.text(x[i], delta + offset, label, ha='center', va=va,
            fontsize=8.5, fontweight='bold', color=edges[i])

# Negative control annotation
ax.annotate('negative\ncontrol', xy=(4, -14.3), xytext=(3.3, -35),
            fontsize=7.5, ha='center', color='#A63D3A', style='italic',
            arrowprops=dict(arrowstyle='->', color='#A63D3A', lw=0.8))

# Legend for normalization types (manual patches)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_map['RevIN'], edgecolor=edge_map['RevIN'], label='RevIN (instance norm)'),
    Patch(facecolor=color_map['LayerNorm'], edgecolor=edge_map['LayerNorm'], label='LayerNorm only'),
    Patch(facecolor=color_map['None'], edgecolor=edge_map['None'], label='No instance norm'),
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')

ax.set_ylabel('RR-MoA improvement over best fixed (%)')
ax.set_xticks(x)
ax.set_xticklabels(backbones)
ax.set_ylim(-42, 75)
ax.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Footnote for Moirai-MoE
ax.text(0.02, -0.12, '$^*$Moirai-MoE: 2 of 6 datasets complete; remaining running.',
        transform=ax.transAxes, fontsize=7, color='#666666', style='italic')

plt.tight_layout()
plt.savefig('figures/cross_backbone_bars.pdf', bbox_inches='tight')
plt.savefig('figures/cross_backbone_bars.png', bbox_inches='tight', dpi=200)
print("Saved figures/cross_backbone_bars.pdf and .png")
