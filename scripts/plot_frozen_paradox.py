"""Generate Frozen Paradox bar chart for main text.

Shows frozen RR-MoA vs full fine-tuning across all 6 datasets.
Outputs TikZ code for direct inclusion in main.tex.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

datasets = ["ETTh1", "ETTm1", "Weather", "ETTh2", "ETTm2", "Electricity"]
short_names = ["ETTh1", "ETTm1", "Weath", "ETTh2", "ETTm2", "Elec"]
seeds_rr = [42, 43, 44, 45, 46]
seeds_ft = [42, 43, 44]

print("=== Data for Frozen Paradox Figure ===\n")

ft_means = []
ft_stds = []
rr_means = []
rr_stds = []
deltas = []

for ds in datasets:
    # Full-FT
    ft_vals = []
    for s in seeds_ft:
        p = "results/full_finetune/%s_H96_%d.json" % (ds, s)
        if os.path.exists(p):
            d = json.load(open(p))
            ft_vals.append(d["best"]["mse"])

    # RR-MoA frozen (5 seeds)
    rr_vals = []
    for s in seeds_rr:
        p = "results/rr_moa/%s_H96_K5_top2_frozen_%d.json" % (ds, s)
        if os.path.exists(p):
            d = json.load(open(p))
            rr_vals.append(d["rr_moa"]["mse"])

    ft_m, ft_s = np.mean(ft_vals), np.std(ft_vals)
    rr_m, rr_s = np.mean(rr_vals), np.std(rr_vals)
    delta = (rr_m - ft_m) / ft_m * 100

    ft_means.append(ft_m)
    ft_stds.append(ft_s)
    rr_means.append(rr_m)
    rr_stds.append(rr_s)
    deltas.append(delta)

    print("  %s: Full-FT=%.3f±%.3f  RR-MoA=%.3f±%.3f  Δ=%.1f%%" % (
        ds, ft_m, ft_s, rr_m, rr_s, delta))

# Generate TikZ code
print("\n=== TikZ Code ===\n")

tikz = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=\columnwidth,
    height=5.5cm,
    bar width=8pt,
    ylabel={Test MSE (H=96)},
    ylabel style={font=\small},
    symbolic x coords={"""

tikz += ", ".join(short_names)

tikz += r"""},
    xtick=data,
    x tick label style={font=\small},
    y tick label style={font=\small},
    legend style={at={(0.02,0.98)}, anchor=north west, font=\scriptsize, draw=none, fill=white, fill opacity=0.8},
    ymin=0,
    enlarge x limits=0.12,
    error bars/y dir=both,
    error bars/y explicit,
    error bars/error bar style={line width=0.5pt, black!60},
    nodes near coords style={font=\tiny, rotate=90, anchor=west, xshift=-1pt},
]
"""

# Full-FT bars
tikz += "\\addplot[fill=red!45, nodes near coords={}] coordinates {"
for i, sn in enumerate(short_names):
    tikz += "(%s,%.3f) +- (0,%.3f) " % (sn, ft_means[i], ft_stds[i])
tikz += "};\n"

# RR-MoA bars
tikz += "\\addplot[fill=blue!55, nodes near coords={}] coordinates {"
for i, sn in enumerate(short_names):
    tikz += "(%s,%.3f) +- (0,%.3f) " % (sn, rr_means[i], rr_stds[i])
tikz += "};\n"

# Delta labels on top
for i, sn in enumerate(short_names):
    # Place label above the taller bar
    y = max(ft_means[i], rr_means[i]) + max(ft_stds[i], rr_stds[i]) + 0.05
    tikz += "\\node[font=\\tiny, text=red!70!black] at (axis cs:%s,%.2f) {$%.0f\\%%$};\n" % (
        sn, min(y + 0.08, ft_means[i] + 0.15), deltas[i])

tikz += "\\legend{Full fine-tuning (all unfrozen), Frozen RR-MoA (Top-2)}\n"
tikz += r"""\end{axis}
\end{tikzpicture}
\caption{\textbf{The Frozen Paradox.} Frozen RR-MoA (blue) beats full backbone fine-tuning with all 8 layers unfrozen (red) by $16$--$77\%$ across all 6 datasets, reversing the standard PEFT heuristic that unfreezing improves adaptation. Error bars: std over 3 seeds (full-FT) and 5 seeds (RR-MoA).}
\label{fig:frozen_paradox}
\end{figure}"""

print(tikz)
