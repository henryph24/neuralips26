"""W7 re-analysis (rebuttal 8b2Z W7): significance under uncorrected / Bonferroni / Holm / BH.

8b2Z worries an over-conservative correction obscures effects. We answer in two directions:
  (1) The claimed effect (RR-MoA vs best fixed adapter) is significant under EVERY correction,
      including the most powerful (uncorrected), so no correction hides it.
  (2) The Traffic boundary NULL is non-significant even UNCORRECTED (the most powerful test),
      so it is a genuine falsification, not a power artifact of a strict correction. (Solar is
      the low-R OUTLIER where RR-MoA improves anyway, so it should be, and is, significant.)
Uses the archived per-run MSEs in results/rr_moa/. Local, scipy only.
"""

import glob
import json
import os
import re

import numpy as np
from scipy.stats import wilcoxon

BASE = "results/rr_moa"


def load_pairs(dataset, unfreeze="frozen"):
    """Return list of (seed, rrmoa_mse, best_fixed_mse) from base (unsuffixed) files."""
    pairs = []
    pat = re.compile(rf"^{re.escape(dataset)}_H96_K5_top2_{unfreeze}_(\d+)\.json$")
    for f in sorted(glob.glob(f"{BASE}/{dataset}_H96_K5_top2_{unfreeze}_*.json")):
        m = pat.match(os.path.basename(f))
        if not m:  # skip suffixed ablation files (bb-*, lr-*, pool-*, ...)
            continue
        d = json.load(open(f))
        rr = d["rr_moa"]["mse"]
        bl = d.get("baselines", {})
        vals = []
        for k in ("linear", "attention", "conv"):
            v = bl.get(k)
            if isinstance(v, dict):
                v = v.get("mse")
            if isinstance(v, (int, float)):
                vals.append(v)
        if not vals:
            continue
        pairs.append((int(m.group(1)), rr, min(vals)))
    return pairs


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * n / (rank + 1))
        adj[i] = min(running, 1.0)
    return adj


MAIN = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]

per_ds, pooled_rr, pooled_bl = {}, [], []
for ds in MAIN:
    pr = load_pairs(ds)
    if not pr:
        print(f"WARN: no data for {ds}")
        continue
    rr = [a for _, a, _ in pr]
    bl = [b for _, _, b in pr]
    _, p = wilcoxon(rr, bl, alternative="less")
    per_ds[ds] = (p, float(np.mean(rr)), float(np.mean(bl)), len(pr))
    pooled_rr += rr
    pooled_bl += bl

names = list(per_ds.keys())
pvals = np.array([per_ds[d][0] for d in names])
bonf = np.minimum(pvals * len(pvals), 1.0)
holm_p, bh_p = holm(pvals), bh(pvals)
_, pooled_p = wilcoxon(pooled_rr, pooled_bl, alternative="less")

print("=== RR-MoA vs best fixed adapter: per-dataset (Wilcoxon, alt=less) ===")
print(f"{'dataset':<12}{'n':>3}{'RRMoA':>8}{'fixed':>8}{'p_unc':>11}{'Bonf':>11}{'Holm':>11}{'BH':>11}")
for i, d in enumerate(names):
    p, rr, bl, n = per_ds[d]
    print(f"{d:<12}{n:>3}{rr:>8.3f}{bl:>8.3f}{p:>11.2e}{bonf[i]:>11.2e}{holm_p[i]:>11.2e}{bh_p[i]:>11.2e}")
print(f"\nsignificant at 0.05:  unc={(pvals<.05).sum()}/{len(names)}  "
      f"Bonf={(bonf<.05).sum()}/{len(names)}  Holm={(holm_p<.05).sum()}/{len(names)}  BH={(bh_p<.05).sum()}/{len(names)}")
print(f"POOLED ({len(pooled_rr)} pairs): p_uncorrected = {pooled_p:.2e}")

print("\n=== Boundary cases: is the NULL genuine even under the MOST POWERFUL (uncorrected) test? ===")
for ds in ["Traffic", "Solar"]:
    pr = load_pairs(ds)
    if not pr:
        print(f"  {ds}: no data")
        continue
    rr = [a for _, a, _ in pr]
    bl = [b for _, _, b in pr]
    delta = 100 * (np.mean(rr) - np.mean(bl)) / np.mean(bl)
    _, p_less = wilcoxon(rr, bl, alternative="less")  # RR-MoA significantly better?
    verdict = "GENUINE NULL (no effect even uncorrected)" if p_less > 0.05 else "significant effect (not a null)"
    print(f"  {ds:<9} n={len(pr)} RRMoA={np.mean(rr):.3f} fixed={np.mean(bl):.3f} "
          f"delta={delta:+.1f}%  p_unc(RR<fixed)={p_less:.3f}  -> {verdict}")
