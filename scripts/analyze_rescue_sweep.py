"""Aggregate results/adamix_rescue/*.json into a paper-ready rescue sub-table.

Usage:
    python3 scripts/analyze_rescue_sweep.py              # print markdown summary
    python3 scripts/analyze_rescue_sweep.py --latex      # emit LaTeX table body

Groups by (router_type, load_balance_coef, load_balance_variant, entropy_reg_coef,
z_loss_coef, relu_l1_coef, capacity_factor).  For each config, averages MSE and
routing entropy across 6 datasets x 2 freeze levels x up to 5 seeds, reporting
mean, std, number of cells, and a "collapsed?" flag (entropy < 0.3 in >=80% of
cells).  Designed to be safe to run mid-sweep — partial configs are flagged.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

CONFIG_LABELS = {
    ("softmax", 0.01, "mean-prob", 0.0, 0.0, 0.0, 2.0): "Baseline legacy ($f_i = P_i$)",
    ("softmax", 0.01, "argmax", 0.0, 0.0, 0.0, 2.0):    "Baseline Switch (correct $f_i$)",
    ("softmax", 0.01, "argmax", 0.01, 0.0, 0.0, 2.0):    "+ Entropy reg $\\lambda{=}0.01$",
    ("softmax", 0.01, "argmax", 0.1, 0.0, 0.0, 2.0):     "+ Entropy reg $\\lambda{=}0.1$",
    ("softmax", 0.01, "argmax", 1.0, 0.0, 0.0, 2.0):     "+ Entropy reg $\\lambda{=}1.0$",
    ("softmax", 0.01, "argmax", 0.0, 0.001, 0.0, 2.0):   "+ Z-loss $c_z{=}0.001$ (ST-MoE)",
    ("softmax", 0.01, "argmax", 0.0, 0.01, 0.0, 2.0):    "+ Z-loss $c_z{=}0.01$",
    ("softmax", 0.01, "argmax", 0.0, 0.1, 0.0, 2.0):     "+ Z-loss $c_z{=}0.1$",
    ("softmax", 0.1, "argmax", 0.0, 0.0, 0.0, 2.0):      "Load-balance $\\alpha{=}0.1$",
    ("softmax", 1.0, "argmax", 0.0, 0.0, 0.0, 2.0):      "Load-balance $\\alpha{=}1.0$",
    ("softmax", 10.0, "argmax", 0.0, 0.0, 0.0, 2.0):     "Load-balance $\\alpha{=}10.0$",
    ("relu", 0.0, "argmax", 0.0, 0.0, 0.01, 2.0):        "ReMoE ReLU $+$ $L_1$",
    ("expert-choice", 0.0, "argmax", 0.0, 0.0, 0.0, 2.0): "Expert-choice ($c{=}2$)",
}


def _key(d):
    return (
        d.get("router_type", "softmax"),
        round(float(d.get("load_balance_coef", 0.01)), 6),
        d.get("load_balance_variant", "mean-prob"),
        round(float(d.get("entropy_reg_coef", 0.0)), 6),
        round(float(d.get("z_loss_coef", 0.0)), 6),
        round(float(d.get("relu_l1_coef", 0.0)), 6),
        round(float(d.get("capacity_factor", 2.0)), 6),
    )


def load(rescue_dir="results/adamix_rescue"):
    by_cfg = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(rescue_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        k = _key(d)
        mse = float(d.get("adamix", {}).get("mse", float("nan")))
        ent = float(d.get("adamix", {}).get("routing_entropy", float("nan")))
        ds = d.get("dataset", "?")
        frz = d.get("unfreeze", "?")
        seed = d.get("seed", -1)
        by_cfg[k].append({"mse": mse, "ent": ent, "ds": ds, "frz": frz, "seed": seed})
    return by_cfg


def summarize(by_cfg, expected_cells=60):
    """One row per config with mean_mse, std_mse, mean_ent, std_ent, n_cells, collapsed_frac."""
    rows = []
    for k, vals in sorted(by_cfg.items(), key=lambda kv: list(CONFIG_LABELS.keys()).index(kv[0]) if kv[0] in CONFIG_LABELS else 999):
        mses = [v["mse"] for v in vals if v["mse"] == v["mse"]]  # drop NaN
        ents = [v["ent"] for v in vals if v["ent"] == v["ent"]]
        n = len(vals)
        collapsed = sum(1 for v in vals if v["ent"] < 0.3)
        collapsed_frac = collapsed / n if n else 0.0
        rows.append({
            "label": CONFIG_LABELS.get(k, "?" + str(k)),
            "n": n,
            "expected": expected_cells,
            "mean_mse": statistics.mean(mses) if mses else float("nan"),
            "std_mse": statistics.stdev(mses) if len(mses) >= 2 else 0.0,
            "mean_ent": statistics.mean(ents) if ents else float("nan"),
            "std_ent": statistics.stdev(ents) if len(ents) >= 2 else 0.0,
            "collapsed_frac": collapsed_frac,
        })
    return rows


def print_markdown(rows):
    print(f"| {'Config':<34} | {'n/60':<8} | {'mean MSE':<9} | {'mean H':<7} | {'collapsed?':<12} |")
    print(f"|{'-'*36}|{'-'*10}|{'-'*11}|{'-'*9}|{'-'*14}|")
    for r in rows:
        tag = "YES" if r["collapsed_frac"] > 0.8 else ("part" if r["collapsed_frac"] > 0.3 else "no")
        print(f"| {r['label']:<34} | {r['n']:>2}/{r['expected']:<5} | "
              f"{r['mean_mse']:7.4f}   | {r['mean_ent']:5.3f}  | "
              f"{tag} ({r['collapsed_frac']:.0%})     |")


def print_latex(rows):
    """Emit a LaTeX tabular body (no surrounding \\begin/\\end)."""
    for r in rows:
        star = "*" if r["n"] < r["expected"] else ""
        tag = r"\cmark" if r["collapsed_frac"] > 0.8 else (r"(part)" if r["collapsed_frac"] > 0.3 else r"\xmark")
        print(f"{r['label']} & {r['mean_mse']:.4f}$\\pm${r['std_mse']:.4f} & "
              f"{r['mean_ent']:.3f}$\\pm${r['std_ent']:.3f} & {tag} & {r['n']}/{r['expected']}{star} \\\\")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latex", action="store_true")
    p.add_argument("--dir", default="results/adamix_rescue")
    args = p.parse_args()
    by_cfg = load(args.dir)
    rows = summarize(by_cfg)
    if args.latex:
        print_latex(rows)
    else:
        print_markdown(rows)
    total = sum(r["n"] for r in rows)
    print(f"\nTotal runs: {total}/720")


if __name__ == "__main__":
    main()
