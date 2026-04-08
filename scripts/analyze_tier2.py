"""Analyze all Tier 2 experiment results for paper update."""
import json
import glob
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

def load_json(path):
    with open(path) as f:
        return json.load(f)


# =====================================================================
# 1. FULL FINE-TUNING vs FROZEN RR-MoA
# =====================================================================
print("=" * 70)
print("EXP 1: FULL FINE-TUNING (all layers unfrozen) vs FROZEN RR-MoA")
print("=" * 70)

datasets = ["ETTh1", "ETTm1", "Weather", "ETTh2", "ETTm2", "Electricity"]
seeds = [42, 43, 44]

for ds in datasets:
    ft_mses = []
    rr_mses = []
    for seed in seeds:
        # Full-FT
        ft_path = "results/full_finetune/%s_H96_%d.json" % (ds, seed)
        if os.path.exists(ft_path):
            ft = load_json(ft_path)
            ft_mses.append(ft["best"]["mse"])

        # RR-MoA frozen
        rr_path = "results/rr_moa/%s_H96_K5_top2_frozen_%d.json" % (ds, seed)
        if os.path.exists(rr_path):
            rr = load_json(rr_path)
            rr_mses.append(rr["rr_moa"]["mse"])

    if ft_mses and rr_mses:
        ft_mean, ft_std = np.mean(ft_mses), np.std(ft_mses)
        rr_mean, rr_std = np.mean(rr_mses), np.std(rr_mses)
        delta = (rr_mean - ft_mean) / ft_mean * 100
        print("  %-12s  Full-FT: %.3f±%.3f  RR-MoA: %.3f±%.3f  Δ=%.1f%%" % (
            ds, ft_mean, ft_std, rr_mean, rr_std, delta))


# =====================================================================
# 2. FIVE-SEED CORE TABLE
# =====================================================================
print("\n" + "=" * 70)
print("EXP 2: 5-SEED CORE TABLE (frozen RR-MoA Top-2)")
print("=" * 70)

all_seeds = [42, 43, 44, 45, 46]
for ds in datasets:
    mses = []
    bl_mses = []
    for seed in all_seeds:
        rr_path = "results/rr_moa/%s_H96_K5_top2_frozen_%d.json" % (ds, seed)
        if os.path.exists(rr_path):
            rr = load_json(rr_path)
            mses.append(rr["rr_moa"]["mse"])
            bl_vals = [v["mse"] for v in rr["baselines"].values() if "mse" in v]
            if bl_vals:
                bl_mses.append(min(bl_vals))

    if mses:
        mean, std = np.mean(mses), np.std(mses)
        if bl_mses:
            bl_mean, bl_std = np.mean(bl_mses), np.std(bl_mses)
            delta = (mean - bl_mean) / bl_mean * 100
            print("  %-12s  n=%d  RR-MoA: %.3f±%.3f  Best-fixed: %.3f±%.3f  Δ=%.1f%%" % (
                ds, len(mses), mean, std, bl_mean, bl_std, delta))
        else:
            print("  %-12s  n=%d  RR-MoA: %.3f±%.3f  (no baselines)" % (
                ds, len(mses), mean, std))


# =====================================================================
# 3. CHRONOS CROSS-BACKBONE
# =====================================================================
print("\n" + "=" * 70)
print("EXP 3: CHRONOS (decoder-only, no RevIN)")
print("=" * 70)

chronos_datasets = ["ETTh1", "ETTm1", "Weather"]
wins = 0
total = 0
for ds in chronos_datasets:
    mses = []
    bl_mses = []
    for seed in seeds:
        path = "results/rr_moa/%s_H96_K5_top2_frozen_%d_bb-chronos.json" % (ds, seed)
        if os.path.exists(path):
            d = load_json(path)
            rm = d["rr_moa"]["mse"]
            bl_vals = [v["mse"] for v in d["baselines"].values() if "mse" in v]
            if not bl_vals:
                continue
            bl = min(bl_vals)
            mses.append(rm)
            bl_mses.append(bl)
            total += 1
            if rm < bl:
                wins += 1

    if mses:
        mean, std = np.mean(mses), np.std(mses)
        bl_mean = np.mean(bl_mses)
        delta = (mean - bl_mean) / bl_mean * 100
        print("  %-12s  RR-MoA: %.3f±%.3f  Best-fixed: %.3f  Δ=%.1f%%" % (
            ds, mean, std, bl_mean, delta))

print("  Wins: %d/%d" % (wins, total))
print("  NOTE: Chronos has no RevIN -> theory predicts less/no benefit from RR-MoA")


# =====================================================================
# 4. MOIRAI EXTENDED (all 6 datasets)
# =====================================================================
print("\n" + "=" * 70)
print("EXP 4: MOIRAI CROSS-BACKBONE (all 6 datasets)")
print("=" * 70)

moirai_wins = 0
moirai_total = 0
for ds in datasets:
    mses = []
    bl_mses = []
    for seed in seeds:
        path = "results/rr_moa/%s_H96_K5_top2_frozen_%d_bb-moirai.json" % (ds, seed)
        if os.path.exists(path):
            d = load_json(path)
            rm = d["rr_moa"]["mse"]
            bl_vals = [v["mse"] for v in d["baselines"].values() if "mse" in v]
            if not bl_vals:
                continue
            bl = min(bl_vals)
            mses.append(rm)
            bl_mses.append(bl)
            moirai_total += 1
            if rm < bl:
                moirai_wins += 1

    if mses:
        mean, std = np.mean(mses), np.std(mses)
        bl_mean = np.mean(bl_mses)
        delta = (mean - bl_mean) / bl_mean * 100
        print("  %-12s  RR-MoA: %.3f±%.3f  Best-fixed: %.3f  Δ=%.1f%%" % (
            ds, mean, std, bl_mean, delta))

print("  Wins: %d/%d" % (moirai_wins, moirai_total))


# =====================================================================
# 5. LORA UNFROZEN
# =====================================================================
print("\n" + "=" * 70)
print("EXP 5: LORA WITH UNFREEZING (last4) vs FROZEN LORA vs FROZEN RR-MoA")
print("=" * 70)

for ds in ["ETTh1", "ETTm1", "Weather"]:
    frozen_mses = []
    unfrozen_mses = []
    rr_mses = []
    for seed in seeds:
        # LoRA frozen (best from sweep)
        frozen_path = "results/lora_baseline/%s_H96_r32_qkvo_mlp2_%d.json" % (ds, seed)
        if os.path.exists(frozen_path):
            d = load_json(frozen_path)
            frozen_mses.append(d.get("mse", d.get("test_mse", 0)))

        # LoRA last4
        unfr_path = "results/lora_baseline/%s_H96_r16_qkvo_mlp2_last4_%d.json" % (ds, seed)
        if os.path.exists(unfr_path):
            d = load_json(unfr_path)
            # Find MSE key
            mse = d.get("mse") or d.get("test_mse") or d.get("results", {}).get("mse")
            if mse:
                unfrozen_mses.append(mse)

        # RR-MoA frozen
        rr_path = "results/rr_moa/%s_H96_K5_top2_frozen_%d.json" % (ds, seed)
        if os.path.exists(rr_path):
            rr = load_json(rr_path)
            rr_mses.append(rr["rr_moa"]["mse"])

    parts = ["  %-12s" % ds]
    if unfrozen_mses:
        parts.append("LoRA-last4: %.3f±%.3f" % (np.mean(unfrozen_mses), np.std(unfrozen_mses)))
    if rr_mses:
        parts.append("RR-MoA: %.3f±%.3f" % (np.mean(rr_mses), np.std(rr_mses)))
    print("  ".join(parts))


# =====================================================================
# 6. BENCHMARK
# =====================================================================
print("\n" + "=" * 70)
print("EXP 6: INFERENCE BENCHMARK")
print("=" * 70)

bench_files = glob.glob("results/benchmark/*.json")
for f in bench_files:
    d = load_json(f)
    print("  %-20s  %10s  %10s  %12s" % ("Method", "Latency", "Peak GPU", "Params"))
    print("  " + "-" * 56)
    for name, v in d.items():
        params = v.get("adapter_params", v.get("params", 0))
        overhead = ""
        if name == "rrmoa_top2" and "single_adapter" in d:
            oh = v["mean_ms"] - d["single_adapter"]["mean_ms"]
            overhead = " (+%.1fms)" % oh
        print("  %-20s  %8.1f ms  %7.0f MB  %10s%s" % (
            name, v["mean_ms"], v.get("peak_mb", 0), "{:,}".format(params), overhead))


# =====================================================================
# SUMMARY TABLE FOR PAPER
# =====================================================================
print("\n" + "=" * 70)
print("PAPER TABLE: Full baseline comparison (MOMENT-small, frozen, 5 seeds)")
print("=" * 70)
print("%-15s  %-20s  %-20s  %-20s" % ("Dataset", "Full-FT (all unfr.)", "Best LoRA (108-run)", "RR-MoA (Top-2)"))
print("-" * 75)

for ds in datasets:
    # Full-FT (3 seeds)
    ft_vals = []
    for s in [42, 43, 44]:
        p = "results/full_finetune/%s_H96_%d.json" % (ds, s)
        if os.path.exists(p):
            ft_vals.append(load_json(p)["best"]["mse"])

    # RR-MoA (5 seeds)
    rr_vals = []
    for s in [42, 43, 44, 45, 46]:
        p = "results/rr_moa/%s_H96_K5_top2_frozen_%d.json" % (ds, s)
        if os.path.exists(p):
            rr_vals.append(load_json(p)["rr_moa"]["mse"])

    ft_str = "%.3f±%.3f" % (np.mean(ft_vals), np.std(ft_vals)) if ft_vals else "---"
    rr_str = "%.3f±%.3f" % (np.mean(rr_vals), np.std(rr_vals)) if rr_vals else "---"
    print("%-15s  %-20s  %-20s  %-20s" % (ds, ft_str, "see Table 5", rr_str))
