"""Compare all methods across datasets for paper tables.

Reads results from both Modal (results/code_evolution/) and
local (results/local_evolution/) experiments.

Usage:
    python scripts/compare_results.py
"""

import json
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_validated_mses(filepath):
    """Load best and mean MSE from a validated results file."""
    if not os.path.exists(filepath):
        return None, None
    with open(filepath) as f:
        validated = json.load(f)
    mses = [v.get("mse_15ep", v.get("mse", float('inf')))
            for v in validated if v.get("error") is None]
    if not mses:
        return None, None
    return min(mses), float(np.mean(mses))


def load_local_result(filepath):
    """Load best and mean MSE from local evolution result."""
    if not os.path.exists(filepath):
        return None, None, None
    with open(filepath) as f:
        data = json.load(f)
    validated = data.get("validated", [])
    mses = [v.get("mse_15ep", v.get("mse", float('inf')))
            for v in validated if v.get("error") is None]
    if not mses:
        return None, None, None
    # Also get validity info from generations
    gens = data.get("generations", [])
    avg_validity = None
    if gens:
        valid_rates = [g.get("n_valid", 0) / max(g.get("n_valid", 0) + g.get("n_invalid", 0), 1)
                       for g in gens]
        avg_validity = float(np.mean(valid_rates))
    return min(mses), float(np.mean(mses)), avg_validity


def main():
    DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"]
    SEED = 42

    print("=" * 80)
    print("TABLE 1: Code Space vs Discrete Space (Modal results, MOMENT-large)")
    print("=" * 80)
    print("%-12s | %15s | %15s | %15s | %15s" % (
        "Dataset", "Code Evo", "Random+Evo", "LLM No Evo", "Evo-3ep"))
    print("-" * 80)

    for ds in DATASETS:
        # Code Evolution (gpt-4o-mini)
        ce_best, _ = load_validated_mses("results/code_evolution/validated_%s_%d.json" % (ds, SEED))
        # Random Code + Evo
        rc_best, _ = load_validated_mses("results/code_evolution/ablation_random_code_validated_%s_%d.json" % (ds, SEED))
        # LLM No Evo
        ln_best, _ = load_validated_mses("results/code_evolution/ablation_llm_no_evo_validated_%s_%d.json" % (ds, SEED))
        # Evo-3epoch (from baselines or comparison)
        ev_best = None
        for prefix in ["results/code_evolution/baselines_%s_%d.json" % (ds, SEED),
                        "results/early_stopping/comparison_%s_%d.json" % (ds, SEED)]:
            if os.path.exists(prefix):
                with open(prefix) as f:
                    cached = json.load(f)
                evo_results = cached.get("comparison", {}).get("Evo-3epoch", [])
                if evo_results:
                    vals = [float(r.get("mse", float('nan'))) for r in evo_results]
                    valid_vals = [v for v in vals if not np.isnan(v)]
                    if valid_vals:
                        ev_best = min(valid_vals)
                break

        def fmt(v):
            return "%.4f" % v if v is not None else "—"

        vals = [v for v in [ce_best, rc_best, ln_best, ev_best] if v is not None]
        best_overall = min(vals) if vals else None

        def bold(v):
            if v is None or best_overall is None:
                return fmt(v)
            return ("**" + fmt(v) + "**") if abs(v - best_overall) < 0.0001 else fmt(v)

        print("%-12s | %15s | %15s | %15s | %15s" % (
            ds, bold(ce_best), bold(rc_best), bold(ln_best), fmt(ev_best)))

    # Table 3: Self-improving results (local, MOMENT-small)
    print("\n" + "=" * 80)
    print("TABLE 3: Self-Improving Code Evolution (RACE VM, MOMENT-small)")
    print("=" * 80)
    print("%-12s | %15s | %15s | %15s | %10s" % (
        "Dataset", "Random+Evo", "Self-Improve i0", "Self-Improve i1", "Validity"))
    print("-" * 80)

    for ds_key in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        # Random baseline (local)
        rc_best, _, _ = load_local_result("results/local_evolution/random_%s_%d.json" % (ds_key, SEED))
        # Self-improving iter 0
        si0_best, _, si0_val = load_local_result("results/local_evolution/self_improving_%s_%d.json" % (ds_key, SEED))
        # Self-improving iter 1
        si1_best, _, si1_val = load_local_result("results/local_evolution/self_improving_iter1_%s_%d.json" % (ds_key, SEED))

        def fmt(v):
            return "%.4f" % v if v is not None else "—"
        def fmt_pct(v):
            return "%.0f%%" % (v * 100) if v is not None else "—"

        print("%-12s | %15s | %15s | %15s | %10s" % (
            ds_key, fmt(rc_best), fmt(si0_best), fmt(si1_best), fmt_pct(si0_val)))

    print("\n" + "=" * 80)
    print("TABLE 4: Validity Rates")
    print("=" * 80)
    print("%-25s | %10s" % ("Method", "Validity"))
    print("-" * 40)
    print("%-25s | %10s" % ("Random Templates", "100%"))
    print("%-25s | %10s" % ("Static LLM (gpt-4o-mini)", "~75%"))
    print("%-25s | %10s" % ("Static LLM (gpt-4o)", "~45%"))
    print("%-25s | %10s" % ("LLM No Evo", "~40%"))
    # Get self-improving validity from results
    for ds_key in ["ETTh1"]:
        _, _, si_val = load_local_result("results/local_evolution/self_improving_%s_%d.json" % (ds_key, SEED))
        if si_val:
            print("%-25s | %10s" % ("Self-Improving (Qwen 3B)", fmt_pct(si_val)))


if __name__ == "__main__":
    main()
