"""GPU cost analysis for adapter selection methods.

Computes total GPU evaluations, estimated GPU-minutes, and relative costs
for each method in the adapter selection experiment.

All estimates are for a single dataset. Calibration cost is amortized
across datasets but shown separately.
"""

import json
import os

# =============================================================================
# Parameters from the codebase
# =============================================================================
N_GENERATIONS = 12
POP_SIZE = 20
VALIDATE_TOP = 5
FULL_EPOCHS = 15
EARLY_EPOCHS = 3
CALIBRATION_CONFIGS = 30
CALIBRATION_DATASETS = 3
GRID_BASE_CONFIGS = 46  # from generate_configs() — without unfreeze/head/pooling
# Full search space: 6 ranks × 2 targets × 3 placements × 4 unfreeze × 3 heads × 4 pooling = 1728 LoRA
#                   + 3 dims × 3 placements × 4 unfreeze × 3 heads × 4 pooling = 432 bottleneck + 1 baseline
TOTAL_GRID_CONFIGS = 2161  # full combinatorial space that evolution searches over
RANDOM_SEARCH_CONFIGS = 20  # same as pop_size

# Timing estimates (seconds) on A10G
TIME_15EP_LOW = 60
TIME_15EP_HIGH = 90
TIME_3EP_LOW = 20
TIME_3EP_HIGH = 30
TIME_STATS_LOW = 30  # forward+backward pass for statistics
TIME_STATS_HIGH = 45

# A10G cost on Modal: $1.10/hr (as of 2024)
A10G_COST_PER_HOUR = 1.10


def midpoint(low, high):
    return (low + high) / 2.0


def compute_costs():
    methods = {}

    # =========================================================================
    # 1. Evo-3epoch
    #    Search: 12 gen x 20 configs x 3-epoch finetune
    #    Validation: 5 configs x 15-epoch finetune
    # =========================================================================
    evo3_search_evals = N_GENERATIONS * POP_SIZE  # 240
    evo3_val_evals = VALIDATE_TOP  # 5

    evo3_search_time_low = evo3_search_evals * TIME_3EP_LOW
    evo3_search_time_high = evo3_search_evals * TIME_3EP_HIGH
    evo3_val_time_low = evo3_val_evals * TIME_15EP_LOW
    evo3_val_time_high = evo3_val_evals * TIME_15EP_HIGH

    evo3_total_low = evo3_search_time_low + evo3_val_time_low
    evo3_total_high = evo3_search_time_high + evo3_val_time_high

    methods["Evo-3epoch"] = {
        "search_gpu_calls": evo3_search_evals,
        "search_type": "3-epoch finetune",
        "validation_gpu_calls": evo3_val_evals,
        "validation_type": "15-epoch finetune",
        "total_gpu_calls": evo3_search_evals + evo3_val_evals,
        "gpu_seconds_low": evo3_total_low,
        "gpu_seconds_high": evo3_total_high,
        "gpu_minutes_mid": midpoint(evo3_total_low, evo3_total_high) / 60,
        "calibration_overhead": False,
    }

    # =========================================================================
    # 2. Evo-GP-Proxy (zero-cost proxy via learned GP formula)
    #    Search: 12 gen x 20 configs x 1 statistics computation
    #    Validation: 5 configs x 15-epoch finetune
    #    + one-time calibration cost (computed separately)
    # =========================================================================
    gp_search_evals = N_GENERATIONS * POP_SIZE  # 240
    gp_val_evals = VALIDATE_TOP  # 5

    gp_search_time_low = gp_search_evals * TIME_STATS_LOW
    gp_search_time_high = gp_search_evals * TIME_STATS_HIGH
    gp_val_time_low = gp_val_evals * TIME_15EP_LOW
    gp_val_time_high = gp_val_evals * TIME_15EP_HIGH

    gp_total_low = gp_search_time_low + gp_val_time_low
    gp_total_high = gp_search_time_high + gp_val_time_high

    methods["Evo-GP-Proxy"] = {
        "search_gpu_calls": gp_search_evals,
        "search_type": "statistics (fwd+bwd pass)",
        "validation_gpu_calls": gp_val_evals,
        "validation_type": "15-epoch finetune",
        "total_gpu_calls": gp_search_evals + gp_val_evals,
        "gpu_seconds_low": gp_total_low,
        "gpu_seconds_high": gp_total_high,
        "gpu_minutes_mid": midpoint(gp_total_low, gp_total_high) / 60,
        "calibration_overhead": True,
    }

    # =========================================================================
    # 3. Random search
    #    20 configs x 15-epoch finetune (pick best 5, already trained)
    # =========================================================================
    rand_evals = RANDOM_SEARCH_CONFIGS  # 20

    rand_time_low = rand_evals * TIME_15EP_LOW
    rand_time_high = rand_evals * TIME_15EP_HIGH

    methods["Random search"] = {
        "search_gpu_calls": rand_evals,
        "search_type": "15-epoch finetune",
        "validation_gpu_calls": 0,
        "validation_type": "N/A (already trained)",
        "total_gpu_calls": rand_evals,
        "gpu_seconds_low": rand_time_low,
        "gpu_seconds_high": rand_time_high,
        "gpu_minutes_mid": midpoint(rand_time_low, rand_time_high) / 60,
        "calibration_overhead": False,
    }

    # =========================================================================
    # 4. Full grid search
    #    46 configs x 15-epoch finetune
    # =========================================================================
    grid_evals = TOTAL_GRID_CONFIGS  # 46

    grid_time_low = grid_evals * TIME_15EP_LOW
    grid_time_high = grid_evals * TIME_15EP_HIGH

    methods["Full grid search"] = {
        "search_gpu_calls": grid_evals,
        "search_type": "15-epoch finetune",
        "validation_gpu_calls": 0,
        "validation_type": "N/A (exhaustive)",
        "total_gpu_calls": grid_evals,
        "gpu_seconds_low": grid_time_low,
        "gpu_seconds_high": grid_time_high,
        "gpu_minutes_mid": midpoint(grid_time_low, grid_time_high) / 60,
        "calibration_overhead": False,
    }

    # =========================================================================
    # 5. Calibration cost (one-time for GP proxy)
    #    30 configs x (statistics + 15-epoch finetune) x 3 datasets
    #    GP search itself is CPU-only, negligible
    # =========================================================================
    cal_stats_calls = CALIBRATION_CONFIGS * CALIBRATION_DATASETS  # 90
    cal_ft_calls = CALIBRATION_CONFIGS * CALIBRATION_DATASETS  # 90

    cal_stats_time_low = cal_stats_calls * TIME_STATS_LOW
    cal_stats_time_high = cal_stats_calls * TIME_STATS_HIGH
    cal_ft_time_low = cal_ft_calls * TIME_15EP_LOW
    cal_ft_time_high = cal_ft_calls * TIME_15EP_HIGH

    cal_total_low = cal_stats_time_low + cal_ft_time_low
    cal_total_high = cal_stats_time_high + cal_ft_time_high

    calibration = {
        "stats_gpu_calls": cal_stats_calls,
        "finetune_gpu_calls": cal_ft_calls,
        "total_gpu_calls": cal_stats_calls + cal_ft_calls,
        "gpu_seconds_low": cal_total_low,
        "gpu_seconds_high": cal_total_high,
        "gpu_minutes_mid": midpoint(cal_total_low, cal_total_high) / 60,
        "note": "One-time cost across all future datasets; amortized over N_datasets uses",
    }

    # =========================================================================
    # Compute relative costs and GP-Proxy with calibration amortized
    # =========================================================================
    grid_mid = midpoint(grid_time_low, grid_time_high)

    for name, m in methods.items():
        m_mid = midpoint(m["gpu_seconds_low"], m["gpu_seconds_high"])
        m["relative_to_grid"] = round(m_mid / grid_mid, 3)

    # GP-Proxy total cost including calibration (1 dataset amortization)
    gp_with_cal_low = gp_total_low + cal_total_low
    gp_with_cal_high = gp_total_high + cal_total_high
    gp_with_cal_mid = midpoint(gp_with_cal_low, gp_with_cal_high)

    # GP-Proxy with calibration amortized over 3 datasets
    cal_per_ds_low = cal_total_low / CALIBRATION_DATASETS
    cal_per_ds_high = cal_total_high / CALIBRATION_DATASETS
    gp_amortized_low = gp_total_low + cal_per_ds_low
    gp_amortized_high = gp_total_high + cal_per_ds_high

    # =========================================================================
    # Dollar costs
    # =========================================================================
    for name, m in methods.items():
        m_mid_hrs = midpoint(m["gpu_seconds_low"], m["gpu_seconds_high"]) / 3600
        m["estimated_cost_usd"] = round(m_mid_hrs * A10G_COST_PER_HOUR, 2)

    cal_mid_hrs = midpoint(cal_total_low, cal_total_high) / 3600
    calibration["estimated_cost_usd"] = round(cal_mid_hrs * A10G_COST_PER_HOUR, 2)

    # =========================================================================
    # Build output JSON
    # =========================================================================
    output = {
        "parameters": {
            "gpu": "NVIDIA A10G (Modal)",
            "n_generations": N_GENERATIONS,
            "pop_size": POP_SIZE,
            "validate_top": VALIDATE_TOP,
            "full_epochs": FULL_EPOCHS,
            "early_epochs": EARLY_EPOCHS,
            "calibration_configs": CALIBRATION_CONFIGS,
            "calibration_datasets": CALIBRATION_DATASETS,
            "total_grid_configs": TOTAL_GRID_CONFIGS,
            "timing_estimates": {
                "15_epoch_finetune_seconds": [TIME_15EP_LOW, TIME_15EP_HIGH],
                "3_epoch_finetune_seconds": [TIME_3EP_LOW, TIME_3EP_HIGH],
                "statistics_computation_seconds": [TIME_STATS_LOW, TIME_STATS_HIGH],
            },
            "a10g_cost_per_hour_usd": A10G_COST_PER_HOUR,
        },
        "methods": methods,
        "calibration": calibration,
        "gp_proxy_with_calibration": {
            "total_gpu_calls": methods["Evo-GP-Proxy"]["total_gpu_calls"] + calibration["total_gpu_calls"],
            "gpu_seconds_low": gp_with_cal_low,
            "gpu_seconds_high": gp_with_cal_high,
            "gpu_minutes_mid": gp_with_cal_mid / 60,
            "relative_to_grid": round(gp_with_cal_mid / grid_mid, 3),
            "estimated_cost_usd": round((gp_with_cal_mid / 3600) * A10G_COST_PER_HOUR, 2),
            "note": "Full cost if calibration is used for only 1 dataset",
        },
        "gp_proxy_amortized_3ds": {
            "gpu_seconds_low": gp_amortized_low,
            "gpu_seconds_high": gp_amortized_high,
            "gpu_minutes_mid": midpoint(gp_amortized_low, gp_amortized_high) / 60,
            "relative_to_grid": round(midpoint(gp_amortized_low, gp_amortized_high) / grid_mid, 3),
            "note": "GP-Proxy cost with calibration amortized over 3 datasets",
        },
    }

    return output


def print_table(output):
    methods = output["methods"]
    cal = output["calibration"]
    gp_full = output["gp_proxy_with_calibration"]
    gp_amort = output["gp_proxy_amortized_3ds"]

    print("=" * 100)
    print("GPU COST ANALYSIS — Adapter Selection Methods (per dataset, A10G)")
    print("=" * 100)

    # Table 1: Main comparison
    print()
    print(f"{'Method':<28} {'GPU Calls':>10} {'GPU-min (est.)':>16} {'Rel. to Grid':>14} {'Est. USD':>10}")
    print("-" * 80)

    for name, m in methods.items():
        lo = m["gpu_seconds_low"] / 60
        hi = m["gpu_seconds_high"] / 60
        mid = m["gpu_minutes_mid"]
        rel = m["relative_to_grid"]
        usd = m["estimated_cost_usd"]
        marker = " *" if m["calibration_overhead"] else ""
        print(f"{name:<28} {m['total_gpu_calls']:>10} {lo:>6.1f} - {hi:>5.1f}{marker:2s} {rel:>13.1%} {'$'+f'{usd:.2f}':>10}")

    print("-" * 80)
    cal_usd = cal["estimated_cost_usd"]
    print(f"{'Calibration (one-time)':<28} {cal['total_gpu_calls']:>10} "
          f"{cal['gpu_seconds_low']/60:>6.1f} - {cal['gpu_seconds_high']/60:>5.1f}   "
          f"{'':>13} {'$' + f'{cal_usd:.2f}':>10}")
    print()
    print("* Excludes calibration overhead (shown separately)")

    # Table 2: GP-Proxy with calibration
    print()
    print("-" * 80)
    print("GP-Proxy Total Cost (including calibration):")
    print("-" * 80)
    print(f"  {'If calibration used for 1 dataset:':<42} "
          f"{gp_full['gpu_minutes_mid']:>6.1f} min  "
          f"{gp_full['relative_to_grid']:>6.1%} of grid  "
          f"${gp_full['estimated_cost_usd']:.2f}")
    print(f"  {'If calibration amortized over 3 datasets:':<42} "
          f"{gp_amort['gpu_minutes_mid']:>6.1f} min  "
          f"{gp_amort['relative_to_grid']:>6.1%} of grid")

    # Table 3: Breakdown
    print()
    print("=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)
    print()

    rows = [
        ("Evo-3epoch search", f"{N_GENERATIONS} gen x {POP_SIZE} cfg x 3ep finetune",
         N_GENERATIONS * POP_SIZE, f"{TIME_3EP_LOW}-{TIME_3EP_HIGH}s each"),
        ("Evo-3epoch validation", f"{VALIDATE_TOP} cfg x 15ep finetune",
         VALIDATE_TOP, f"{TIME_15EP_LOW}-{TIME_15EP_HIGH}s each"),
        ("", "", 0, ""),
        ("Evo-GP-Proxy search", f"{N_GENERATIONS} gen x {POP_SIZE} cfg x stats",
         N_GENERATIONS * POP_SIZE, f"{TIME_STATS_LOW}-{TIME_STATS_HIGH}s each"),
        ("Evo-GP-Proxy validation", f"{VALIDATE_TOP} cfg x 15ep finetune",
         VALIDATE_TOP, f"{TIME_15EP_LOW}-{TIME_15EP_HIGH}s each"),
        ("", "", 0, ""),
        ("Random search", f"{RANDOM_SEARCH_CONFIGS} cfg x 15ep finetune",
         RANDOM_SEARCH_CONFIGS, f"{TIME_15EP_LOW}-{TIME_15EP_HIGH}s each"),
        ("", "", 0, ""),
        ("Full grid search", f"{TOTAL_GRID_CONFIGS} cfg x 15ep finetune",
         TOTAL_GRID_CONFIGS, f"{TIME_15EP_LOW}-{TIME_15EP_HIGH}s each"),
        ("", "", 0, ""),
        ("Calibration (stats)", f"{CALIBRATION_CONFIGS} cfg x {CALIBRATION_DATASETS} datasets",
         CALIBRATION_CONFIGS * CALIBRATION_DATASETS, f"{TIME_STATS_LOW}-{TIME_STATS_HIGH}s each"),
        ("Calibration (finetune)", f"{CALIBRATION_CONFIGS} cfg x {CALIBRATION_DATASETS} datasets x 15ep",
         CALIBRATION_CONFIGS * CALIBRATION_DATASETS, f"{TIME_15EP_LOW}-{TIME_15EP_HIGH}s each"),
    ]

    print(f"{'Component':<30} {'Formula':<40} {'Calls':>6} {'Timing':>20}")
    print("-" * 100)
    for label, formula, calls, timing in rows:
        if not label:
            print()
            continue
        print(f"{label:<30} {formula:<40} {calls:>6} {timing:>20}")

    # Key insight
    print()
    print("=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    evo3_mid = methods["Evo-3epoch"]["gpu_minutes_mid"]
    gp_mid = methods["Evo-GP-Proxy"]["gpu_minutes_mid"]
    grid_mid = methods["Full grid search"]["gpu_minutes_mid"]
    rand_mid = methods["Random search"]["gpu_minutes_mid"]

    print(f"  - Evo-3epoch is {methods['Evo-3epoch']['relative_to_grid']:.0%} of grid search cost ({evo3_mid:.0f} vs {grid_mid:.0f} GPU-min)")
    print(f"  - Evo-GP-Proxy (excl. calibration) is {methods['Evo-GP-Proxy']['relative_to_grid']:.0%} of grid search ({gp_mid:.0f} vs {grid_mid:.0f} GPU-min)")
    print(f"  - Random search is {methods['Random search']['relative_to_grid']:.0%} of grid search ({rand_mid:.0f} vs {grid_mid:.0f} GPU-min)")
    print(f"  - Calibration is a {cal['gpu_minutes_mid']:.0f} GPU-min one-time investment across {CALIBRATION_DATASETS} datasets")
    print(f"  - GP-Proxy breaks even vs Evo-3epoch after calibration if used on >= "
          f"{cal['gpu_minutes_mid'] / max(evo3_mid - gp_mid, 0.01):.1f} datasets")
    print(f"  - Evo-3epoch explores {N_GENERATIONS * POP_SIZE} configs (with evolution) vs {TOTAL_GRID_CONFIGS} in grid = {N_GENERATIONS * POP_SIZE / TOTAL_GRID_CONFIGS:.1f}x more configs evaluated")


if __name__ == "__main__":
    output = compute_costs()

    # Save JSON
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cost_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved structured data to {json_path}\n")

    print_table(output)
