"""Budget scaling ablation for AAS paper.

Tests whether:
1. Performance improves with more search budget (space is rich)
2. Evolutionary selection adds value over pure random sampling

Usage:
    python scripts/run_budget_ablation.py --dataset ETTh1
    python scripts/run_budget_ablation.py --dataset ETTm1
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)
from run_ablations import run_random_code_ablation
from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS


BUDGET_CONFIGS = [
    # (label, pop_size, n_generations, is_pure_random)
    ("random-20",  20,  1, True),
    ("random-60",  60,  1, True),
    ("random-120", 120, 1, True),
    ("random-240", 240, 1, True),
    ("evo-60",     20,  3, False),
    ("evo-120",    20,  6, False),
    # evo-240 = default, already exists as {dataset}_H96_42.json
    ("evo-480",    20, 24, False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/budget_ablation", exist_ok=True)

    # Load model
    print("Loading %s..." % args.backbone)
    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks)-4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True
    print("Unfreezing last 4 encoder blocks")

    # Load data
    splits, n_ch = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]
    X_test, Y_test = splits["test"]
    print("%s: train=%d, val=%d, test=%d" % (
        args.dataset, len(X_train), len(X_val), len(X_test)))

    def evaluate_fn(code_strs):
        results = []
        for code in code_strs:
            try:
                result = train_adapter(
                    code, model, blocks, X_train, Y_train, X_val, Y_val,
                    device=args.device, n_epochs=3, forecast_horizon=args.horizon,
                    backbone_type=bb_type,
                )
                results.append(result)
            except Exception as e:
                results.append({"error": "%s: %s" % (type(e).__name__, e)})
        return results

    all_results = {}

    for label, pop_size, n_gens, is_pure_random in BUDGET_CONFIGS:
        total_budget = pop_size * n_gens
        print("\n" + "=" * 60)
        print("BUDGET ABLATION: %s | %s H=%d | budget=%d (%dx%d)" % (
            label, args.dataset, args.horizon, total_budget, pop_size, n_gens))
        print("=" * 60)

        start = time.time()
        result = run_random_code_ablation(
            evaluate_fn=evaluate_fn,
            n_generations=n_gens,
            pop_size=pop_size,
            seed=args.seed,
        )
        elapsed = time.time() - start

        # Evaluate top-5 on TEST set at 15 epochs
        valid_pop = [
            ind for ind in result["final_population"]
            if ind["error"] is None and ind["mse"] < float('inf')
        ][:5]

        test_results = []
        for ind in valid_pop:
            try:
                tr = train_adapter(
                    ind["code"], model, blocks, X_train, Y_train, X_test, Y_test,
                    device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                    backbone_type=bb_type,
                )
                test_results.append({
                    "val_mse": ind["mse"],
                    "test_mse": tr["mse"],
                    "test_mae": tr["mae"],
                    "param_count": tr["param_count"],
                })
            except Exception:
                pass

        best_test_mse = min((r["test_mse"] for r in test_results), default=float("inf"))

        all_results[label] = {
            "pop_size": pop_size,
            "n_generations": n_gens,
            "total_budget": total_budget,
            "is_pure_random": is_pure_random,
            "best_val_mse": result["best_mse"],
            "best_test_mse": best_test_mse,
            "elapsed": elapsed,
            "top5_test": test_results,
        }

        print("  best_test_mse=%.4f  elapsed=%.0fs" % (best_test_mse, elapsed))

    # Also evaluate baselines for reference
    print("\nBaseline adapters on TEST set (15 epochs):")
    baselines = {
        "linear": SEED_ADAPTERS[0],
        "attention": SEED_ADAPTERS[3],
        "conv": SEED_ADAPTERS[4],
    }
    baseline_results = {}
    for name, code in baselines.items():
        val = validate_adapter_code(code, d_model=hdim, output_dim=args.horizon)
        if not val["valid"]:
            continue
        try:
            tr = train_adapter(
                code, model, blocks, X_train, Y_train, X_test, Y_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                backbone_type=bb_type,
            )
            baseline_results[name] = tr
            print("  %-15s test_mse=%.4f" % (name, tr["mse"]))
        except Exception as e:
            print("  %s: ERROR %s" % (name, e))

    # Summary table
    print("\n" + "=" * 60)
    print("BUDGET ABLATION SUMMARY: %s H=%d" % (args.dataset, args.horizon))
    print("=" * 60)
    print("%-15s %8s %10s %10s" % ("Label", "Budget", "Test MSE", "Time(s)"))
    print("-" * 50)
    for label, info in sorted(all_results.items(), key=lambda x: x[1]["total_budget"]):
        print("%-15s %8d %10.4f %10.0f" % (
            label, info["total_budget"], info["best_test_mse"], info["elapsed"]))

    best_baseline_name = min(baseline_results, key=lambda k: baseline_results[k]["mse"]) if baseline_results else None
    if best_baseline_name:
        print("%-15s %8s %10.4f" % (
            "baseline(%s)" % best_baseline_name, "—",
            baseline_results[best_baseline_name]["mse"]))

    # Save
    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "results": all_results,
        "baselines": {k: v for k, v in baseline_results.items()},
    }
    path = "results/budget_ablation/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nSaved to %s" % path)


if __name__ == "__main__":
    main()
