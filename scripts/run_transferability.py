"""Cross-dataset transferability experiment for AAS paper.

Takes the best adapter architecture discovered on each dataset and evaluates
it on ALL other datasets (no re-evolution, just training).

Produces a transfer matrix:
  Row = source dataset (where architecture was discovered)
  Col = target dataset (where it's evaluated)

If off-diagonal wins exist, discovered architectures have some generalization.

Usage:
    python scripts/run_transferability.py
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.code_evolution import validate_adapter_code, SEED_ADAPTERS
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]


def load_winning_codes(results_dir="results/standard_evolution"):
    """Extract best adapter code from each dataset's evolution results."""
    codes = {}
    for dataset in DATASETS:
        path = os.path.join(results_dir, "%s_H96_42.json" % dataset)
        if not os.path.exists(path):
            print("WARNING: %s not found, skipping" % path)
            continue
        d = json.load(open(path))
        evolved = d.get("evolved", [])
        if not evolved:
            continue
        best = min(evolved, key=lambda x: x["test_mse"])
        codes[dataset] = {
            "code": best["code"],
            "source_mse": best["test_mse"],
            "param_count": best["param_count"],
        }
    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/transferability", exist_ok=True)

    # Load model once
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

    # Load winning adapter codes
    codes = load_winning_codes()
    print("Loaded winning codes from: %s" % list(codes.keys()))

    # Also include the best fixed baseline (conv) for comparison
    baseline_code = SEED_ADAPTERS[4]  # conv adapter

    # Load all datasets
    all_data = {}
    for dataset in DATASETS:
        try:
            splits, n_ch = load_standard_data(dataset, args.horizon)
            all_data[dataset] = splits
            print("Loaded %s: train=%d, val=%d, test=%d" % (
                dataset, len(splits["train"][0]), len(splits["val"][0]), len(splits["test"][0])))
        except Exception as e:
            print("ERROR loading %s: %s" % (dataset, e))

    # Build transfer matrix
    # transfer_matrix[source][target] = test_mse
    transfer_matrix = {}
    baseline_mses = {}

    # First: evaluate baseline on all targets
    print("\n" + "=" * 60)
    print("BASELINE (conv) on all datasets")
    print("=" * 60)
    for target in DATASETS:
        if target not in all_data:
            continue
        splits = all_data[target]
        X_train, Y_train = splits["train"]
        X_test, Y_test = splits["test"]

        val = validate_adapter_code(baseline_code, d_model=hdim, output_dim=args.horizon)
        if not val["valid"]:
            print("  %s: baseline INVALID" % target)
            continue

        try:
            tr = train_adapter(
                baseline_code, model, blocks, X_train, Y_train, X_test, Y_test,
                device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                backbone_type=bb_type,
            )
            baseline_mses[target] = tr["mse"]
            print("  %-10s MSE=%.4f" % (target, tr["mse"]))
        except Exception as e:
            print("  %-10s ERROR: %s" % (target, e))

    # Then: cross-evaluate each discovered architecture
    print("\n" + "=" * 60)
    print("TRANSFER MATRIX (source architecture → target dataset)")
    print("=" * 60)

    for source in DATASETS:
        if source not in codes:
            continue
        transfer_matrix[source] = {}
        adapter_code = codes[source]["code"]

        for target in DATASETS:
            if target not in all_data:
                continue
            splits = all_data[target]
            X_train, Y_train = splits["train"]
            X_test, Y_test = splits["test"]

            # Validate adapter for this target's horizon
            val = validate_adapter_code(adapter_code, d_model=hdim, output_dim=args.horizon)
            if not val["valid"]:
                print("  %s → %s: INVALID (%s)" % (source, target, val["error"]))
                transfer_matrix[source][target] = {"error": val["error"]}
                continue

            try:
                tr = train_adapter(
                    adapter_code, model, blocks, X_train, Y_train, X_test, Y_test,
                    device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                    backbone_type=bb_type,
                )
                is_diagonal = source == target
                baseline = baseline_mses.get(target, float("inf"))
                beats_baseline = tr["mse"] < baseline
                delta = (tr["mse"] - baseline) / baseline * 100 if baseline < float("inf") else 0

                transfer_matrix[source][target] = {
                    "test_mse": tr["mse"],
                    "test_mae": tr["mae"],
                    "param_count": tr["param_count"],
                    "beats_baseline": beats_baseline,
                    "delta_vs_baseline": delta,
                }

                marker = "DIAG" if is_diagonal else ("WIN" if beats_baseline else "lose")
                print("  %s → %-10s MSE=%.4f  baseline=%.4f  Δ=%+.1f%%  %s" % (
                    source, target, tr["mse"], baseline, delta, marker))
            except Exception as e:
                print("  %s → %-10s ERROR: %s" % (source, target, e))
                transfer_matrix[source][target] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("TRANSFER SUMMARY")
    print("=" * 60)

    total_off_diagonal = 0
    wins_off_diagonal = 0
    for source in transfer_matrix:
        for target in transfer_matrix[source]:
            if source == target:
                continue
            entry = transfer_matrix[source][target]
            if isinstance(entry, dict) and "beats_baseline" in entry:
                total_off_diagonal += 1
                if entry["beats_baseline"]:
                    wins_off_diagonal += 1

    print("Off-diagonal transfers: %d/%d beat baseline (%.0f%%)" % (
        wins_off_diagonal, total_off_diagonal,
        100 * wins_off_diagonal / total_off_diagonal if total_off_diagonal > 0 else 0))

    # Print compact matrix
    print("\nTransfer matrix (Δ%% vs baseline, bold = wins):")
    header = "%-10s" % "Source→"
    for t in DATASETS:
        header += " %10s" % t
    print(header)
    for source in DATASETS:
        if source not in transfer_matrix:
            continue
        row = "%-10s" % source
        for target in DATASETS:
            entry = transfer_matrix.get(source, {}).get(target, {})
            if "delta_vs_baseline" in entry:
                delta = entry["delta_vs_baseline"]
                marker = "*" if entry["beats_baseline"] and source != target else ""
                row += " %9.1f%s" % (delta, marker)
            else:
                row += " %10s" % "—"
        print(row)

    # Save
    save_data = {
        "horizon": args.horizon,
        "transfer_matrix": transfer_matrix,
        "baseline_mses": baseline_mses,
        "source_codes": {k: {"source_mse": v["source_mse"], "param_count": v["param_count"]}
                         for k, v in codes.items()},
    }
    path = "results/transferability/transfer_H%d.json" % args.horizon
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nSaved to %s" % path)


if __name__ == "__main__":
    main()
