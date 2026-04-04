"""Load results and generate plots + statistics.

Usage:
    python scripts/run_analysis.py --sweep results/sweep_results.json
    python scripts/run_analysis.py --sweep results/sweep_results.json --finetune results/finetune_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feasibility.viz import (
    plot_score_heatmap,
    plot_component_variance,
    plot_smoothness,
    plot_correlation,
    print_success_criteria,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="Path to sweep results JSON")
    parser.add_argument("--finetune", default=None, help="Path to finetune results JSON")
    args = parser.parse_args()

    with open(args.sweep) as f:
        sweep_results = json.load(f)

    finetune_results = None
    if args.finetune:
        with open(args.finetune) as f:
            finetune_results = json.load(f)

    # Group by dataset
    by_dataset = {}
    for r in sweep_results:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    ft_by_dataset = {}
    if finetune_results:
        for r in finetune_results:
            ds = r.get("dataset", "unknown")
            ft_by_dataset.setdefault(ds, []).append(r)

    for ds_name, results in by_dataset.items():
        print(f"\n{'#'*60}")
        print(f"# Dataset: {ds_name} ({len(results)} configs)")
        print(f"{'#'*60}")

        plot_score_heatmap(results, ds_name)
        print(f"  Saved heatmap_{ds_name}.png")

        plot_component_variance(results, ds_name)
        print(f"  Saved component_variance_{ds_name}.png")

        plot_smoothness(results, ds_name)
        print(f"  Saved smoothness_{ds_name}.png")

        ft = ft_by_dataset.get(ds_name)
        if ft:
            metric_key = "mse" if any("mse" in r for r in ft) else "accuracy"
            plot_correlation(results, ft, ds_name, metric_key=metric_key)
            print(f"  Saved correlation_{ds_name}.png")

        print_success_criteria(results, ds_name, ft,
                               metric_key="mse" if ds_name == "ETTh1" else "accuracy")

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
