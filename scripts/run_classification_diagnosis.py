"""Diagnose classification performance on EthanolConcentration.

Selects top-3 and bottom-3 TEMPLATE-scored configs, fine-tunes with
adapter params trainable, and checks if TEMPLATE score predicts accuracy.

Usage:
    python scripts/run_classification_diagnosis.py --results results/sweep_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from feasibility.config import AdapterConfig
from feasibility.data import load_ethanol_concentration
from feasibility.finetune import finetune_classification


def select_top_bottom(results: list, n: int = 3) -> list:
    """Select top-n and bottom-n configs by TEMPLATE composite score."""
    sorted_results = sorted(results, key=lambda r: r["scores"]["composite"])
    bottom = sorted_results[:n]
    top = sorted_results[-n:]
    return bottom + top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-select", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.results) as f:
        all_results = json.load(f)

    ec_results = [r for r in all_results if r.get("dataset") == "EthanolConcentration"]
    if not ec_results:
        print("No EthanolConcentration results found in sweep file")
        return

    selected = select_top_bottom(ec_results, n=args.n_select)
    print(f"Selected {len(selected)} configs (top-{args.n_select} + bottom-{args.n_select})")
    for r in selected:
        print(f"  {r['config']['config_id']}: composite={r['scores']['composite']:.4f}")

    print(f"\nLoading EthanolConcentration dataset...")
    ds = load_ethanol_concentration()
    samples = ds["samples"]
    labels = ds["labels"]
    print(f"  Samples: {samples.shape}, Labels: {labels.shape}")
    print(f"  Classes: {np.unique(labels)}, counts: {np.bincount(labels)}")

    print(f"\nFine-tuning with {args.n_epochs} epochs, device={args.device}")
    print(f"{'='*70}")

    ft_results = []
    for i, r in enumerate(selected):
        cfg = AdapterConfig.from_dict(r["config"])
        composite = r["scores"]["composite"]
        print(f"\n[{i+1}/{len(selected)}] {cfg.config_id} (TEMPLATE={composite:.4f})")

        try:
            metrics = finetune_classification(
                cfg, samples, labels,
                device=args.device,
                n_epochs=args.n_epochs,
                lr=1e-3,
                batch_size=64,
            )
            accuracy = metrics["accuracy"]
            print(f"  => accuracy = {accuracy:.4f}")

            ft_results.append({
                "config_id": cfg.config_id,
                "composite": composite,
                "accuracy": accuracy,
                "config": r["config"],
            })
        except Exception as e:
            print(f"  => FAILED: {e}")
            import traceback
            traceback.print_exc()

    if len(ft_results) < 4:
        print("\nToo few results for correlation analysis")
        return

    # Correlation analysis
    from scipy.stats import kendalltau, spearmanr

    composites = [r["composite"] for r in ft_results]
    accuracies = [r["accuracy"] for r in ft_results]

    tau, tau_p = kendalltau(composites, accuracies)
    rho, rho_p = spearmanr(composites, accuracies)

    print(f"\n{'='*70}")
    print(f"DIAGNOSIS RESULTS")
    print(f"{'='*70}")
    print(f"{'Config ID':<35} {'TEMPLATE':>10} {'Accuracy':>10}")
    print(f"{'-'*55}")
    for r in sorted(ft_results, key=lambda x: x["composite"]):
        print(f"{r['config_id']:<35} {r['composite']:>10.4f} {r['accuracy']:>10.4f}")

    print(f"\nKendall tau  = {tau:.3f} (p={tau_p:.4f})")
    print(f"Spearman rho = {rho:.3f} (p={rho_p:.4f})")

    if tau > 0.3 and tau_p < 0.1:
        print("\nVERDICT: Correlation exists. Classification issue was random init + frozen adapter.")
        print("Fix: The finetune code now trains adapter params. Re-run full classification sweep.")
    elif all(a < 0.1 for a in accuracies):
        print("\nVERDICT: All accuracies near zero. Problem is deeper than init.")
        print("Possible causes: MOMENT features not discriminative for this task,")
        print("or channel-independent approach loses class signal.")
    else:
        print("\nVERDICT: Mixed results. TEMPLATE scores don't predict classification performance.")
        print("Consider scoping classification out of the paper's claims.")

    # Save results
    out_path = "results/classification_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump({
            "selected_configs": ft_results,
            "kendall_tau": tau,
            "kendall_p": tau_p,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "n_epochs": args.n_epochs,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
