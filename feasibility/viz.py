"""Plotting score landscapes and analysis figures."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional


FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def plot_score_heatmap(results: List[Dict], dataset_name: str, save: bool = True):
    """Score heatmap: LoRA rank (x) vs target_modules × placement (y), color = composite."""
    lora_results = [r for r in results if r["config"]["adapter_type"] == "lora"]
    if not lora_results:
        return

    ranks = sorted(set(r["config"]["lora_rank"] for r in lora_results))
    combos = sorted(set(
        (r["config"]["target_modules_key"], r["config"]["layer_placement"])
        for r in lora_results
    ))

    matrix = np.full((len(combos), len(ranks)), np.nan)
    for r in lora_results:
        rank = r["config"]["lora_rank"]
        combo = (r["config"]["target_modules_key"], r["config"]["layer_placement"])
        i = combos.index(combo)
        j = ranks.index(rank)
        matrix[i, j] = r["scores"]["composite"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".3f",
        xticklabels=[str(r) for r in ranks],
        yticklabels=[f"{tm}_{pl}" for tm, pl in combos],
        cmap="YlOrRd",
    )
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Target Modules × Placement")
    ax.set_title(f"TEMPLATE Composite Score — {dataset_name}")
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / f"heatmap_{dataset_name}.png", dpi=150)
    return fig


def plot_component_variance(results: List[Dict], dataset_name: str, save: bool = True):
    """Bar chart of DL, PL, TA score variance across all configs."""
    scores = {k: [] for k in ["dl", "pl", "ta"]}
    for r in results:
        for k in scores:
            scores[k].append(r["scores"][k])

    means = {k: np.mean(v) for k, v in scores.items()}
    stds = {k: np.std(v) for k, v in scores.items()}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ["dl", "pl", "ta"]):
        vals = scores[key]
        ax.hist(vals, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(means[key], color="red", linestyle="--", label=f"mean={means[key]:.3f}")
        ax.set_title(f"{key.upper()} (std={stds[key]:.4f})")
        ax.set_xlabel("Score")
        ax.legend()

    plt.suptitle(f"Score Component Distributions — {dataset_name}")
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / f"component_variance_{dataset_name}.png", dpi=150)
    return fig


def plot_smoothness(results: List[Dict], dataset_name: str, save: bool = True):
    """For LoRA configs differing by one rank step, plot Δscore vs Δrank."""
    lora_results = [r for r in results if r["config"]["adapter_type"] == "lora"]
    ranks = sorted(set(r["config"]["lora_rank"] for r in lora_results))

    # Group by (target_modules, placement)
    groups = {}
    for r in lora_results:
        key = (r["config"]["target_modules_key"], r["config"]["layer_placement"])
        groups.setdefault(key, {})[r["config"]["lora_rank"]] = r["scores"]["composite"]

    adjacent_deltas = []
    skip_one_deltas = []

    for key, rank_scores in groups.items():
        sorted_ranks = sorted(rank_scores.keys())
        for i in range(len(sorted_ranks) - 1):
            r1, r2 = sorted_ranks[i], sorted_ranks[i + 1]
            adjacent_deltas.append(abs(rank_scores[r2] - rank_scores[r1]))
        for i in range(len(sorted_ranks) - 2):
            r1, r3 = sorted_ranks[i], sorted_ranks[i + 2]
            skip_one_deltas.append(abs(rank_scores[r3] - rank_scores[r1]))

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = [1, 2]
    bp = ax.boxplot(
        [adjacent_deltas, skip_one_deltas],
        positions=positions,
        widths=0.5,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightsalmon")

    ax.set_xticks(positions)
    ax.set_xticklabels(["Adjacent Δrank", "Skip-one Δrank"])
    ax.set_ylabel("|Δ composite score|")
    ax.set_title(f"Score Smoothness — {dataset_name}")

    # Add ratio annotation
    adj_mean = np.mean(adjacent_deltas) if adjacent_deltas else 0
    skip_mean = np.mean(skip_one_deltas) if skip_one_deltas else 0
    ratio = skip_mean / adj_mean if adj_mean > 0 else float("inf")
    ax.text(0.95, 0.95, f"ratio = {ratio:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat"))
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / f"smoothness_{dataset_name}.png", dpi=150)
    return fig


def plot_correlation(
    results: List[Dict],
    finetune_results: List[Dict],
    dataset_name: str,
    metric_key: str = "mse",
    save: bool = True,
):
    """TEMPLATE score vs fine-tuned performance scatter plot with Kendall's tau."""
    from scipy.stats import kendalltau

    # Match configs
    ft_map = {r["config"]["config_id"]: r[metric_key] for r in finetune_results}
    matched_scores = []
    matched_metrics = []

    for r in results:
        cfg_id = r["config"]["config_id"]
        if cfg_id in ft_map:
            matched_scores.append(r["scores"]["composite"])
            matched_metrics.append(ft_map[cfg_id])

    if len(matched_scores) < 3:
        print(f"Warning: only {len(matched_scores)} matched configs for correlation plot")
        return None

    tau, pvalue = kendalltau(matched_scores, matched_metrics)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(matched_scores, matched_metrics, s=60, alpha=0.7, edgecolors="black")
    ax.set_xlabel("TEMPLATE Composite Score")
    ax.set_ylabel(metric_key.upper())
    ax.set_title(f"Score-Performance Correlation — {dataset_name}\nKendall τ = {tau:.3f} (p = {pvalue:.3f})")

    # Trend line
    z = np.polyfit(matched_scores, matched_metrics, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(matched_scores), max(matched_scores), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.5)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / f"correlation_{dataset_name}.png", dpi=150)
    return fig


def print_success_criteria(results: List[Dict], dataset_name: str,
                           finetune_results: Optional[List[Dict]] = None,
                           metric_key: str = "mse"):
    """Evaluate and print the three success criteria."""
    from scipy.stats import kendalltau

    composites = [r["scores"]["composite"] for r in results]
    variance = np.std(composites)

    # Smoothness
    lora_results = [r for r in results if r["config"]["adapter_type"] == "lora"]
    ranks = sorted(set(r["config"]["lora_rank"] for r in lora_results))
    groups = {}
    for r in lora_results:
        key = (r["config"]["target_modules_key"], r["config"]["layer_placement"])
        groups.setdefault(key, {})[r["config"]["lora_rank"]] = r["scores"]["composite"]

    adj, skip = [], []
    for key, rank_scores in groups.items():
        sr = sorted(rank_scores.keys())
        for i in range(len(sr) - 1):
            adj.append(abs(rank_scores[sr[i + 1]] - rank_scores[sr[i]]))
        for i in range(len(sr) - 2):
            skip.append(abs(rank_scores[sr[i + 2]] - rank_scores[sr[i]]))

    adj_mean = np.mean(adj) if adj else 0
    skip_mean = np.mean(skip) if skip else 0
    smooth = adj_mean < 2 * skip_mean if skip_mean > 0 else False
    ratio = skip_mean / adj_mean if adj_mean > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"SUCCESS CRITERIA — {dataset_name}")
    print(f"{'='*60}")
    print(f"1. Variance:   std = {variance:.4f}  {'PASS' if variance > 0.05 else 'FAIL'} (threshold > 0.05)")
    print(f"2. Smoothness: ratio = {ratio:.2f}  {'PASS' if smooth else 'FAIL'} (adjacent < 2× skip-one)")

    if finetune_results:
        ft_map = {r["config"]["config_id"]: r[metric_key] for r in finetune_results}
        scores, metrics = [], []
        for r in results:
            if r["config"]["config_id"] in ft_map:
                scores.append(r["scores"]["composite"])
                metrics.append(ft_map[r["config"]["config_id"]])
        if len(scores) >= 3:
            tau, pval = kendalltau(scores, metrics)
            # For MSE (lower=better), expect negative τ; for accuracy (higher=better), expect positive τ
            effective_tau = -tau if metric_key == "mse" else tau
            print(f"3. Correlation: τ = {tau:.3f} (p={pval:.3f})  {'PASS' if effective_tau > 0.5 else 'FAIL'} (|τ| > 0.5, correct sign)")
        else:
            print(f"3. Correlation: insufficient data ({len(scores)} configs matched)")
    else:
        print("3. Correlation: not computed (no finetune results)")
    print(f"{'='*60}\n")
