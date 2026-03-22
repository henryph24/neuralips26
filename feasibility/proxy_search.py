"""Calibration data collection and GP-based proxy search."""

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import kendalltau

from feasibility.config import (
    AdapterConfig,
    LORA_RANKS,
    BOTTLENECK_DIMS,
    LAYER_PLACEMENTS,
    TARGET_MODULE_SETS,
    UNFREEZE_STRATEGIES,
    HEAD_TYPES,
    POOLING_STRATEGIES,
    _make_id,
)
from feasibility.proxy_gp import (
    Node,
    random_tree,
    point_mutation,
    subtree_crossover,
    evaluate_proxy,
    serialize_tree,
    deserialize_tree,
)


# --- Calibration data ---

@dataclass
class CalibrationData:
    """Ground-truth data for proxy fitness evaluation."""
    configs: List[AdapterConfig]
    statistics: List[Dict[str, float]]
    performances: List[float]
    dataset_name: str
    metric_key: str  # "mse" or "accuracy"
    higher_is_better: bool

    def to_dict(self) -> dict:
        return {
            "configs": [c.to_dict() for c in self.configs],
            "statistics": self.statistics,
            "performances": self.performances,
            "dataset_name": self.dataset_name,
            "metric_key": self.metric_key,
            "higher_is_better": self.higher_is_better,
        }

    @staticmethod
    def from_dict(d: dict) -> "CalibrationData":
        return CalibrationData(
            configs=[AdapterConfig.from_dict(c) for c in d["configs"]],
            statistics=d["statistics"],
            performances=d["performances"],
            dataset_name=d["dataset_name"],
            metric_key=d["metric_key"],
            higher_is_better=d["higher_is_better"],
        )


def select_calibration_configs(n: int = 30, seed: int = 42) -> List[AdapterConfig]:
    """Stratified sampling across all 8 adapter dimensions.

    Generates configs that cover the search space more uniformly than
    pure random sampling, ensuring GP has diverse training signal.
    """
    rng = np.random.default_rng(seed)
    configs = []
    seen_ids = set()

    adapter_types = ["lora", "bottleneck"]
    placement_keys = list(LAYER_PLACEMENTS.keys())
    target_keys = list(TARGET_MODULE_SETS.keys())

    # Stratified: cycle through dimensions
    attempts = 0
    while len(configs) < n and attempts < n * 20:
        attempts += 1

        # Stratified selection: ensure coverage
        idx = len(configs)
        adapter_type = adapter_types[idx % len(adapter_types)]
        placement = placement_keys[idx % len(placement_keys)]
        unfreeze = UNFREEZE_STRATEGIES[idx % len(UNFREEZE_STRATEGIES)]
        head_type = HEAD_TYPES[idx % len(HEAD_TYPES)]
        pooling = POOLING_STRATEGIES[idx % len(POOLING_STRATEGIES)]

        if adapter_type == "lora":
            rank = LORA_RANKS[rng.integers(0, len(LORA_RANKS))]
            tm_key = target_keys[idx % len(target_keys)]
            cfg = AdapterConfig(
                adapter_type="lora",
                lora_rank=int(rank),
                lora_alpha=int(rank) * 2,
                target_modules_key=tm_key,
                layer_placement=placement,
                unfreeze=unfreeze,
                head_type=head_type,
                pooling=pooling,
            )
        else:
            dim = BOTTLENECK_DIMS[rng.integers(0, len(BOTTLENECK_DIMS))]
            cfg = AdapterConfig(
                adapter_type="bottleneck",
                bottleneck_dim=int(dim),
                layer_placement=placement,
                unfreeze=unfreeze,
                head_type=head_type,
                pooling=pooling,
            )

        cfg.config_id = _make_id(cfg)

        if cfg.config_id not in seen_ids:
            seen_ids.add(cfg.config_id)
            configs.append(cfg)

    return configs


def collect_calibration_data(
    configs: List[AdapterConfig],
    dataset: dict,
    compute_stats_fn: Callable,
    finetune_fn: Callable,
    n_epochs: int = 15,
) -> CalibrationData:
    """Collect calibration data: statistics + downstream performance for each config.

    Args:
        configs: Adapter configs to evaluate.
        dataset: Dataset dict with 'samples', optional 'labels', 'name', 'task'.
        compute_stats_fn: (config) -> Dict[str, float] — computes statistics.
        finetune_fn: (config) -> Dict[str, float] — finetunes and returns metrics.
        n_epochs: Number of fine-tuning epochs.

    Returns:
        CalibrationData with all results.
    """
    is_classification = dataset.get("task") == "classification"
    metric_key = "accuracy" if is_classification else "mse"
    higher_is_better = is_classification  # higher accuracy = better, lower mse = better

    statistics = []
    performances = []

    for cfg in configs:
        stats = compute_stats_fn(cfg)
        statistics.append(stats)

        ft_result = finetune_fn(cfg)
        performances.append(ft_result[metric_key])

    return CalibrationData(
        configs=configs,
        statistics=statistics,
        performances=performances,
        dataset_name=dataset.get("name", "unknown"),
        metric_key=metric_key,
        higher_is_better=higher_is_better,
    )


def save_calibration_data(cal: CalibrationData, path: str):
    """Save calibration data to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cal.to_dict(), f, indent=2)


def load_calibration_data(path: str) -> CalibrationData:
    """Load calibration data from JSON."""
    with open(path) as f:
        return CalibrationData.from_dict(json.load(f))


# --- GP search ---

@dataclass
class ProxyIndividual:
    """Individual in the GP population."""
    tree: Node
    fitness: float = 0.0


@dataclass
class ProxySearchResult:
    """Result of a GP proxy search run."""
    best_tree: Node
    best_fitness: float
    best_formula: str
    generation_log: List[dict]
    final_population: List[dict]

    def to_dict(self) -> dict:
        return {
            "best_tree": serialize_tree(self.best_tree),
            "best_fitness": self.best_fitness,
            "best_formula": self.best_formula,
            "generation_log": self.generation_log,
            "final_population": self.final_population,
        }


def evaluate_proxy_fitness(
    tree: Node,
    calibration: CalibrationData,
    parsimony_coeff: float = 0.001,
) -> float:
    """Evaluate a proxy tree against calibration data.

    Returns Kendall tau (sign-corrected) with parsimony pressure.
    For MSE (lower=better), we negate proxy scores before computing tau
    so that a good proxy has positive tau.
    """
    proxy_scores = []
    for stats in calibration.statistics:
        score = evaluate_proxy(tree, stats)
        proxy_scores.append(score)

    # Check for degenerate proxy (all same score)
    if len(set(proxy_scores)) <= 1:
        return -1.0

    perfs = calibration.performances

    # If lower is better (MSE), negate proxy scores so positive tau = good
    if not calibration.higher_is_better:
        proxy_scores = [-s for s in proxy_scores]

    tau, _ = kendalltau(proxy_scores, perfs)
    if not np.isfinite(tau):
        return -1.0

    # Parsimony pressure
    fitness = tau - parsimony_coeff * tree.size()
    return fitness


def evaluate_proxy_fitness_multi(
    tree: Node,
    calibrations: List[CalibrationData],
    parsimony_coeff: float = 0.001,
) -> float:
    """Mean fitness across multiple datasets."""
    taus = []
    for cal in calibrations:
        tau = evaluate_proxy_fitness(tree, cal, parsimony_coeff=0.0)
        taus.append(tau)

    mean_tau = float(np.mean(taus))
    fitness = mean_tau - parsimony_coeff * tree.size()
    return fitness


def run_proxy_search(
    calibration: CalibrationData,
    multi_calibrations: Optional[List[CalibrationData]] = None,
    pop_size: int = 100,
    n_generations: int = 50,
    max_depth: int = 5,
    elite_count: int = 3,
    tournament_k: int = 5,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    parsimony_coeff: float = 0.001,
    seed: int = 42,
) -> ProxySearchResult:
    """Run GP search to discover a zero-cost proxy formula.

    Args:
        calibration: Primary calibration dataset.
        multi_calibrations: Optional list for cross-dataset fitness.
        pop_size: Population size.
        n_generations: Number of GP generations.
        max_depth: Maximum tree depth.
        elite_count: Elites passed unchanged.
        tournament_k: Tournament selection size.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation after crossover.
        parsimony_coeff: Penalty per tree node.
        seed: Random seed.

    Returns:
        ProxySearchResult with best proxy and search log.
    """
    rng = np.random.default_rng(seed)

    use_multi = multi_calibrations is not None and len(multi_calibrations) > 0
    eval_cals = multi_calibrations if use_multi else [calibration]

    def _fitness(tree: Node) -> float:
        if use_multi:
            return evaluate_proxy_fitness_multi(tree, eval_cals, parsimony_coeff)
        return evaluate_proxy_fitness(tree, calibration, parsimony_coeff)

    # Initialize population with ramped half-and-half
    population = []
    for i in range(pop_size):
        depth = 1 + (i % max_depth)
        method = "grow" if i % 2 == 0 else "full"
        tree = random_tree(depth, rng, method)
        fitness = _fitness(tree)
        population.append(ProxyIndividual(tree=tree, fitness=fitness))

    generation_log = []

    for gen in range(n_generations):
        # Sort by fitness descending
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        best = population[0]
        fitnesses = [ind.fitness for ind in population]
        gen_stats = {
            "gen": gen,
            "best_fitness": best.fitness,
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "best_formula": best.tree.to_str(),
            "best_size": best.tree.size(),
            "best_depth": best.tree.depth(),
        }
        generation_log.append(gen_stats)

        print(
            f"GP Gen {gen:3d}: best_tau={best.fitness:.4f} "
            f"mean={gen_stats['mean_fitness']:.4f} "
            f"size={best.tree.size()} "
            f"[{best.tree.to_str()[:60]}]"
        )

        if gen == n_generations - 1:
            break

        # Build next generation
        next_gen = []

        # Elitism
        for i in range(elite_count):
            elite = ProxyIndividual(
                tree=population[i].tree.copy(),
                fitness=population[i].fitness,
            )
            next_gen.append(elite)

        # Fill remaining
        while len(next_gen) < pop_size:
            if rng.random() < crossover_rate:
                # Tournament selection
                p1 = _tournament_select(population, tournament_k, rng)
                p2 = _tournament_select(population, tournament_k, rng)
                child_tree = subtree_crossover(p1.tree, p2.tree, rng, max_depth)
                if rng.random() < mutation_rate:
                    child_tree = point_mutation(child_tree, rng, max_depth)
            else:
                parent = _tournament_select(population, tournament_k, rng)
                child_tree = point_mutation(parent.tree, rng, max_depth)

            # Enforce depth limit
            if child_tree.depth() > max_depth:
                child_tree = random_tree(max_depth, rng, "grow")

            fitness = _fitness(child_tree)
            next_gen.append(ProxyIndividual(tree=child_tree, fitness=fitness))

        population = next_gen

    # Final sort
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    best = population[0]

    final_pop = [
        {
            "formula": ind.tree.to_str(),
            "fitness": ind.fitness,
            "tree": serialize_tree(ind.tree),
            "size": ind.tree.size(),
        }
        for ind in population[:20]  # Top 20
    ]

    return ProxySearchResult(
        best_tree=best.tree,
        best_fitness=best.fitness,
        best_formula=best.tree.to_str(),
        generation_log=generation_log,
        final_population=final_pop,
    )


def _tournament_select(
    population: List[ProxyIndividual],
    k: int,
    rng: np.random.Generator,
) -> ProxyIndividual:
    """Tournament selection."""
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    candidates = [population[i] for i in indices]
    return max(candidates, key=lambda ind: ind.fitness)


# --- Bridge to evolution.py ---

def make_proxy_evaluate_fn(
    best_tree: Node,
    dataset: dict,
    device: str = "cpu",
    batch_size: int = 64,
) -> Callable:
    """Create an evaluate_fn compatible with run_evolution().

    The returned function computes statistics for each config, then evaluates
    the discovered proxy formula to produce a 'composite' score.
    """
    from feasibility.statistics import compute_all_statistics

    samples = dataset["samples"]
    labels = dataset.get("labels")

    def evaluate_fn(configs: List[AdapterConfig]) -> List[dict]:
        results = []
        for cfg in configs:
            stats = compute_all_statistics(
                cfg, samples, labels=labels, device=device, batch_size=batch_size,
            )
            score = evaluate_proxy(best_tree, stats)
            results.append({
                "config": cfg.to_dict(),
                "scores": {"composite": score},
                "dataset": dataset.get("name", "unknown"),
            })
        return results

    return evaluate_fn
