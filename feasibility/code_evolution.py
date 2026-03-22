"""LLM-guided code evolution for adapter architecture discovery.

Instead of searching over discrete hyperparameters, the LLM generates actual
PyTorch nn.Module code for adapter architectures. The search space is unbounded —
the LLM can discover novel architectures that don't exist in the discrete space.
"""

import copy
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from feasibility.model import (
    load_moment,
    _get_encoder_blocks,
    _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch


# --- Data structures ---

@dataclass
class CodeIndividual:
    code: str
    fitness: float = 0.0       # -MSE (higher = better)
    mse: float = float('inf')
    param_count: int = 0
    error: str | None = None
    generation: int = 0
    reasoning: str = ""


# --- Seed adapters ---

SEED_ADAPTERS = [
    # 1. MeanPool + Linear (simplest baseline)
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        return self.linear(pooled)""",

    # 2. MeanPool + MLP2 (matches best LoRA head)
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        mid = d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid // 2, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled)""",

    # 3. LastToken + MLP
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        mid = d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last = hidden_states[:, -1, :]
        return self.net(last)""",

    # 4. Attention pooling + Linear (learnable weighted pooling)
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(d_model, 1)
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.attn_weights(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1)
        return self.linear(pooled)""",

    # 5. Conv1d downsample + Linear (temporal pattern extraction)
    """class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        ch = 128
        self.proj = nn.Linear(d_model, ch)
        self.conv = nn.Conv1d(ch, ch, kernel_size=8, stride=4)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(ch, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden_states)
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.mean(dim=2)
        return self.linear(x)""",
]

MAX_ADAPTER_PARAMS = 500_000
ALLOWED_MODULES = {"torch", "torch.nn", "torch.nn.functional", "math"}


# --- Validation ---

def validate_adapter_code(code: str, d_model: int = 768, output_dim: int = 96) -> dict:
    """Validate adapter code by exec + instantiate + dummy forward pass.

    Returns {"valid": bool, "error": str|None, "param_count": int}.
    """
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "math": __import__("math"),
    }

    try:
        exec(code, namespace)
    except Exception as e:
        return {"valid": False, "error": f"exec error: {e}", "param_count": 0}

    if "Adapter" not in namespace:
        return {"valid": False, "error": "No 'Adapter' class defined", "param_count": 0}

    AdapterClass = namespace["Adapter"]
    if not (isinstance(AdapterClass, type) and issubclass(AdapterClass, nn.Module)):
        return {"valid": False, "error": "Adapter is not an nn.Module subclass", "param_count": 0}

    try:
        adapter = AdapterClass(d_model, output_dim)
    except Exception as e:
        return {"valid": False, "error": f"__init__ error: {e}", "param_count": 0}

    param_count = sum(p.numel() for p in adapter.parameters())
    if param_count > MAX_ADAPTER_PARAMS:
        return {
            "valid": False,
            "error": f"Too many params: {param_count} > {MAX_ADAPTER_PARAMS}",
            "param_count": param_count,
        }

    try:
        dummy = torch.randn(2, 512, d_model)
        with torch.no_grad():
            out = adapter(dummy)
    except Exception as e:
        return {"valid": False, "error": f"forward error: {e}", "param_count": param_count}

    if out.shape != (2, output_dim):
        return {
            "valid": False,
            "error": f"Wrong output shape: {out.shape}, expected (2, {output_dim})",
            "param_count": param_count,
        }

    return {"valid": True, "error": None, "param_count": param_count}


# --- Training from code ---

def train_adapter_from_code(
    code: str,
    model: nn.Module,
    encoder_blocks: list,
    samples: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 3,
    lr: float = 1e-3,
    forecast_horizon: int = 96,
    batch_size: int = 64,
) -> dict:
    """Train an adapter defined by code string on the MOMENT backbone.

    Mirrors finetune_forecasting but takes code instead of AdapterConfig.
    Backbone last-4 layers unfrozen + adapter params trained.

    Returns {"mse": float, "param_count": int}.
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Build adapter from code
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "math": __import__("math"),
    }
    exec(code, namespace)
    AdapterClass = namespace["Adapter"]

    hidden_dim = _get_hidden_dim(model)
    adapter = AdapterClass(hidden_dim, forecast_horizon).to(device)
    param_count = sum(p.numel() for p in adapter.parameters())

    # Collect trainable params: adapter + unfrozen backbone
    trainable_params = list(adapter.parameters())
    param_ids = {id(p) for p in trainable_params}
    for param in model.parameters():
        if param.requires_grad and id(param) not in param_ids:
            trainable_params.append(param)
            param_ids.add(id(param))

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.MSELoss()

    # Prepare data
    seq_len = samples.shape[1]
    input_len = seq_len - forecast_horizon
    X = samples[:, :input_len]
    Y = samples[:, input_len:]

    X_padded = np.zeros_like(samples)
    X_padded[:, :input_len] = X

    n = len(X_padded)
    split = int(0.8 * n)
    X_train, X_val = X_padded[:split], X_padded[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()),
        batch_size=batch_size,
    )

    # Train
    for epoch in range(n_epochs):
        model.train()
        adapter.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            pred = adapter(feat)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    adapter.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2], device=device)

            feat = _extract_features_batch(model, encoder_blocks, batch_x, input_mask)
            pred = adapter(feat)
            val_losses.append(criterion(pred, batch_y).item())

    return {"mse": float(np.mean(val_losses)), "param_count": param_count}


# --- LLM code generation ---

CODE_SYSTEM_PROMPT = """You are an expert in neural architecture design for time series forecasting. You design PyTorch adapter modules that sit on top of a frozen transformer encoder (MOMENT).

The adapter replaces pooling + prediction head. It receives hidden states from the last encoder block and produces forecasts.

Interface contract:
```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        # Your architecture here

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        \"\"\"(batch, seq_len=512, d_model=768) -> (batch, output_dim=96)\"\"\"
        # Your architecture here
```

Constraints:
- Max 500K parameters
- Only use: torch, torch.nn (as nn), torch.nn.functional (as F), math
- Must handle variable batch sizes
- Output shape must be exactly (batch, output_dim)

Design principles for good adapters:
- The input has 512 timesteps × 768 features from a pretrained transformer
- You need to reduce 512×768 to just 96 forecast values
- Consider: attention pooling, multi-scale convolutions, gating mechanisms, learned positional weighting, feature selection, residual connections
- Simpler is often better — avoid overly complex architectures that overfit"""

CODE_USER_PROMPT_TEMPLATE = """# Current Population (Generation {generation})

## Top 5 adapters (best fitness = -MSE, higher is better):
{top_adapters}

## Bottom 3 adapters:
{bottom_adapters}

## Failed adapters (errors to avoid):
{failed_adapters}

## Fitness trajectory (best per generation): {fitness_history}

# Task

Generate {n_offspring} new adapter architectures. For each, provide:
1. Brief reasoning for the design choice
2. Complete Python code for the Adapter class

Return a JSON object with key "adapters", containing a list of objects each with "reasoning" (string) and "code" (string containing the full class definition).

Balance exploitation (variations of top adapters) with exploration (novel architecture ideas). Avoid repeating failed patterns."""


def llm_generate_code(
    population: List[CodeIndividual],
    n_offspring: int,
    generation: int,
    best_fitness_history: List[float],
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Use LLM to generate new adapter code.

    Returns list of {"reasoning": str, "code": str}. Empty list on failure.
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    # Top 5
    top_5 = []
    for ind in sorted_pop[:5]:
        if ind.fitness > float('-inf'):
            top_5.append({
                "fitness": round(ind.fitness, 6),
                "mse": round(ind.mse, 6) if ind.mse < float('inf') else "N/A",
                "param_count": ind.param_count,
                "code": ind.code,
            })

    # Bottom 3
    bottom_3 = []
    for ind in sorted_pop[-3:]:
        if ind.fitness > float('-inf'):
            bottom_3.append({
                "fitness": round(ind.fitness, 6),
                "mse": round(ind.mse, 6) if ind.mse < float('inf') else "N/A",
                "code": ind.code,
            })

    # Failed adapters
    failed = []
    for ind in sorted_pop:
        if ind.error:
            failed.append({
                "error": ind.error,
                "code_snippet": ind.code[:200] + "..." if len(ind.code) > 200 else ind.code,
            })
    failed = failed[:5]  # Cap at 5 failures

    history_str = [round(f, 6) for f in best_fitness_history]

    user_prompt = CODE_USER_PROMPT_TEMPLATE.format(
        generation=generation,
        top_adapters=json.dumps(top_5, indent=2),
        bottom_adapters=json.dumps(bottom_3, indent=2),
        failed_adapters=json.dumps(failed, indent=2) if failed else "None",
        fitness_history=history_str,
        n_offspring=n_offspring,
    )

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            max_tokens=8192,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)
        adapters = parsed.get("adapters", [])

        if not adapters:
            print(f"    LLM returned no adapters")
            return []

        print(f"    LLM proposed {len(adapters)} adapters (requested {n_offspring})")
        return [{"reasoning": a.get("reasoning", ""), "code": a.get("code", "")} for a in adapters]

    except Exception as e:
        print(f"    LLM call failed ({type(e).__name__}: {e})")
        traceback.print_exc()
        return []


# --- Evolution logger for code evolution ---

class CodeEvolutionLogger:
    """Track code evolution progress."""

    def __init__(self):
        self.generations = []

    def log_generation(self, gen: int, population: List[CodeIndividual]):
        valid = [ind for ind in population if ind.error is None]
        invalid = [ind for ind in population if ind.error is not None]

        valid_fitnesses = [ind.fitness for ind in valid] if valid else [0.0]
        best = min(valid, key=lambda ind: ind.mse) if valid else None

        stats = {
            "gen": gen,
            "n_valid": len(valid),
            "n_invalid": len(invalid),
            "best_mse": best.mse if best else float('inf'),
            "best_fitness": max(valid_fitnesses) if valid else float('-inf'),
            "mean_mse": float(np.mean([ind.mse for ind in valid])) if valid else float('inf'),
            "best_param_count": best.param_count if best else 0,
        }
        self.generations.append(stats)

        best_mse_str = f"{best.mse:.4f}" if best else "N/A"
        mean_mse_str = f"{stats['mean_mse']:.4f}" if valid else "N/A"
        print(
            f"Gen {gen:2d}: best_mse={best_mse_str} "
            f"mean_mse={mean_mse_str} "
            f"valid={len(valid)}/{len(population)} "
            f"params={stats['best_param_count']}"
        )

    def to_dict(self) -> dict:
        return {"generations": self.generations}


# --- Main evolution loop ---

def run_code_evolution(
    evaluate_fn: Callable[[List[str]], List[dict]],
    n_generations: int = 12,
    pop_size: int = 20,
    elite_count: int = 2,
    seed: int = 42,
    model: str = "gpt-4o-mini",
) -> dict:
    """Run code-level evolutionary search for adapter architectures.

    Args:
        evaluate_fn: Takes list of code strings, returns list of
                     {"mse": float, "param_count": int} or {"error": str}.
                     Batch interface for parallel GPU evaluation.
        n_generations: Number of generations.
        pop_size: Population size.
        elite_count: Number of elites to pass unchanged.
        seed: Random seed.
        model: LLM model for code generation.

    Returns:
        Dict with best_code, best_mse, logger, all_reasoning.
    """
    logger = CodeEvolutionLogger()
    best_fitness_history = []
    all_reasonings = []

    # Initialize population with seed adapters
    population: List[CodeIndividual] = []
    for code in SEED_ADAPTERS:
        population.append(CodeIndividual(code=code, generation=0))

    # Fill remaining slots with LLM cold-start
    if len(population) < pop_size:
        n_needed = pop_size - len(population)
        print(f"  Cold-start: requesting {n_needed} adapters from LLM...")
        cold_start = llm_generate_code(
            population, n_needed, generation=0,
            best_fitness_history=[], model=model,
        )
        for item in cold_start:
            if len(population) >= pop_size:
                break
            population.append(CodeIndividual(
                code=item["code"],
                reasoning=item["reasoning"],
                generation=0,
            ))

    # If still under pop_size, pad with seed variations
    while len(population) < pop_size:
        idx = len(population) % len(SEED_ADAPTERS)
        population.append(CodeIndividual(code=SEED_ADAPTERS[idx], generation=0))

    # Main loop
    for gen in range(n_generations):
        # Step 1: Validate locally (CPU)
        for ind in population:
            if ind.fitness == 0.0 and ind.mse == float('inf') and ind.error is None:
                val = validate_adapter_code(ind.code)
                if not val["valid"]:
                    ind.error = val["error"]
                    ind.fitness = float('-inf')
                    ind.param_count = val["param_count"]
                    continue
                ind.param_count = val["param_count"]

        # Step 2: Evaluate valid codes on GPU (batch for parallelism)
        to_eval = [
            ind for ind in population
            if ind.error is None and ind.mse == float('inf')
        ]
        if to_eval:
            codes = [ind.code for ind in to_eval]
            try:
                results = evaluate_fn(codes)
            except Exception as e:
                results = [{"error": f"batch evaluate error: {e}"}] * len(codes)

            for ind, result in zip(to_eval, results):
                if "error" in result:
                    ind.error = result["error"]
                    ind.fitness = float('-inf')
                else:
                    ind.mse = result["mse"]
                    ind.fitness = -result["mse"]  # Higher = better
                    ind.param_count = result.get("param_count", ind.param_count)

        # Step 3: Sort by fitness (descending)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        logger.log_generation(gen, population)

        valid_pop = [ind for ind in population if ind.error is None and ind.mse < float('inf')]
        if valid_pop:
            best_fitness_history.append(valid_pop[0].fitness)
        else:
            best_fitness_history.append(float('-inf'))

        if gen == n_generations - 1:
            break

        # Step 4: Elitism — top elites pass unchanged
        next_gen: List[CodeIndividual] = []
        seen_codes: set[str] = set()
        for ind in population[:elite_count]:
            if ind.error is None and ind.mse < float('inf'):
                elite = CodeIndividual(
                    code=ind.code,
                    fitness=ind.fitness,
                    mse=ind.mse,
                    param_count=ind.param_count,
                    generation=gen + 1,
                    reasoning=ind.reasoning,
                )
                next_gen.append(elite)
                seen_codes.add(ind.code.strip())

        # Step 5: LLM generates offspring
        n_needed = pop_size - len(next_gen)
        llm_results = llm_generate_code(
            population, n_needed, gen, best_fitness_history, model=model,
        )
        all_reasonings.append({
            "generation": gen,
            "n_proposed": len(llm_results),
            "reasonings": [r["reasoning"] for r in llm_results],
        })

        # Step 6: Dedup and add LLM codes
        for item in llm_results:
            if len(next_gen) >= pop_size:
                break
            code = item["code"].strip()
            if code not in seen_codes:
                seen_codes.add(code)
                next_gen.append(CodeIndividual(
                    code=item["code"],
                    reasoning=item["reasoning"],
                    generation=gen + 1,
                ))

        # Step 7: Fill remaining with seed adapter variations
        fill_idx = 0
        while len(next_gen) < pop_size:
            code = SEED_ADAPTERS[fill_idx % len(SEED_ADAPTERS)]
            if code.strip() not in seen_codes:
                seen_codes.add(code.strip())
                next_gen.append(CodeIndividual(code=code, generation=gen + 1))
            fill_idx += 1
            if fill_idx > len(SEED_ADAPTERS) * 2:
                # Avoid infinite loop — just duplicate
                next_gen.append(CodeIndividual(
                    code=SEED_ADAPTERS[fill_idx % len(SEED_ADAPTERS)],
                    generation=gen + 1,
                ))

        population = next_gen

    # Final result
    valid_pop = [ind for ind in population if ind.error is None and ind.mse < float('inf')]
    if valid_pop:
        best = min(valid_pop, key=lambda ind: ind.mse)
    else:
        best = population[0] if population else CodeIndividual(code="", mse=float('inf'))

    return {
        "best_code": best.code,
        "best_mse": best.mse,
        "best_fitness": best.fitness,
        "best_param_count": best.param_count,
        "best_reasoning": best.reasoning,
        "logger": logger,
        "all_reasonings": all_reasonings,
        "final_population": [
            {
                "code": ind.code,
                "mse": ind.mse,
                "fitness": ind.fitness,
                "param_count": ind.param_count,
                "error": ind.error,
                "reasoning": ind.reasoning,
            }
            for ind in sorted(population, key=lambda ind: ind.fitness, reverse=True)
        ],
    }
