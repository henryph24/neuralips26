# Project: Transferability-Guided Adapter Search for Time Series Foundation Models

NeurIPS 2026 submission — feasibility study exploring whether LLM-guided evolutionary search can discover effective adapter architectures for the MOMENT time series foundation model.

## Architecture

```
feasibility/          Core library (importable modules)
scripts/              Modal GPU runners (entrypoints)
template_code/        Vendored TEMPLATE transferability score code (CKA utils)
results/              Experiment outputs (JSON logs, comparisons) — gitignored
```

All GPU computation runs on **Modal.com** (A10G). LLM calls (OpenAI gpt-4o-mini) run locally. Data is serialized as numpy and sent to Modal containers.

## Key Modules

| Module | Purpose |
|--------|---------|
| `feasibility/config.py` | `AdapterConfig` dataclass, search space constants (ranks, dims, placements) |
| `feasibility/model.py` | `load_moment()`, LoRA/bottleneck attachment, encoder block discovery, hook registration |
| `feasibility/finetune.py` | `finetune_forecasting()` / `finetune_classification()` — train adapter+head on extracted features |
| `feasibility/features.py` | Batched feature extraction via forward hooks on first/last encoder blocks |
| `feasibility/scores.py` | TEMPLATE transferability scores (DL, PL, TA, composite) |
| `feasibility/data.py` | Dataset loaders (ETTh1/h2/m1/m2, Weather, EthanolConcentration, etc.), serialize/deserialize for Modal |
| `feasibility/evolution.py` | Traditional evolutionary search (`Individual`, `EvolutionLogger`, crossover, mutation, tournament selection) |
| `feasibility/llm_operators.py` | LLM-guided evolution over discrete hyperparameters (OpenAI function calling) |
| `feasibility/code_evolution.py` | **Code-level evolution** — LLM generates PyTorch `nn.Module` adapter code (FunSearch-style) |
| `feasibility/statistics.py` | Extended feature statistics for GP proxy search |
| `feasibility/proxy_gp.py` | Genetic programming for proxy score formulas |
| `feasibility/proxy_search.py` | End-to-end proxy search pipeline (calibrate → GP → validate) |
| `feasibility/modal_app.py` | Modal `app` + `image` definition, remote GPU functions (`evaluate_config`, `finetune_config`, `compute_statistics_remote`) |

## Experiment Progression

1. **TEMPLATE sweep** — 45 adapter configs × 2 datasets, compute transferability scores → found TEMPLATE fitness doesn't predict downstream MSE
2. **Evolutionary search** — Traditional evo over 8-dim discrete config space using 3-epoch MSE as fitness
3. **LLM-guided evo** — LLM proposes discrete configs instead of random mutation → marginal improvement over traditional evo (search space too small)
4. **Proxy search** — GP-evolved proxy scores from feature statistics → τ=0.50 but fails at selection
5. **Code-level evolution** (current) — LLM generates actual PyTorch adapter code (FunSearch-style). Multi-seed results (42-44):

### Code Evolution Results

| Dataset | Code-Evo (mean±std) | Evo-3epoch (mean±std) | Random (mean) |
|---------|--------------------|-----------------------|---------------|
| ETTh1   | **0.5525 ± 0.016** | 0.6189 ± 0.106       | 0.8704        |
| ETTm1   | **0.3633 ± 0.006** | 0.3697                | 0.3871        |

Key: Code-Evo has 6.7× lower variance than Evo-3epoch on ETTh1. Winning architectures: Conv1d+BatchNorm (ETTh1, 228K params), attention-weighted pooling (ETTm1, 49K params)

## Running Experiments

All scripts use Modal. Pattern:
```bash
modal run scripts/run_code_evolution.py                                    # full run (both datasets)
modal run scripts/run_code_evolution.py --n-generations 2 --dataset etth1  # smoke test
modal run scripts/run_code_evolution.py --seed 43 --dataset both           # different seed
modal run scripts/run_llm_evolution.py --seed 42                           # discrete LLM evo
```

Cost per run: ~$5.50 (4 GPU-hours + $0.05 LLM). Runtime: ~55 min per seed (both datasets).

The `OPENAI_API_KEY` must be set in `.env` (loaded via dotenv in scripts). LLM calls happen locally; only GPU training runs on Modal.

## Code Evolution Interface

The adapter contract for code-level evolution:
```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        # LLM generates

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len=512, d_model=768) -> (batch, output_dim=96)"""
```

Constraints: max 500K params, only `torch`/`nn`/`F`/`math` imports. Backbone: MOMENT with last-4 layers unfrozen.

## Data Flow (Code Evolution)

```
LLM (gpt-4o-mini) → code strings
  → Local CPU validation (exec, shape check, param count ≤ 500K)
  → Valid codes → Modal GPU starmap (3-epoch train) → MSE
  → Invalid codes → fitness=-inf, error fed back to LLM
  → Elitism (top-2) + LLM offspring → next generation
  → After 12 gens: top-5 validated at 15 epochs
```

## Search Space Comparison

| Dimension | Discrete Evo | Code Evo |
|-----------|-------------|----------|
| Space size | 2,160 configs (6 ranks × 2 targets × 3 placements × 4 unfreeze × 3 heads × 4 pooling) | Infinite (all valid PyTorch programs) |
| What varies | Hyperparameters (rank, placement, head type, pooling) | Entire architecture (arbitrary nn.Module code) |
| Evals/run | ~240 | ~240 (179 valid, 25% rejection rate) |
| LLM value-add | Marginal (space too small for guidance to matter) | Strong (discovers architectures outside discrete menu) |

The code space includes architectures that don't decompose into pooling+head (e.g., Conv1d→Conv1d→MeanPool→Linear), which is why Code-Evo outperforms discrete search.

## Conventions

- `evaluate_fn` in evolution modules takes **batch** input (list of configs or code strings) and returns list of result dicts — enables parallel Modal `starmap`
- Fitness is always "higher is better" (composite score, or -MSE for forecasting)
- All results saved to `results/<experiment_name>/` as JSON
- Cached baselines loaded from `results/early_stopping/comparison_*.json` for cross-experiment comparison
- Seeds: primary seed=42, multi-seed runs use 42-44

## Dependencies

Core: `torch>=2.1.0`, `momentfm`, `peft>=0.7.0`, `numpy`, `scipy`, `scikit-learn`, `pandas`
Classification: `aeon`
GPU infra: `modal`
LLM: `openai` (via `OPENAI_API_KEY` in `.env`)
Viz: `matplotlib`, `seaborn`
