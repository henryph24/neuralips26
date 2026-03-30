# Project: Code Evolution — LLM-Designed Adapter Architectures for Time Series Foundation Models

NeurIPS 2026 submission. LLM-guided evolutionary search generates PyTorch `nn.Module` adapter code for the MOMENT time series foundation model, searching an unbounded architecture space where LLM guidance genuinely outperforms discrete hyperparameter search.

## Architecture

```
feasibility/          Core library (importable modules)
scripts/              Modal GPU runners (entrypoints)
results/              Experiment outputs (JSON logs, comparisons)
template_code/        [LEGACY] Vendored TEMPLATE transferability code
```

GPU computation runs on **Modal.com** (A10G). LLM calls (OpenAI gpt-4o-mini) run locally. Data serialized as numpy for Modal containers.

## Active Modules

| Module | Purpose |
|--------|---------|
| `feasibility/code_evolution.py` | Core method — `CodeIndividual`, `validate_adapter_code()`, `train_adapter_from_code()`, `llm_generate_code()`, `run_code_evolution()` |
| `feasibility/modal_app.py` | Modal app/image, remote GPU functions: `evaluate_config`, `finetune_config`, `compute_statistics_remote` |
| `feasibility/model.py` | `load_moment()`, encoder block discovery, `_apply_unfreeze()`, LoRA/bottleneck attachment, hook registration |
| `feasibility/finetune.py` | `finetune_forecasting()` / `finetune_classification()`, head factories (`LinearHead`, `MLPHead`), pooling |
| `feasibility/data.py` | 7 forecasting loaders (ETTh1/h2/m1/m2, Weather, Electricity, Traffic), 2 classification, serialize/deserialize |
| `feasibility/config.py` | `AdapterConfig` dataclass, discrete search space constants (2,160 configs) |
| `scripts/run_code_evolution.py` | Main experiment script — CLI args: `--seed`, `--dataset`, `--n-generations`, `--pop-size`, `--model` |

## Baseline Modules (kept for comparison, do not extend)

| Module | Purpose |
|--------|---------|
| `feasibility/evolution.py` | Traditional evo search — `Individual`, crossover, mutation, tournament selection |
| `feasibility/llm_operators.py` | LLM-guided discrete hyperparameter evolution (OpenAI function calling) |
| `scripts/run_evolution.py` | Traditional evo runner |
| `scripts/run_llm_evolution.py` | Discrete LLM evo runner |

## Legacy Modules (abandoned directions, do NOT modify or extend)

- `feasibility/scores.py`, `feasibility/features.py`, `template_code/` — TEMPLATE transferability (scores don't predict MSE)
- `feasibility/statistics.py`, `feasibility/proxy_gp.py`, `feasibility/proxy_search.py` — GP proxy search (τ=0.50, fails at selection)
- `feasibility/viz.py` — old visualization
- `scripts/run_classification_*.py`, `scripts/run_proxy_search.py`, `scripts/analyze_calibration.py`, `scripts/test_early_stopping_proxy.py`, `scripts/finish_early_stop_validation.py` — abandoned experiments

## Code Evolution Interface

Adapter contract:
```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        # d_model=768 (MOMENT hidden dim), output_dim=96 (forecast horizon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len=512, d_model=768) -> (batch, output_dim=96)"""
```

Constraints: max 500K params, only `torch`/`nn`/`F`/`math` imports. Backbone: MOMENT with last-4 encoder layers unfrozen.

## Data Flow

```
LLM (gpt-4o-mini) → adapter code strings
  → Local CPU validation (exec, shape check, param count ≤ 500K)
  → Valid codes → Modal GPU starmap (3-epoch train) → MSE
  → Invalid codes → fitness=-inf, error fed back to LLM
  → Elitism (top-2) + 18 LLM offspring → next generation
  → After 12 gens: top-5 validated at 15 epochs
```

## Running Experiments

```bash
modal run scripts/run_code_evolution.py                                    # default: seed=42, dataset=both (ETTh1+ETTm1)
modal run scripts/run_code_evolution.py --dataset all                      # all 7 forecasting datasets
modal run scripts/run_code_evolution.py --seed 43 --dataset weather        # specific seed+dataset
modal run scripts/run_code_evolution.py --n-generations 2 --dataset etth1  # smoke test
```

Dataset options: `etth1`, `ettm1`, `etth2`, `ettm2`, `weather`, `electricity`, `traffic`, `both`, `all`

Cost: ~$3.30/run (~4 GPU-hours + $0.05 LLM). Runtime: ~55 min per seed per dataset.

`OPENAI_API_KEY` must be in `.env` (loaded via dotenv). LLM calls local; GPU training on Modal.

## Current Results (seeds 42-44)

| Dataset | Code-Evo (mean±std) | Evo-3ep (mean±std) | Random | Δ |
|---------|----|----|----|----|
| ETTh1 | **0.5614±0.011** | 0.6189±0.106 | 0.8704 | −9.3% |
| ETTh2 | **0.6580±0.010** | 0.7398±0.060 | 0.7951 | −11.1% |
| ETTm1 | **0.3619±0.007** | 0.3697 | 0.3871 | −2.1% |
| ETTm2 | 0.5501±0.009 | **0.5280±0.006** | 0.5810 | +4.2% |
| Weather | **0.3325±0.007** | 0.3376±0.007 | 0.3484 | −1.5% |
| Electricity* | **0.1956** | — | — | — |

*Single seed, no baselines yet. Weather missing seed 44. Traffic not yet run.

Winning architectures: Conv1d+BatchNorm (ETTh1, 228K params), attention-weighted pooling (ETTm1, 49K params), depthwise conv + learned positional weighting (Electricity, 52K params).

## Results Directory Convention

```
results/code_evolution/
  evo_log_{DATASET}_{SEED}.json      # per-generation evolution log with LLM reasoning
  validated_{DATASET}_{SEED}.json    # top-5 adapters evaluated at 15 epochs
  baselines_{DATASET}_{SEED}.json   # cached Evo-3epoch + Random results
  comparison_{DATASET}_{SEED}.json  # final comparison table
  run_{DATASET}_{SEED}.log          # stdout
```

## Conventions

- `evaluate_fn` takes **batch** input (list of code strings) → list of result dicts (enables Modal `starmap`)
- Fitness: always higher-is-better (−MSE for forecasting)
- Seeds: 42, 43, 44 for multi-seed statistical analysis
- Datasets use sliding windows (stride=64) of length 512, StandardScaler normalized

## Dependencies

Core: `torch>=2.1.0`, `momentfm`, `peft>=0.7.0`, `numpy`, `scipy`, `scikit-learn`, `pandas`
GPU infra: `modal`
LLM: `openai` (via `OPENAI_API_KEY` in `.env`)
Viz: `matplotlib`, `seaborn`
Classification (legacy): `aeon`
