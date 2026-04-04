# Experiment Scripts

All scripts for reproducing the results in the paper. Requires a single A10G GPU.

## Core Methods

| Script | Paper Section | What it does |
|--------|-------------|-------------|
| `run_rr_moa.py` | Table 4, Section 3.4 | **RR-MoA** (17/18 wins, -56% MSE). Raw-routed mixture of adapters. |
| `run_standard_evolution.py` | Tables 2-3 | **AAS** per-dataset adapter search (25/25 wins). |
| `run_ablations.py` | Section 4.3 | Template generator + random code evolution baseline. |

## Ablations & Analysis

| Script | Paper Section | What it does |
|--------|-------------|-------------|
| `run_adamix.py` | Table 4 | AdaMix (hidden-state routing, 1/18 wins — proves collapse). |
| `run_darts_aas.py` | Section 5 | DARTS supernet (discretization gap evidence). |
| `run_budget_ablation.py` | Appendix C | Budget scaling (B=20 to 480). |
| `run_transferability.py` | Appendix D | 5x5 cross-dataset transfer matrix. |
| `run_zero_cost_proxy.py` | Appendix B | T-ZCP zero-cost proxy (grad norm, synflow). |
| `run_imputation.py` | Appendix G | Imputation task (proves task-dependence). |
| `run_trace_baseline.py` | — | TRACE-style adapter comparison. |
| `run_crossover_evolution.py` | — | Crossover vs random ablation. |
| `run_augmented_grammar.py` | — | G_tmpl vs G_tmpl+ comparison. |
| `run_sr_moa.py` | — | Spectral-routed MoA (weaker than RR-MoA). |
| `run_ensemble_aas.py` | — | Top-K adapter ensembling. |
| `run_adapter_selector.py` | — | Dataset feature extraction + cross-eval matrix. |
| `validate_spectral_hypothesis.py` | — | Spectral fingerprinting (negative result). |

## Supporting

| Script | Purpose |
|--------|---------|
| `run_local_evolution.py` | Local GPU runner for RACE VM (no Modal). |
| `run_code_evolution.py` | Original code evolution (Modal cloud GPU). |
| `collect_training_data.py` | Collects SFT dataset from evolution winners. |
| `finetune_qwen.py` | QLoRA fine-tuning of Qwen 2.5 3B. |

## Reproducing Key Results

### RR-MoA (Table 4)
```bash
# All 6 datasets x 3 seeds (~36 min)
for dataset in ETTh1 ETTm1 ETTh2 ETTm2 Weather Electricity; do
    for seed in 42 43 44; do
        python scripts/run_rr_moa.py --dataset $dataset --seed $seed
    done
done
```

### AAS Single-Adapter (Tables 2-3)
```bash
# Per-dataset evolution (~30 min each)
python scripts/run_standard_evolution.py --dataset ETTh1 --horizon 96
python scripts/run_standard_evolution.py --dataset ETTh1 --horizon 96 --seed 43
python scripts/run_standard_evolution.py --dataset ETTh1 --horizon 336
```

### Imputation Task (Appendix G)
```bash
python scripts/run_imputation.py --dataset ETTh1
```

## Legacy Scripts
Archived in `scripts/legacy/`. These were used during exploration but are not needed to reproduce paper results.
