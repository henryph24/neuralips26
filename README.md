# RR-MoA: Raw-Routed Mixture of Adapters

Reference implementation for the NeurIPS 2026 submission *Raw-Routed Mixture of Adapters: A Causal Intervention for Routing Collapse in Time Series Foundation Models*.

The paper diagnoses **normalization-induced routing collapse** in MoE adapters on Time Series Foundation Models (TSFMs), formalises the failure with a mutual-information decomposition and a tractable signal-ratio predictor R(D), and proposes three architecturally distinct fixes that all route on the raw, pre-normalisation input: **RR-MoA** (external raw router), **SR-MoA** (per-expert sigmoid gates, no router), and **Residual-IA⁺** (expert-level dual stream).

## Repository layout

```
feasibility/        Library: backbone loaders, fine-tuning loops, dataset helpers
scripts/            Per-experiment runners (Python) + orchestrators (shell)
evidence_vm/        Curated JSON evidence + verify.py (re-derives every numeric claim)
figures/            Tracked figure PDFs and inline TikZ source
data/               Dataset CSVs (gitignored; download instructions below)
results/            Per-run JSON outputs (gitignored)
main.tex            Paper source
```

## Installation

```bash
pip install -r requirements.txt
# For non-MOMENT backbones (Moirai / Moirai-MoE / Chronos):
pip install uni2ts chronos-forecasting
```

Tested on Python 3.10–3.12, CUDA 12.4, single A10G GPU (23 GB VRAM).

## Datasets

Six LTSF benchmarks are used in the main paper; expected layout under `data/`:

- `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` — auto-downloaded on first run from the public ETDataset GitHub release.
- `weather.csv`, `electricity.csv` — public LTSF benchmarks. Download from the official Time-Series-Library distribution and place under `data/`.

## Reproducing a single experiment

```bash
# RR-MoA Top-2, frozen backbone, ETTh1, seed 42 (one of the 54/54 main-table cells)
python scripts/run_rr_moa.py --dataset ETTh1 --horizon 96 --seed 42 \
    --top-k 2 --unfreeze frozen --router-input-mode raw

# AdaMix collapse demonstration (entropy → 0.000 under unfreezing)
python scripts/run_adamix.py --dataset ETTh1 --horizon 96 --seed 42 --unfreeze last2

# SR-MoA (router-free variant)
python scripts/run_self_routed_moa.py --dataset ETTh1 --horizon 96 --seed 42

# Residual-IA⁺ (DLinear-gap closing); pinned recipe used in the paper
python scripts/run_gap_closing.py --variant residual-ia --raw-arch nlinear \
    --raw-branch-shared --gate-init -2 --dataset ETTh1 --seed 42
```

Each runner writes one JSON to `results/<runner>/<dataset>_H<H>_K<K>_<freeze>_<seed>.json`.

## Verifying the paper's numeric claims

```bash
python evidence_vm/verify.py
```

Re-derives every quantitative claim in `main.tex` (Tables 3–5, baseline comparisons, cross-backbone percentages, dose-response, learnable-α, imputation, MI tightness) from the curated JSON evidence under `evidence_vm/`. Exits 0 only if all 107 checks match within tolerance (MSE 0.005, entropy 0.01, percentage 1.0pp).

Expected final line:

```
PASS: all 107 numeric claims match within tolerance (MSE 0.005, entropy 0.01, pct 1.0pp). RR-MoA wins: 27/27
```

## Seeds and statistics

- Core 5-seed grid: {42, 43, 44, 45, 46}
- Extended 10-seed robustness: {42, …, 51}
- Significance: Wilcoxon signed-rank, Bonferroni-corrected over 6 datasets × 5 seeds (`scripts/compute_significance.py`)

## Backbones

| Backbone | Identifier | Normalisation |
|---|---|---|
| MOMENT-small (primary) | `AutonLab/MOMENT-1-small` | RevIN |
| MOMENT-large | `AutonLab/MOMENT-1-large` | RevIN |
| Moirai-1.1-R-small | `Salesforce/moirai-1.1-R-small` | LayerNorm |
| Moirai-MoE-small | `Salesforce/moirai-moe-1.0-R-small` | LayerNorm |
| Chronos-T5-small | `amazon/chronos-t5-small` | T5 (no instance norm) |
| Timer-XL | `thuml/timer-base-84m` | LayerNorm (negative control) |

## License

MIT. See `LICENSE`.
