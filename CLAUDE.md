# Project: Raw-Routed Mixture of Adapters (RR-MoA) for Time Series Foundation Models

NeurIPS 2026 submission. The paper diagnoses **normalization-induced routing collapse** in MoE adapters on Time Series Foundation Models (TSFMs), proves a mutual-information lower bound for the failure (Proposition 2), and proposes **Raw-Routed Mixture of Adapters (RR-MoA)** as the minimal causal fix. The companion experiment battery covers 6 LTSF datasets, 5 backbones, two tasks (forecasting + imputation), and a vision cross-modality control.

> **Note on history.** This project began as an LLM-guided code-evolution search for adapter architectures (the prior CLAUDE.md). That direction was abandoned in early April 2026 when the LLM-search results revealed a much more interesting phenomenon: standard MoE routing on TSFMs catastrophically collapses to a single expert. The current paper is about that collapse, its mechanism, and the fix. The LLM-search code (`feasibility/code_evolution.py`, `feasibility/evolution.py`, `feasibility/modal_app.py`) is **legacy** — kept for the appendix `app:aas_details` which references it as a small contribution but **NOT extended**.

---

## Architecture overview

```
feasibility/                Core library (importable PyTorch modules)
scripts/                    Experiment runners (Python) and orchestrators (shell)
results/                    Per-experiment JSON outputs (one file per dataset×config×seed)
evidence_vm/                Verification artifacts: verify.py + a curated subset of JSONs
figures/                    Paper figures (PDFs from PGFPlots/TikZ + 2 PNG-converted plots)
data/                       Cached LTSF dataset files (CSV → numpy)
main.tex                    Primary paper (~1900 lines, single file, neurips_2026.sty)
checklist.tex               NeurIPS reproducibility checklist (\input from main.tex)
template_code/              [LEGACY] vendored TEMPLATE transferability code (do NOT extend)
```

**Compute: RMIT RACE VM, NOT Modal.com.** The current workflow runs experiments on a single AWS A10G GPU (23 GB VRAM, CUDA 12.4, Ubuntu 22.04) accessed via SSH, with `tmux` sessions for long-running orchestrators and `rsync` for code/result transfers. Modal-based scripts (`scripts/run_ablations.py`, `scripts/run_code_evolution.py`) are legacy.

**RACE VM access** (see `~/.claude/projects/-Users-hungpq2412-neuralips26/memory/reference_race_ssh.md`):
```bash
ssh -i hungphanphd.pem ec2-user@ec2-13-238-161-176.ap-southeast-2.compute.amazonaws.com
```
SSH access requires the operator's public IP whitelisted in the RACE security group on port 22 via https://race.rmit.edu.au → Workspaces → Edit Security Group. The professor manages the security group.

---

## Active modules (`feasibility/`)

| Module | Purpose | Key entry points |
|---|---|---|
| `feasibility/model.py` | Backbone loading + encoder discovery + freeze control + LoRA/bottleneck hooks | `load_backbone(name)` (dispatches to MOMENT / Moirai / Moirai-MoE / Chronos), `_get_encoder_blocks(model)`, `_get_hidden_dim(model)`, `_apply_unfreeze(model, level)`, `attach_lora()`, `attach_bottleneck()` |
| `feasibility/finetune.py` | Forecasting + classification fine-tuning loops, feature extraction across backbones, head factories | `finetune_forecasting()`, `finetune_classification()`, `_extract_features_batch(model, blocks, batch_x, mask, backbone_type)`, `_forward_chronos()`, `_forward_moirai()`, `LinearHead`, `MLPHead` |
| `feasibility/data.py` | LTSF dataset loaders + multi-horizon support + serialize/deserialize for transport | `load_dataset_multihor(name, horizon, ...)`, `deserialize_dataset(...)` |
| `feasibility/config.py` | `AdapterConfig` dataclass and discrete search-space constants | `AdapterConfig` |
| `feasibility/rrmoa_macro_experts.py` | LLM-discovered "macro motifs" used as the AAS-pool ablation in RR-MoA (`MACRO_EXPERT_CLASSES`) | `MACRO_EXPERT_CLASSES`, `MACRO_EXPERT_NAMES` |
| `feasibility/code_evolution.py` | **Mostly legacy**, but other scripts import `SEED_ADAPTERS` and `validate_adapter_code()` from here as a stable canonical pool | `SEED_ADAPTERS` (5 hand-coded adapters used as fixed baselines), `validate_adapter_code()` |

**`load_backbone()` dispatch order matters** in `feasibility/model.py`: the substring `"moirai-moe"` MUST be matched BEFORE `"moirai"` (otherwise Moirai-MoE gets mis-routed to the regular Moirai loader). The current dispatch is: MOMENT → Chronos → Moirai-MoE → Moirai.

**MOMENT-small hidden states are `(B, 64, 512)`**, NOT `(B, 512, 768)` as some early documentation said: MOMENT pre-tokenizes 512 raw inputs into 64 patches of 512-dim, so the encoder output is 64×512 not 512×768. This matters for any code that index-slices hidden states.

### Legacy modules (do NOT extend or modify)

These are kept because they're imported by historical scripts or referenced in the AAS appendix, but the current paper does not depend on them being maintained:

| Module | Why legacy |
|---|---|
| `feasibility/evolution.py` | Discrete-config evolutionary search (pre-RR-MoA era) |
| `feasibility/llm_operators.py` | LLM-guided discrete hyperparameter mutation |
| `feasibility/modal_app.py` | Modal.com infrastructure — replaced by direct SSH + RACE VM |
| `feasibility/scores.py`, `feasibility/features.py` | TEMPLATE transferability (scores didn't predict MSE) |
| `feasibility/proxy_gp.py`, `feasibility/proxy_search.py` | GP proxy search (τ=0.50, fails at selection) |
| `feasibility/statistics.py` | Old statistics utilities replaced by `scripts/compute_significance.py` |
| `feasibility/viz.py` | Old visualization replaced by per-experiment `scripts/plot_*.py` |
| `template_code/` | Vendored TEMPLATE transferability — abandoned |

---

## Active experiment runners (`scripts/`)

The current paper's experiments live almost entirely in the scripts below. Each is a self-contained Python program with a CLI interface that takes `--dataset`, `--seed`, `--epochs`, etc.

### Core method + main baselines

| Script | Purpose | Result directory |
|---|---|---|
| `run_rr_moa.py` | Main RR-MoA experiment with `--router-input-mode {raw,revin,uniform}`, `--top-k`, `--unfreeze {frozen,last2,last4,all}`, `--expert-pool {canonical,macro}`, `--backbone` | `results/rr_moa/` |
| `run_adamix.py` | AdaMix baseline (hidden-state routing). **Now exposes a complete MoE-rescue CLI surface**: `--router-type {softmax,relu,expert-choice}`, `--load-balance-coef`, `--load-balance-variant {mean-prob,argmax}`, `--entropy-reg-coef`, `--z-loss-coef`, `--relu-l1-coef`, `--capacity-factor`. Used both as the main collapse demonstration AND as the rescue-baseline sweep host. | `results/adamix/` (default), `results/adamix_rescue/` (when any rescue flag is set) |
| `run_dlinear_baseline.py` | DLinear (Zeng et al. 2023) supervised calibration anchor | `results/dlinear/` |
| `run_lora_baseline.py` / `run_lora_sweep.py` | LoRA fine-tuning baseline; sweep covers 12 configs × 3 seeds × 3 datasets = 108 runs | `results/lora_baseline/` |
| `run_trace_baseline.py` | TRACE-style multi-scale adapter (Li and Zhu 2025) | `results/trace_baseline/` |
| `run_independent_ensemble.py` | 5 experts trained independently and averaged at inference (isolates routing from expert diversity) | `results/independent_ensemble/` |
| `run_full_finetune.py` / `run_extended_ft.py` | Full backbone fine-tuning (all 8 encoder blocks unfrozen, best-of 5 heads × 2 LRs); extended adds 50-epoch cosine schedules | `results/full_finetune/`, `results/extended_ft/` |
| `run_gap_closing.py` | Dual-stream / Multi-resolution / Raw-Input Expert / FiLM gap-closing variants for the DLinear gap | `results/gap_closing/` |
| `run_imputation.py` | Imputation task (20% masked reconstruction) | `results/imputation/` |
| `run_raw_mlp_moe.py` | **NEW (April 2026)**: Pure 5-MLP MoE on raw input with **NO TSFM at all** — Claim D defense ablation | `results/raw_mlp_moe/` |
| `run_freeze_ablation.py` | Orchestrator for freeze-level grid (cycles `frozen / last2 / last4`) | `results/freeze_ablation_summary_*.json` |

### Diagnostics + analysis (CPU, no GPU needed)

| Script | Purpose |
|---|---|
| `analyze_rescue_sweep.py` | **NEW** — aggregates `results/adamix_rescue/*.json` into a paper-ready rescue sub-table. Safe to run mid-sweep (marks partial configs with `*`). Supports `--latex` for direct LaTeX table emission. |
| `analyze_routing_signal_ratio.py` | Computes the routing signal ratio R(D) = [Var(M)+Var(Σ)] / Var(S) per dataset for Proposition 2 |
| `compute_significance.py` | Wilcoxon signed-rank + Bonferroni correction across 6 datasets × 5 seeds |
| `bootstrap_correlation.py` | Spearman ρ bootstrap CI for the R(D) ↔ Δ% correlation in Figure 2 |
| `knn_diagnostic.py`, `residual_diagnostic.py`, `run_residual_e1.py` | DLinear gap diagnostics: k-NN on backbone vs raw features, residual correction tests, neural E1 adapter |
| `run_fewshot_curve.py` | Few-shot learning curve N ∈ {10, ..., 5000} for the DLinear gap explanation |

### Visualization

| Script | Purpose |
|---|---|
| `plot_adamix_trajectory.py` | Generates `figures/adamix_trajectory.pdf` (per-step routing entropy + gradient norms) |
| `plot_routing_viz.py` | Generates `figures/routing_viz.pdf` (expert assignment vs amplitude/volatility scatter) |
| `plot_frozen_paradox.py` | Generates the Frozen Paradox bar chart (now embedded as inline TikZ) |

### Cross-modality control (vision)

| Script | Purpose |
|---|---|
| `run_vision_moe_collapse_v2.py` | **Active** — ResNet-18 + CIFAR-10 vision MoE control. 4 conditions (with/without normalization × frozen/unfrozen) testing whether the collapse mechanism is modality-general. Result feeds `tab:vision_moe`. |

### Orchestrator shell scripts (RACE VM workflow)

These all share the same pattern: `nohup` + `tee` + check-and-skip on existing JSONs + per-worker sharding for parallel execution within a tmux session.

| Script | Purpose | Sweep size |
|---|---|---:|
| `run_rescue_baseline_race.sh` | **NEW** — B1 sweep: 12 rescue configurations × 6 datasets × 2 freeze levels × 5 seeds | 720 runs |
| `run_moirai_moe_race.sh` | **NEW** — B2 sweep: Moirai-MoE backbone with RR-MoA + AdaMix | 60 runs (6×5×2) |
| `run_raw_mlp_moe_race.sh` | **NEW** — E1 sweep: pure raw-MLP MoE Claim D ablation | 18 runs (6×3) |
| `run_10seed_race.sh` | **NEW** — E2 sweep: 10-seed robustness extension on the core trio | 15 runs (3×5) |
| `run_gap_closing_race.sh` | Dual-stream / multi-resolution / raw-input-expert / FiLM × 6 datasets × 3 seeds | varies |
| `run_vision_moe_race.sh` | Vision MoE 4-condition control on CIFAR-10 | 12 runs (4×3) |
| `run_tier1_race.sh` … `run_tier5_race.sh` | Historical phased experiment batches; preserved but **not actively edited** | varies |

### Worker-sharding pattern

The newer orchestrators (`run_rescue_baseline_race.sh`, `run_raw_mlp_moe_race.sh`) accept a `worker K N` argument for parallel sharding within a single GPU:
```bash
tmux new-session -d -s e1_w1 'bash scripts/run_rescue_baseline_race.sh worker 1 4'
tmux new-session -d -s e1_w2 'bash scripts/run_rescue_baseline_race.sh worker 2 4'
# ... etc
```
Each worker iterates the same global config grid but only executes runs where `(global_run_index % N) == (K - 1)`. This gives deterministic non-overlapping work distribution. **Note**: the GPU is the bottleneck on a single A10G — adding more workers does not increase total throughput, it just distributes a fixed compute pie.

### Legacy scripts (do NOT extend)

The following scripts are kept for historical reference but should not be modified or referenced by new code:

- **Code-evolution era**: `run_code_evolution.py`, `run_local_evolution.py`, `run_crossover_evolution.py`, `run_evolution.py`, `run_llm_evolution.py`, `run_standard_evolution.py` (the latter is exception: still imported for `load_standard_data` data helpers)
- **AAS / proxy era**: `run_darts_aas.py`, `run_ensemble_aas.py`, `run_augmented_grammar.py`, `run_zero_cost_proxy.py`, `run_proxy_search.py`, `run_budget_ablation.py`, `run_adapter_selector.py`, `run_transferability.py`, `run_patchwise_analysis.py`
- **Spectral routing dead end**: `validate_spectral_hypothesis.py`, `synthetic_routing_collapse.py`
- **Misc**: `finetune_qwen.py` (unrelated experiment)
- **Modal-era**: `run_ablations.py` (only script that still imports `feasibility.modal_app`)
- **Phased orchestrators**: `run_phase1/2_race.sh`, `run_checkmate_race.sh`, `run_overnight_batch.sh`, `run_reviewer_response.sh`, `run_8plus_experiments.sh` — historical batches, not actively maintained

---

## Verification system: `evidence_vm/verify.py`

The paper's reproducibility hinges on `evidence_vm/verify.py` — a self-verification script that re-derives every quantitative claim from the raw JSON evidence and exits 0 only if every value matches within tolerance.

**Current state: 63 checks pass.** Categories:
- **TAB3** (RR-MoA freeze ablation main table): 9 cells, mean+std vs raw seeds
- **TAB4** (AdaMix collapse table): 9 MSE cells + 9 entropy cells
- **TAB2 no-RevIN green rows**: 3 MSE + 3 entropy
- **TAB5 top-k**: 4 sparsity values
- **TAB_BACKBONE_PCT** (cross-backbone arithmetic): 9 cells, recomputes `(fixed-rr_moa)/fixed*100` and compares to claimed percentage
- **TAB_HORIZON_GAP** (DLinear-gap arithmetic): 4 cells, recomputes `(rr_moa-dlinear)/dlinear*100`
- **TAB_BASELINES_LORA**: parses `main.tex` directly to extract the Best LoRA row and verify each cell against the bolded appendix sweep minima

**Helper functions** for percentage arithmetic:
- `check_improvement_pct(larger, smaller, claimed_pct)` — for `{-XX%}` improvement cells
- `check_gap_pct(higher, lower, claimed_pct)` — for `{+XX%}` gap-to-baseline cells
- `grep_main_tex_lora_row()` — direct main.tex regex parse for the LoRA row

**Tolerance**: `TOL = 0.005` for MSE values, `ENT_TOL = 0.01` for entropy, `PCT_TOL = 1.0` percentage point for arithmetic cells.

**When you add a new numeric claim to `main.tex`, you should ALSO add a corresponding check to `verify.py`.** This catches arithmetic errors at paper-build time. The current verify.py was extended in this April session after an LLM reviewer audit found 4 small but real arithmetic errors that the previous (narrower) verify.py would not have caught.

**Negative-test convention**: revert one numeric value in `main.tex`, run `verify.py`, confirm it exits 1 with a clear discrepancy report, then restore. This ensures the verification is actually exercised by the changed cell.

---

## Datasets, backbones, conventions

### Datasets (LTSF benchmarks)

6 forecasting datasets in the main paper, sliding-window length 512, **channel-wise StandardScaler-normalized** per the LTSF convention (`scripts/run_standard_evolution.py::load_standard_data` is the canonical loader and is imported by every active runner):

| Dataset | Train / Val / Test | Channels | Notes |
|---|---|---:|---|
| ETTh1 | 8640 / 2880 / 2880 | 7 | Hourly temperature |
| ETTh2 | 8640 / 2880 / 2880 | 7 | Hourly temperature, harder |
| ETTm1 | 34560 / 11520 / 11520 | 7 | 15-min temperature, **largest dataset** (4× ETTh1) |
| ETTm2 | 34560 / 11520 / 11520 | 7 | 15-min temperature, harder |
| Weather | 60/20/20 % split | 21 | Weather features |
| Electricity | 60/20/20 % split | 321 | Power consumption |

**Traffic** (7th LTSF dataset) is included in the boundary-case Figure 2 (R(D)=0.14, where RR-MoA correctly does NOT help) but is NOT in the main 6-dataset grid. Imputation tasks use ETTh1, ETTm1, Weather only.

### Backbones

5 backbones supported via `feasibility/model.py::load_backbone()`:

| Backbone | Identifier | d_model | Layers | Normalization |
|---|---|---:|---:|---|
| MOMENT-small | `AutonLab/MOMENT-1-small` | 512 | 8 | RevIN (internal, affine=False) — **primary** |
| MOMENT-large | `AutonLab/MOMENT-1-large` | 1024 | 24 | RevIN |
| Moirai-1.1-R-small | `Salesforce/moirai-1.1-R-small` | 384 | 6 | LayerNorm only (no RevIN) |
| Moirai-MoE-small | `Salesforce/moirai-moe-1.0-R-small` | 384 | 6 | LayerNorm only (sparsely-routed FFN experts) |
| Chronos-T5-small | `amazon/chronos-t5-small` | 512 | 6 | T5 encoder-decoder, no instance normalization (negative control) |

**MOMENT-small is the primary backbone** for almost all experiments. The other backbones appear in the cross-backbone analysis (`tab:cross_backbone`, `tab:moirai_moe`).

### Seeds and statistical conventions

- **Core 5 seeds**: {42, 43, 44, 45, 46} — used for the main RR-MoA grid and most baselines
- **Extended 10 seeds**: {42, ..., 51} — used in `run_10seed_race.sh` for the core trio (ETTh1, ETTm1, Weather) as a robustness check
- **Smaller ablations**: 3 seeds {42, 43, 44}
- **Significance**: Wilcoxon signed-rank, Bonferroni-corrected across 6 datasets × 5 seeds, computed by `scripts/compute_significance.py`

### MSE convention

**All MSE values in the paper are on channel-wise StandardScaler-normalized inputs** (the standard LTSF convention used by Informer, PatchTST, etc.). The paper reports both normalized and (in some appendix tables) denormalized variants. `compute_denorm_mse(preds, tgts, test_ch, scaler)` in `scripts/run_standard_evolution.py` converts back to original units.

### Result file naming convention

Each experiment produces one JSON per (dataset, config, seed) combination:

```
results/rr_moa/{DATASET}_H{HORIZON}_K{K}_top{k}_{freeze}_{seed}{suffix}.json
results/adamix/{DATASET}_H{HORIZON}_K{K}_{freeze}_{seed}{suffix}.json
results/adamix_rescue/{DATASET}_H{HORIZON}_K{K}_{freeze}_{seed}_rtr{router}_lb{lb}_lv{variant}_ent{ent}_z{z}_l1{l1}_cf{cf}.json
results/raw_mlp_moe/{DATASET}_H{HORIZON}_K{K}_top{k}_{seed}.json
```

The `{suffix}` field encodes optional ablation flags: `_no_revin`, `_batchnorm`, `_groupnorm`, `_bb-moirai`, `_bb-moirai-moe`, `_bb-chronos`, `_bb-moment-large`, `_pool-macro`, `_router-revin`, `_router-uniform`. **Filename uniqueness is the only collision protection** — orchestrators rely on `[ -f $OUT ] && continue` for resumability.

---

## RACE VM workflow

All major experiments run on a single AWS A10G GPU via SSH to the RMIT RACE VM. The standard workflow:

1. **Develop locally** (write/edit Python or shell scripts)
2. **Rsync to RACE**:
   ```bash
   rsync -az -e "ssh -i hungphanphd.pem" scripts/run_*.py scripts/run_*.sh \
       ec2-user@ec2-13-238-161-176.ap-southeast-2.compute.amazonaws.com:~/neuralips26/scripts/
   ```
3. **Smoke test on RACE** (1 run, short epoch count, verify no crash)
4. **Launch orchestrator inside `tmux`**:
   ```bash
   ssh -i hungphanphd.pem ec2-user@... "cd ~/neuralips26 && \
       tmux new-session -d -s b1_w1 'bash scripts/run_rescue_baseline_race.sh worker 1 4 2>&1 | tee results/b1_w1.log'"
   ```
5. **Periodically rsync results back**:
   ```bash
   rsync -az -e "ssh -i hungphanphd.pem" \
       ec2-user@...:~/neuralips26/results/{rr_moa,adamix_rescue,raw_mlp_moe}/ results/
   ```
6. **Run analysis locally** (`scripts/analyze_*.py`, `evidence_vm/verify.py`)
7. **Edit `main.tex`** with the new numbers
8. **Verify** (`python3 evidence_vm/verify.py` must exit 0)
9. **Commit + push to GitHub origin** (origin/main, NOT on the RACE VM)

### Per-run runtime estimates (single GPU, no contention)

| Backbone | Per-run wall clock (15 epochs, 1 dataset, 1 seed) |
|---|---:|
| MOMENT-small | ~45 sec (ETTh1) — ~3 min (ETTm1, larger data) |
| MOMENT-large | ~90 sec |
| Moirai-MoE-small | ~90 sec solo, ~7 min under 5-way contention |
| Chronos-T5-small | ~90 sec |
| Pure raw-MLP MoE (no backbone) | ~30 sec (no backbone forward pass) |

### GPU contention

With **N concurrent workers** sharing the single A10G, throughput per run scales roughly as `N × t_solo` (the GPU is fully serialized). **Total throughput is approximately constant** regardless of N; the only benefit of more workers is overlapping I/O / model loading across runs. In practice, 4 concurrent workers is the sweet spot — beyond that, OOM risk grows without throughput gain.

---

## Paper structure (`main.tex`)

Single ~1900-line file using `neurips_2026.sty`. The order is:
1. **Abstract** (one paragraph, ~200 words, includes the headline numbers)
2. **§1 Introduction** — frames the diagnosis, deployment motivation, contributions reordered theory-first (April 2026 reframe)
3. **§2 Related Work** — TSFM adaptation, MoE in foundation models with explicit MoE-rescue positioning, normalization in time series, NAS/LLM-guided search
4. **§3 Method** — RR-MoA architecture, expert pool, problem formulation
5. **§4 Experiments** — backbones, datasets, training, baselines, then `sec:main_results` covering AdaMix collapse + rescue sub-table + comprehensive baseline comparison + cross-backbone + dual-stream gap-closing + sparse routing/horizon/imputation. **Theory + Proposition 2 + signal-ratio plot are inlined here.**
6. **§5 Conclusion + Limitations**
7. **References** (`thebibliography` in-file)
8. **Appendix** with ~15 subsections covering proofs, AAS details, LoRA sweep, formal definitions, extended grids, vision control, top-k, imputation, deployment, benchmarks, extended FT, gap-closing, and the new pure raw-MLP MoE ablation

### Key tables (with labels)

| Label | Section | Purpose |
|---|---|---|
| `tab:rrmoa` | §3 | RR-MoA freeze-level ablation 3 datasets × 3 freeze × 3 seeds (54/54 wins) |
| `tab:adamix` | §3 | AdaMix collapse demonstration with green-row no-RevIN ablation |
| `tab:rescue` | §3 | **NEW** rescue-baseline sub-table — 12 rescue mechanisms, currently 5 rows complete |
| `tab:baselines` | §3 | Comprehensive baseline comparison (LoRA, TRACE, Ind. Ensemble, AdaMix, Full FT, RR-MoA, DLinear) |
| `tab:gap_closing_headline` | §3 | Dual-stream beats DLinear on Weather (main-text headline) |
| `tab:router_input` | §3 | Rawness vs bypass ablation (RevIN router degrades MSE 60-88%) |
| `tab:horizon` | App | Multi-horizon RR-MoA vs DLinear |
| `tab:cross_backbone` | App | Full cross-backbone grid (MOMENT-small/large, Moirai) |
| `tab:moirai_moe` | App | **NEW** Moirai-MoE extension table (currently ETTh1 + ETTh2 done, 4 datasets running) |
| `tab:lora_sweep` | App | Full 108-run LoRA sweep |
| `tab:vision_moe` | App | Vision cross-modality control |
| `tab:gap_closing` / `tab:gap_closing_ext` | App | Full 6-dataset gap-closing variants |
| `tab:raw_mlp_moe` | App | **NEW** Pure raw-MLP MoE ablation (Claim D defense) |

### Figures
- `fig:framework` — RR-MoA architecture (TikZ inline)
- `fig:trajectory` — adamix_trajectory.pdf (per-step routing entropy + gradient norms, 2 panels)
- `fig:routing_viz` — routing_viz.pdf (expert assignment scatter, **promoted to main paper** in this session)
- `fig:frozen_paradox` / `fig:signal_ratio` — combined inline TikZ+PGFPlots (Frozen Paradox bar + R(D) scatter)

---

## Auto-memory pointers

The session state for this paper is mirrored in `~/.claude/projects/-Users-hungpq2412-neuralips26/memory/`. Key files:
- `MEMORY.md` — index
- `project_neurips26_idea.md` — paper direction
- `project_acceptance_assessment.md` — review-distribution estimates and target trajectory
- `project_neurips_push_progress.md` — current B1-B9 experiment queue + writing-task status
- `project_dlinear_gap_analysis.md` — settled DLinear-gap diagnosis
- `reference_race_ssh.md` — RACE VM access details (key, host, IP whitelist procedure)
- `reference_moe_rescue_formulas.md` — verified Switch / ST-MoE / ReMoE / Expert-Choice formulas (extracted directly from source PDFs, NOT folklore)
- `feedback_race_workflow.md` — user preferences: drive SSH directly, verify papers before coding, no scope trimming at 2-week horizon

---

## Conventions and gotchas

1. **Standard imports for new experiment scripts**:
   ```python
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   from feasibility.model import load_backbone, _get_encoder_blocks, _get_hidden_dim, _disable_gradient_checkpointing
   from feasibility.finetune import _extract_features_batch
   from scripts.run_standard_evolution import load_standard_data, _detect_backbone_type, compute_denorm_mse
   ```

2. **Always disable gradient checkpointing** before training: `_disable_gradient_checkpointing(model)`. Otherwise hooks fire inconsistently in train mode and feature extraction silently breaks.

3. **AMP convention**: bfloat16 autocast on `cuda`, MSE loss in float32. Always wrap forward + loss in `with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):`.

4. **Backbone detection**: every experiment script calls `_detect_backbone_type(args.backbone)` which returns `"moment"`, `"moirai"`, or `"chronos"`. Moirai-MoE shares `"moirai"` since the feature-extraction path is identical (both use `model.in_proj`, `model.patch_sizes`, `model.encoder.layers`). The dispatch in `_extract_features_batch()` is by type, so adding a new backbone family means adding a new branch.

5. **Don't commit `results/` directories or `.DS_Store`**. The `.gitignore` covers most of this but be careful when staging — `git add .` will pick up the macOS junk files. Prefer explicit `git add main.tex scripts/run_*.py`.

6. **The `evidence_vm/` directory contains a curated subset of JSONs** (not all results) used by `verify.py`. When you add a new experiment that should be verifiable, also copy a representative subset to `evidence_vm/` and extend `verify.py` with the corresponding check block.

7. **Routing entropy convention**: Shannon entropy in nats, max value `log K = log(5) ≈ 1.609`. **Any reported entropy value > 1.609 is impossible** and indicates a logging bug. (An LLM reviewer once hallucinated an "entropy = 3.000" value that didn't exist; verify.py should also catch this category of bug.)

8. **NeurIPS phrasing discipline**: avoid absolute claims that the data does not support. Specifically:
   - Don't say "all rescue mechanisms fail" — say "all 11 rescue mechanisms in our sweep fail to recover MSE within X% of RR-MoA"
   - Don't say "frozen always beats unfrozen" — say "frozen RR-MoA is the best on 4/6 datasets; unfreezing wins by up to 13% on the remaining 2"
   - Don't say "RR-MoA wins everywhere" — count and report the wins explicitly (e.g., "54/54 wins" or "1/9 on Chronos")

   The paper currently passes this discipline; new edits should preserve it.

9. **Commit messages should describe changes, not the session**: don't write commit messages like "this session's work" — describe each modification. Multi-paragraph commit messages with `feat:` / `fix:` prefixes and explicit numeric deltas are the local convention (see `git log --oneline` for examples).

10. **Never include Claude Code attribution footers in commit messages**. The user explicitly forbids this in their global CLAUDE.md.

---

## Dependencies (`requirements.txt`)

```
torch>=2.1.0
momentfm
peft>=0.7.0
numpy
scipy
scikit-learn
pandas
aeon            # legacy classification (UEA datasets)
matplotlib
seaborn
modal           # legacy — only run_ablations.py and run_code_evolution.py use it
openai          # legacy — only the AAS appendix path uses LLM calls
anthropic       # legacy — used for Claude API calls in earlier code-evo experiments
```

For Moirai / Moirai-MoE / Chronos: install `uni2ts` (provides `uni2ts.model.moirai`, `uni2ts.model.moirai_moe`) and `chronos-forecasting` (provides `chronos.ChronosPipeline`). These are not in `requirements.txt` because they are RACE-VM-only.

---

## What's actively in flight (snapshot — may be stale)

As of the last RACE VM contact at 00:44 AEST on 2026-04-11:

| Experiment | Progress | Status |
|---|---|---|
| B1 rescue sweep (12 configs × 60 cells = 720 runs) | 337/720 (47%) | ✅ 5 configs fully complete; 7 still running on RACE |
| B2 Moirai-MoE (60 runs) | 21/60 (35%) | ✅ ETTh1+ETTh2 done with publication-quality 5-seed numbers |
| E1 raw-MLP MoE (18 runs) | **18/18 ✅** | DONE — full appendix subsection in main.tex |
| E2 10-seed extension (15 runs) | 9/15 (60%) | ✅ ETTh1 5/5 done, ETTm1 5/5 nearly done |

All are queued to resume / complete on the RACE VM. Their tmux sessions persist across SSH outages, so the orchestrators continue accumulating result JSONs even when local SSH access is interrupted.
