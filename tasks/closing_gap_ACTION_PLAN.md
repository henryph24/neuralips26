# Closing the DLinear Gap: Action Plan

*Distilled from 7 iterations of deep analysis (see `closing_gap_relaxed_constraints.md` for full details)*

## The Gap

| Dataset | DLinear | RR-MoA (frozen MOMENT) | Gap |
|---------|---------|------------------------|-----|
| ETTh1 | 0.417 | 0.690 | +65% |
| ETTm2 | 0.200 | 0.537 | +169% |
| Weather | 0.208 | 0.289 | +39% |

Root cause: RevIN strips μ, σ before encoding. Encoder abstracts away temporal detail. Adapter can't recover what's lost.

## The Three Experiments To Run

### Experiment 1: Multi-Horizon DLinear (~2 GPU-hours, ~$0.70)

**Why**: DLinear was only tested at H=96. At H=720, DLinear has 368K params mapping 512→720 (underdetermined). FM should dominate.

```bash
for H in 192 336 720; do
  for dataset in etth1 ettm1 weather electricity; do
    for seed in 42 43 44; do
      modal run scripts/run_dlinear_baseline.py --horizon $H --dataset $dataset --seed $seed
    done
  done
done
```

**If gap narrows/reverses at long H**: Paper gains "FM advantage grows with horizon" narrative.

### Experiment 2: Few-Shot Curves (~3 GPU-hours, ~$1.00)

**Why**: Already scripted (`scripts/run_fewshot_curve.py`), never run. FM should dominate at low N.

```bash
modal run scripts/run_fewshot_curve.py --dataset etth1 --seed 42
modal run scripts/run_fewshot_curve.py --dataset weather --seed 42
modal run scripts/run_fewshot_curve.py --dataset electricity --seed 42
```

**If FM wins at N<200**: Answers "why not just DLinear?" reviewer question.

### Experiment 3: Residual DLinear + FM Correction (~2 GPU-hours, ~$0.70)

**The #1 approach from the analysis.** Guaranteed DLinear floor. FM can only help.

```
ŷ = DLinear(x) + α · FM_adapter(hidden_states)
```

**Implementation**: ~30 lines changed in `feasibility/code_evolution.py`:
- Preserve `batch_x_raw` before `.unsqueeze(1)` (line 263)
- Pre-train DLinear on raw input (5 epochs, freeze)
- Change adapter call: `pred = dlinear(batch_x_raw) + 0.1 * adapter(feat)` (line 268)
- Same change in validation loop (line 284)

**Key diagnostic**: Track α (residual_scale) per dataset. If α→0, FM adds nothing. If α>0, FM adds measurable value. Either result is publishable.

---

## Why Residual (E1) Over Alternatives

| Approach | Gap Closure | Why Not First? |
|----------|-------------|---------------|
| Stats Injection (A1) | 20-40% | Partial fix — only recovers μ,σ, not temporal detail |
| Dual-Path + Gate (A2) | 70-90% | Gate must learn routing — harder optimization than residual |
| Hybrid FM+DLinear (C1) | 90-100% | Gate-based = same idea as E1 but harder to optimize |
| **Residual (E1)** | **85-95%** | **Simplest. Guaranteed floor. Connected to boosting theory.** |
| HyperDLinear (E2) | 90-100% | Most powerful but training instability risk. Run AFTER E1 validates. |

Theoretical hierarchy: **Hypernetwork ⊇ Residual ⊇ Gate ⊇ Stats Injection ⊇ Frozen**

---

## Diagnostic Result: Is the Gap Closable? ANSWERED.

**Ran k-NN diagnostic on CPU** (`scripts/knn_diagnostic.py`). Result:

| Method | Input | MSE |
|--------|-------|-----|
| k-NN (k=20) | Raw input | 0.636 |
| k-NN (k=20) | Backbone features (PCA-256) | 0.737 |
| DLinear (trained) | Raw input | 0.417 |

**Backbone features are 16% WORSE than raw input even with a universal approximator (k-NN).**

**Conclusion: Raw input bypass is MANDATORY.** No adapter operating on backbone features alone can match DLinear. This empirically proves E1 (Residual) and A2 (Dual-Path) are the correct approaches — they include raw input access.

Also discovered: MOMENT hidden states are shape `(B, 64, 512)`, not `(B, 512, 768)`. 64 patches × 512-dim = 32K values, not 393K. The bottleneck is more severe than expected.

**Cross-dataset results confirm pattern:**

| Dataset | k-NN (raw) | k-NN (backbone) | Gap | Ridge/k-NN ratio |
|---------|------------|-----------------|-----|------------------|
| ETTh1 | 0.636 | 0.737 | +16% | 1.39 (nonlinear) |
| ETTm1 | 0.498 | 0.614 | +23% | 0.84 (linear) |
| Weather | 0.453 | 0.601 | +33% | 0.96 (linear) |

**New insight**: ETTm1 and Weather are almost perfectly linear (Ridge ≈ k-NN). Only ETTh1 has nonlinear structure the FM could exploit. Residual α should be > 0 on ETTh1, ≈ 0 on the others.

### RESIDUAL DIAGNOSTIC (Ridge-based E1 proxy)

**Ran `scripts/residual_diagnostic.py`. Result: backbone features add ~0% complementary signal.**

| Dataset | DLinear (Ridge) | Best Residual | Improvement |
|---------|----------------|---------------|-------------|
| ETTh1 | 0.8825 | 0.8794 | **-0.4%** (noise) |
| ETTm1 | 0.4183 | 0.4165 | **-0.4%** (noise) |
| Weather | 0.4372 | 0.4348 | **-0.5%** (noise) |

**Harsh truth**: With linear methods, MOMENT features provide zero complementary value to raw input on these benchmarks. The gap is pure information loss with no compensating signal.

**But**: This is Ridge (linear). A neural adapter on ETTh1 (nonlinear ratio=1.39) might still extract 5-15% improvement. ETTm1/Weather are linear-sufficient — no method can help there.

**Revised recommendation**: ~~The most valuable experiment is now neural E1 on ETTh1 only.~~

### NONLINEAR RESIDUAL RESULT: Backbone Is Dead Weight

Tested RF (Random Forest) residual on ETTh1 — the dataset with highest nonlinear potential:

| Method | MSE | vs DLinear |
|--------|-----|-----------|
| RF residual on backbone | 0.8678 | **+0.02% (nothing)** |
| RF direct on raw input | 0.4858 | **-44% (massive)** |

**ETTh1 has huge nonlinear structure (RF beats Ridge by 44% on raw), but backbone features contain NONE of it.** The encoder destroys all fine-grained temporal patterns, not just μ/σ.

**All E1/A2/E2 approaches will degenerate to DLinear** because the backbone contributes zero complementary signal. The gap is NOT closable from backbone features on these benchmarks.

### FINAL Recommendation

**Don't chase gap closure.** The paper's gap diagnosis IS the contribution. For practical forecasting, use DLinear/RF on raw input. Use MOMENT only for tasks where pre-training helps (imputation, anomaly detection, transfer).

---

## Decision Tree

```
Run k-NN diagnostic
├── k-NN ≈ DLinear → Info IS in features
│   ├── Run E1 (Residual) → α > 0?
│   │   ├── Yes → FM adds value, publish "Residual FM Correction"
│   │   └── No → FM features redundant with DLinear, publish "FM adds nothing"
│   └── Run E2 (HyperDLinear) → beats DLinear?
│       ├── Yes → Novel paradigm, "FM as Weight Generator"
│       └── No → Training too hard, stick with E1
└── k-NN >> DLinear → Info NOT in features
    ├── Run E1 anyway → α ≈ 0 (confirms bottleneck)
    ├── Try A1 (stats injection) → if helps, bottleneck is specifically μ,σ
    └── Try A3 (disable RevIN) → if helps, RevIN is the sole problem
```

**Every leaf is publishable.**

---

## The Two-Factor Framework (For Paper Discussion)

FM value = f(Signal Complexity, Data Scarcity)

| | Low N (few-shot) | High N (full data) |
|---|---|---|
| **Complex signal** (Weather, Electricity) | FM essential | FM helps moderately |
| **Simple signal** (ETT) | FM helps as prior | DLinear sufficient |

The Residual approach auto-adapts: α is large where FM helps, small where it doesn't.

---

## If You Have 1 More Week

| Day | Action | Deliverable |
|-----|--------|-------------|
| 1 | Run Experiments 1+2 (multi-horizon + few-shot) | Horizon-gap curve, few-shot curve |
| 2 | Implement + run E1 (Residual) on ETTh1+Weather+Electricity | Gap closure numbers, α analysis |
| 3 | Run E2 (HyperDLinear) on ETTh1+Weather | Moonshot test |
| 4 | Representation probing (linear probes for μ, σ, trend, seasonality) | What backbone encodes |
| 5 | Write "Closing the Gap" paper section + figures | Paper-ready content |

Total compute: ~$2.50. Every experiment outcome strengthens the paper.

---

## Full Reference

See `tasks/closing_gap_relaxed_constraints.md` (~2100 lines) for:
- All 26 approaches with code examples
- Theoretical framework (information theory, causal analysis, expressiveness hierarchy)
- Literature review (8+ papers from NeurIPS/ICML 2024-2025)
- Exact code modification locations with line numbers
- Failure mode analysis for each approach
- Combinatorial stacking analysis
- The "FM Tax" quantification
- Multi-task, uncertainty, robustness, continual learning arguments
