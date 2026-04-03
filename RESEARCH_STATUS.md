# Research Status: Adapter Architecture Search for TSFMs

**Target:** NeurIPS 2026 (Abstract: May 4, Paper: May 6)
**Last updated:** 2026-04-03

---

## Core Thesis

Adapter architecture is an overlooked design dimension for Time Series Foundation Models. We formalize **Adapter Architecture Search (AAS)** — searching over PyTorch `nn.Module` adapter programs instead of discrete hyperparameter configurations — and show that even simple random search in the right search space consistently outperforms hand-designed adapters.

---

## What's in the Paper (Done)

### Theoretical Contributions
- **6 formal definitions**: Adapter, Architecture Space, AAS Problem (bilevel optimization), Discrete Space C (2,160 configs), Code Space G_code, Template Space G_tmpl (~1,200 architectures)
- **Proposition 1 (Strict Hierarchy)**: C ⊂ G_tmpl ⊂ G_code with proof sketch
- **Proposition 2 (Budget Sufficiency)**: P(success) = 1-(1-f)^B where f ≈ 0.03-0.05. B ≥ 60 suffices for 95% success. Explains why random ≈ evolution.
- **Search space expressiveness table**: 3-column (C, G_tmpl, G_code) showing winning architectures require G_code

### Empirical Results (25/25 wins)
| Dataset | Seeds | Horizons | Wins |
|---------|-------|----------|------|
| ETTh1 | 42,43,44 | 96,192,336,720 | 7/7 |
| ETTm1 | 42,43,44 | 96,336,720 | 6/6 |
| ETTh2 | 42,43,44 | 96 | 3/3 |
| ETTm2 | 42,43,44 | 96 | 3/3 |
| Weather | 42,43 | 96 | 2/2 |
| Electricity | 42 | 96 | 1/1 |
| **Cross-backbone** (MOMENT-large, Chronos, Moirai) | | | **5/5** |

Average improvement: ~9% MSE over best hand-designed baseline (conv).

### Ablations & Analysis (Done)
- **Budget scaling** (14 configs on ETTh1+ETTm1): All beat baseline, even B=20
- **Transfer matrix** (5×5): 7/20 off-diagonal wins (35%). Weather universally easy (4/4), ETTh2 hard (0/4)
- **Factorial decomposition**: Search space is dominant factor (p=0.031, d=0.82)
- **LLM patterns**: 19.6% better individually (p=0.003) but 25% invalid rate
- **Design principles**: Conv1d, attention, BatchNorm help; deep MLPs, last-token hurt
- **Crossover ablation**: Random ≈ crossover (no advantage from recombination)

### Discovered Architectures
- Depthwise separable conv (from MobileNet) — Weather
- Conv1d + BatchNorm (from ResNet) — ETTh1
- Learned feature attention (from SE-Net) — ETTm1
- None of these exist in any published TSFM adapter

### Paper Stats
- 72 citations, 15 pages (10 content + 5 references), compiles cleanly
- Title: "Adapter Architecture Search for Time Series Foundation Models"

---

## What We Tried That Failed

| Approach | Result | Why it failed |
|----------|--------|---------------|
| Evolutionary crossover | ≈ random | Template grammar too small for selection pressure to matter |
| DARTS supernet | 3/6 win (soft mixture) | Discretization kills it (1/6 retrain). Weight coupling. |
| DARTS retrain | 1/6 win | Discretization gap — soft ensemble > any single arch |
| AdaMix (MoE router) | 1/18 win | Router collapses to single expert despite load balancing |
| Spectral-guided search | No signal | MOMENT normalizes hidden states — spectrally homogeneous |
| LLM as search operator | Loses to random | 25% invalid code wastes evaluation budget |

**Key lesson:** The search OPERATOR doesn't matter in G_tmpl. The search SPACE is what matters.

---

## What's Running Now (RACE VM)

### 1. Adapter Selector Feasibility (running)
**Question:** Can raw time series features predict which adapter architecture works?

**Status:** Cross-evaluation matrix in progress (6 portfolios × 6 datasets = 36 trainings)

**Early result (promising):** Raw TS features DO discriminate datasets (CV > 0.3):
- acf_24: 0.135–0.870 (daily cycle strength)
- trend_strength: 0.000–0.908
- kurtosis: 0.04–7876 (Weather is extreme)
- Unlike hidden-state spectra (which failed), raw data features have clear signal

**If successful → Novel contribution:** Zero-shot adapter selection from dataset features. No per-dataset search needed.

### 2. Augmented Grammar G_tmpl+ (queued, runs after selector)
**Question:** Does adding LLM-discovered patterns as grammar rules improve the search space?

**What we're adding to the grammar:**
- Depthwise separable Conv1d (MobileNet pattern)
- Feature attention with softmax gating (SE-Net pattern)
- BatchNorm/LayerNorm feature transforms
- Residual connections

**Experiment:** Compare G_tmpl (base, ~1,200 archs) vs G_tmpl+ (augmented, ~3,600 archs) on all 6 datasets, same budget.

**If G_tmpl+ > G_tmpl → Novel contribution:** LLM contributes to the search SPACE, not the search operator. Automated grammar augmentation via LLM discovery.

---

## Planned Next Steps

### If Augmented Grammar Wins
1. Formalize as G_tmpl ⊂ G_tmpl+ ⊂ G_code (extends Proposition 1)
2. Run multi-seed (3 seeds × 6 datasets)
3. Add to paper as new Section 3.4: "Grammar Augmentation via LLM Discovery"
4. Paper story: "LLM discovers patterns → formalized as grammar rules → random search in richer space"

### If Adapter Selector Works
1. Implement Feature-Conditioned PCFG (FC-PPS)
2. Production probabilities conditioned on dataset features
3. Paper story: "Data-conditioned program synthesis for adapter architecture"
4. Needs more datasets (add 4 from Monash/GIFT-Eval for 10 total)

### If Both Work (Best Case)
Combine: Feature-Conditioned sampling from the Augmented Grammar.
Paper becomes: problem formulation + augmented grammar + data-conditioned sampling + 25/25 wins

### Pre-Submission Polish (regardless)
- [ ] NeurIPS checklist (checklist.tex)
- [ ] Anonymize (remove RACE VM references, author names)
- [ ] Supplementary: full adapter code listings, additional tables
- [ ] Proofread entire paper
- [ ] Abstract ≤ 1500 characters for OpenReview

---

## Honest Assessment

| Scenario | Probability | Acceptance % |
|----------|-------------|-------------|
| Paper as-is (no novel method) | Baseline | 55-60% |
| + Augmented grammar works | 50% chance | 60-65% |
| + Adapter selector works | 40% chance | 65-70% |
| + Both work + combined | 25% chance | 70-75% |
| NeurIPS rejects → ICLR 2027 backup | — | Sept 2026 deadline |

The paper is solid on theory + evidence. The method novelty gap is the risk. The current experiments are our best remaining shots at closing it.

---

## File Locations

| What | Where |
|------|-------|
| Paper | `main.tex` (root) |
| Main results | `results/standard_evolution/*.json` |
| Budget ablation | `results/budget_ablation/*.json` |
| Transfer matrix | `results/transferability/transfer_H96.json` |
| TRACE baselines | `results/trace_baseline/*.json` |
| Crossover ablation | `results/crossover_ablation/*.json` |
| DARTS results | `results/darts_aas/*.json` |
| AdaMix results | `results/adamix/*.json` |
| Ensemble results | `results/ensemble_aas/*.json` |
| Spectral validation | `results/spectral_validation/validation.json` |
| Adapter selector | `results/adapter_selector/` (in progress) |
| Augmented grammar | `results/augmented_grammar/` (queued) |
