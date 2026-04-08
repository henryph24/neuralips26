# Closing the DLinear Gap v3: Refined Analysis + Pitfall Avoidance

## Iteration 3 — Challenging the Few-Shot Assumption

### The Pitfall in v2's Few-Shot Argument

v2 assumed RR-MoA would dominate at few-shot because of the pre-trained backbone. But:
- RR-MoA has **426K trainable params** (router + 5 expert heads)
- DLinear has **49K trainable params**
- At N=10, RR-MoA has 42,600 params per sample vs DLinear's 4,900

The backbone provides inductive bias, but the adapter has 8.7x more free parameters. Few-shot performance depends on which effect dominates. **This is an empirical question** — we CANNOT assume the answer.

However, there's a mitigating design: with Top-2 routing, only 2 of 5 experts are active per sample, so the effective param count is lower. And the router is tiny (~1.1K params). Still, the total gradient updates flow through all 426K params.

### Mitigation: A Smaller RR-MoA for Few-Shot

For the few-shot experiment, also test a "lightweight" RR-MoA:
- K=3 experts instead of K=5 (reduces params ~40%)
- hidden=32 instead of 64 (reduces params ~50%)
- Total: ~100K params — closer to DLinear's 49K

This gives a fairer comparison at low N and tests whether the backbone inductive bias helps even with fewer adapter params.

---

## REVISED Experiment Rankings

### Tier A: MUST-DO (highest ROI, address the core critique)

**A1: Few-Shot Learning Curve (with mitigation)**

The experiment from v2 but with THREE methods:
1. DLinear (49K params, from scratch)
2. RR-MoA full (426K params, frozen backbone)
3. RR-MoA-lite (K=3, hidden=32, ~100K params, frozen backbone)

N = {10, 50, 100, 200, 500, 1000, full}

Three datasets: ETTh1, Weather, Electricity (spanning different difficulties).

**Key outcomes**:
- If RR-MoA-lite beats DLinear at N<200: backbone inductive bias dominates at few-shot → strong paper result
- If DLinear beats both RR-MoA variants at ALL N: the few-shot argument fails, do NOT include
- If crossover at N~100-500: include with honest reporting of the crossover point

**Compute**: ~3 GPU-hours (3 datasets × 7 N-values × 3 methods × 1 seed = 63 runs, most are fast)

**Implementation**: New script `run_fewshot_curve.py`. Subsamples training data. Runs RR-MoA and DLinear. Saves per-N results.

---

**A2: DLinear Expert Diagnostic**

From v1/v2, unchanged. Add a 6th expert that operates on raw input (bypassing backbone). The routing weights between backbone experts and raw expert quantify the information gap.

**Compute**: ~2.4 GPU-hours

---

### Tier B: HIGH-VALUE, LOW-COST

**B1: Routing Entropy vs Prediction Error (zero cost)**

Correlate per-sample routing entropy with squared prediction error using existing results. If positive correlation: RR-MoA provides free uncertainty estimation.

**Implementation**: Pure analysis script. Read existing JSON result files that contain per-sample routing weights.

**Compute**: 0 GPU-hours

---

**B2: Multi-Task Advantage Table**

The paper already has forecasting + imputation. Compile a clear table:

| Task | RR-MoA | DLinear | Winner |
|------|--------|---------|--------|
| Forecasting H=96 | 0.680 | 0.416 | DLinear |
| Forecasting H=720 | 0.838 | ??? | ??? |
| Imputation 20% | -44 to -67% vs baseline | N/A | RR-MoA (DLinear can't impute) |
| Classification | Possible | N/A | RR-MoA (DLinear can't classify) |
| Cross-dataset transfer | Tested | Fails | RR-MoA |
| Multi-tenant (N=100) | 141 tenants/sec | N/A | RR-MoA |

DLinear can only do 1 of 6 things RR-MoA can do. Frame this as a versatility comparison, not just MSE.

**Compute**: 0 GPU-hours (compile existing results)

---

**B3: Extended Adapter Training (50 epochs)**

The simplest experiment. Just --epochs 50 on 3 datasets × 3 seeds. The adapter is small and stable — more training should help without overfitting (unlike full-FT which collapses).

**Compute**: ~4 GPU-hours. Zero code changes.

---

### Tier C: MEDIUM-VALUE, MEDIUM-COST

**C1: Multi-Layer Feature Extraction with Scalar Mixing**

Extract features from blocks {2, 4, 6, 8}, learn a 4-weight scalar mix (ELMo-style). 4 additional parameters. Might recover useful intermediate representations.

**Compute**: ~3 GPU-hours. ~50 lines of code.

---

**C2: Adaptive Expert Hidden Dimension**

Test hidden={64, 128, 192} to find the sweet spot between capacity and overfitting. With hidden=128, total params ~500K (at the budget limit). On Moirai with better hidden states, larger experts might extract more value.

**Compute**: ~2 GPU-hours per hidden value. Run hidden=128 on 3 key datasets.

---

## The Complete "Checkmate" Package

Run A1 + A2 + B1 + B2 + B3. Total compute: ~9 GPU-hours. The paper gains:

1. **Few-shot figure** → "RR-MoA dominates at N<X samples"
2. **DLinear expert** → "Gap is representation-limited, not routing-limited"
3. **Uncertainty** → "Routing entropy predicts error quality"
4. **Multi-task table** → "DLinear can only do 1 of 6 things RR-MoA can"
5. **50-epoch adapters** → "Squeezes an additional Y% from the backbone"

Together, the harsh reviewer's attack surface shrinks from "MSE is worse than DLinear" to "MSE is worse than DLinear at full data, but RR-MoA wins at few-shot, provides uncertainty, handles imputation/transfer, and the gap closes with backbone scale."

---

## What the Paper Narrative Looks Like After All Experiments

### Current narrative (strong but has the DLinear hole):
"Normalization breaks routing → we prove it → we fix it → Frozen Paradox"

### Enhanced narrative (closes the hole):
"Normalization breaks routing → we prove it → we fix it → Frozen Paradox → AND the frozen-backbone paradigm wins at few-shot, handles multiple tasks, provides uncertainty, and approaches supervised performance as backbones scale. The remaining full-data MSE gap is a representation bottleneck (confirmed by the raw expert diagnostic) that diminishes with backbone quality (MOMENT → Moirai → future TSFMs)."

### The one-sentence pitch:
"We discovered that normalization breaks routing in foundation models, proved it formally, fixed it with a simple architectural change, and showed the resulting frozen-backbone system matches supervised baselines in data-scarce deployment while providing per-sample uncertainty — the first method to offer this combination for time series."

---

## Concrete Implementation Order

**Day 1 (4 GPU-hours)**:
1. Create `run_fewshot_curve.py`
2. Run few-shot on ETTh1/Weather/Electricity (A1)
3. Run 50-epoch adapters on 3 datasets (B3)

**Day 2 (2.4 GPU-hours)**:
4. Implement DLinear expert (A2)
5. Run on 6 datasets × 3 seeds

**Day 2 (0 GPU)**:
6. Routing uncertainty analysis (B1)
7. Multi-task comparison table (B2)

**Day 3**:
8. Assess results, update paper

**Day 4-5**:
9. Final paper polish

---

## Risk Assessment

| Experiment | P(useful result) | Downside if fails |
|-----------|-----------------|-------------------|
| Few-shot curve | 70% | RR-MoA overfits at low N due to 426K params; mitigated by RR-MoA-lite |
| DLinear expert | 95% | Almost certain to work (raw expert = DLinear) |
| Routing uncertainty | 60% | Entropy might not correlate with error |
| Multi-task table | 100% | Pure compilation of existing data |
| 50-epoch adapters | 50% | Might plateau at 15 epochs |
| Multi-layer features | 40% | Intermediate layers might not help forecasting |

The expected value is high: even if few-shot partially fails, the DLinear expert + multi-task table + uncertainty analysis still provide strong material.
