# Closing the DLinear Gap: Deep Analysis & Experiment Plan

## The Problem

| Backbone | ETTh1 | ETTm1 | Weather | ETTh2 | ETTm2 | Electricity |
|----------|-------|-------|---------|-------|-------|-------------|
| DLinear (from scratch) | **0.417** | **0.322** | **0.208** | **0.341** | **0.200** | **0.158** |
| MOMENT-small + RR-MoA | 0.680 | 0.564 | 0.276 | 0.778 | 0.538 | 0.402 |
| Moirai-small + RR-MoA | 0.471 | 0.396 | **0.209** | 0.446 | 0.250 | 0.206 |
| Gap (Moirai vs DLinear) | +13% | +23% | **+0.5%** | +31% | +25% | +30% |

Weather on Moirai matches DLinear. Everything else has a 13-31% gap on Moirai, 40-90% on MOMENT.

## Root Cause Analysis

DLinear: `Y = W @ X` where W is (96, 512). It maps raw input directly to output. 49K params, trained per-dataset from scratch. It sees ALL raw information.

RR-MoA: `Y = Σ w_k * Expert_k(Backbone(X))`. The backbone processes X through RevIN + patching + 8 transformer blocks + LayerNorm cascade. By the time hidden states reach the experts, per-window temporal statistics have been stripped (RevIN), spatial patterns have been abstracted (transformer), and signal has been homogenized (LayerNorm × N).

**The gap is an information bottleneck**: the frozen backbone discards information that DLinear retains. RR-MoA perfectly routes over this impoverished representation — it can't recover information the backbone threw away.

Evidence:
1. Gap narrows with better backbone (MOMENT → Moirai): backbone quality drives the gap
2. Gap vanishes on Weather-Moirai: when the backbone retains enough info, routing matches supervised
3. The Frozen Paradox: unfreezing makes things WORSE, so the issue isn't expressivity — it's normalization-induced information loss

## Six Directions to Close the Gap

---

### Direction 1: The "DLinear Expert" (HIGHEST IMPACT)

**Idea**: Add a 6th expert that maps raw input → output directly, bypassing the backbone.

```python
class RawLinearExpert(nn.Module):
    def __init__(self, input_len=512, output_dim=96, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden), nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, raw_input):  # (B, 512) → (B, 96)
        return self.net(raw_input)
```

The router already sees raw input and already decides per-sample which expert to use. If DLinear is better for a given sample, the router learns to select the raw expert. The mixture UPPER-BOUNDS at DLinear's performance.

**Why this is clean**:
- Doesn't change the routing mechanism
- Doesn't require unfreezing
- Naturally degrades to DLinear when backbone is useless
- Naturally degrades to RR-MoA when backbone is valuable
- The ROUTING PATTERN is itself a finding: "X% of samples benefit from deep features, (100-X)% are better served by raw signal"
- ~78K additional params (well within 500K budget)

**Expected result**: MSE should approach DLinear on all datasets. The router will learn to use the raw expert heavily on MOMENT-small (where backbone features are poor) and less on Moirai (where backbone features are rich).

**Interpretation for paper**: "Adding a raw-signal expert closes the DLinear gap to within X%, confirming that the remaining gap is a backbone representation bottleneck, not a routing limitation. The per-sample routing weight between backbone and raw experts provides an automatic diagnostic of backbone representation quality."

**Implementation**: ~20 lines of code change in `run_rr_moa.py`. Add `--include-raw-expert` flag. When enabled, K=6 with the 6th expert receiving raw_input instead of hidden_states.

**Compute**: 6 datasets × 3 seeds × ~8 min = ~2.4 GPU-hours

---

### Direction 2: Skip Connection from Raw Input

**Idea**: Add a learned residual: `Y = RR-MoA(H, X_raw) + α * Linear(X_raw)`

Simpler than Direction 1. The raw signal always contributes, with a learned weight α. This is like a residual connection that preserves information the backbone lost.

**Implementation**: Add `self.raw_residual = nn.Linear(input_len, output_dim)` and `self.alpha = nn.Parameter(torch.tensor(0.0))` to RawRoutedMoA.

**Downside**: Less interpretable than Direction 1. We can't see per-sample routing between backbone and raw.

**Compute**: Same as Direction 1 (~2.4 GPU-hours)

---

### Direction 3: Multi-Layer Feature Extraction

**Idea**: Instead of only the last encoder block, extract from blocks {2, 4, 6, 8} and concatenate.

Early blocks capture local patterns (useful for short horizons), later blocks capture global patterns. The adapter sees a richer, multi-scale representation.

**Implementation**: Modify `_extract_features_batch()` to register hooks on multiple blocks. Concatenate features along the d_model dimension: (B, T, 4*d_model). Add a projection layer (4*d_model → d_model) before the experts.

**Downside**: 4x memory for features. Projection layer adds complexity. The intermediate blocks might not add useful information for forecasting.

**Compute**: ~3 GPU-hours (slightly slower due to larger features)

---

### Direction 4: Larger Expert Hidden Dimension

**Idea**: Current hidden=64 is a severe bottleneck. DLinear uses 512→96 directly. Try hidden=128, 192, 256.

With hidden=256: each expert is nn.Linear(512, 256) + nn.Linear(256, 96) = 156K params. 5 experts = 780K — over the 500K budget. With hidden=128: each expert ~80K, 5 experts = 400K — fits.

**Implementation**: Add `--hidden` flag to `run_rr_moa.py` (may already exist). Run with --hidden 128.

**Expected impact**: Small. The bottleneck is information quality, not adapter capacity. Bigger experts can't recover information the backbone lost.

**Compute**: ~2.4 GPU-hours

---

### Direction 5: Adapter-Only Extended Training

**Idea**: 15 epochs might not be enough for adapters to learn optimal mappings from frozen features. Try 30, 50, 100 epochs (adapter only, backbone stays frozen).

**Rationale**: Unlike full-FT (where more epochs → overfitting to one expert → collapse), adapter-only training is stable. The adapter might continue improving with more epochs.

**Expected impact**: Small-medium. If the representations are poor, no amount of training helps. But if there's useful signal that 15 epochs didn't fully exploit, more training could help.

**Compute**: ~4-6 GPU-hours for 50ep on 6 datasets × 3 seeds

---

### Direction 6: Moirai with Native Data Pipeline

**Idea**: Instead of our custom patching → zero-padding → encoder approach, use Moirai's native data pipeline (uni2ts) which handles patching, normalization, and input projection natively.

**Rationale**: Our custom forward pass (patch → pad/in_proj → encoder) might not match what Moirai saw during pre-training. The native pipeline would produce representations the encoder was actually trained to process.

**Downside**: Requires significant refactoring. The uni2ts pipeline has its own data format, tokenization, and batching. Integrating this with our train loop is non-trivial.

**Expected impact**: Could be large — if the backbone sees properly formatted inputs, its representations should be much better. This might be why Moirai-small (even with zero-padding) nearly matches DLinear on Weather: Weather's distribution happens to be compatible with the hacky input format.

**Compute**: ~4 GPU-hours for experiments, but ~4-8 hours of engineering work.

---

## Recommendation: Direction 1 (DLinear Expert) First

**Why this is the best first experiment:**

1. **It answers the fundamental question**: "Is the gap routing-limited or representation-limited?" If adding the raw expert closes the gap, the answer is definitive: representation-limited. This is a FINDING, not a hack.

2. **It's architecturally clean**: Still a mixture of experts, still per-sample routing, still frozen backbone. The only difference is one expert bypasses the backbone.

3. **It creates a new narrative**: "RR-MoA with a raw-signal fallback expert achieves X.XX on ETTh1 — within Y% of DLinear — while maintaining per-sample routing and multi-tenant deployment capability."

4. **The routing weights are diagnostic**: If the router sends 80% of samples to the raw expert on MOMENT-small but only 20% on Moirai, that QUANTIFIES the backbone quality difference. This is publishable.

5. **It's fast**: ~20 lines of code, 2.4 GPU-hours.

**If Direction 1 succeeds** (gap closes to <10% on most datasets): Include in paper as a "representation diagnostic" experiment. Frame as: "The raw expert experiment confirms that the DLinear gap reflects backbone representation quality, not routing limitation."

**If Direction 1 partially succeeds** (gap narrows but doesn't close): Still valuable — the routing weights show per-sample backbone vs raw preference.

**If Direction 1 fails** (gap doesn't change): Unlikely — adding DLinear as an expert should at worst match DLinear, since the router can learn to always select it.

---

## Implementation Plan

### Phase 1: DLinear Expert (2.4 GPU-hours)
```
Modify: scripts/run_rr_moa.py
  - Add RawLinearExpert class
  - Add --include-raw-expert flag
  - Modify RawRoutedMoA to support K+1 experts where the last takes raw_input
  - Log per-expert routing weights (especially raw expert %)
Run: 6 datasets × 3 seeds, MOMENT-small, frozen, Top-2(6)
```

### Phase 2 (if Phase 1 works): Cross-backbone raw expert
```
Run: same on Moirai-small
Compare: raw expert usage % on MOMENT vs Moirai
Expected: MOMENT uses raw expert more → confirms backbone quality story
```

### Phase 3 (optional): Multi-layer features
```
Only if Phase 1 shows gap is NOT just representation quality
Modify: feasibility/finetune.py to extract multi-layer features
Run: 6 datasets × 3 seeds with 4-layer concatenation
```

---

## What This Means for the Paper

**If DLinear expert closes the gap**: The paper can say "The remaining MSE gap to supervised baselines is entirely attributable to the frozen backbone's representation quality, not to the routing mechanism. When a raw-signal fallback expert is included in the pool (Table X), RR-MoA matches DLinear within Y%, while maintaining per-sample routing and multi-tenant deployment advantages. As backbone quality improves (MOMENT → Moirai → future TSFMs), the raw expert's routing share decreases from Z% to W%, confirming that the frozen-backbone paradigm becomes self-sufficient at scale."

**If it doesn't**: The paper's current framing is already correct and well-defended. The DLinear gap is an acknowledged limitation, the deployment motivation justifies the frozen regime, and the normalization-routing insight is the contribution — not absolute MSE.
