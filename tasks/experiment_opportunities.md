# Experiment Opportunities — Audited Against Current Paper Content

**Date:** 2026-04-09 (audited)
**Paper state:** main.tex with fixes applied (12-79%, 3-seeds, p<0.001, ReMoE citation)
**Deadline:** May 4-6, 2026 (~4 weeks)

---

## VALIDATED: Existing Moirai results ready to add (ZERO cost)

**Status: VERIFIED.** All 6 Moirai datasets use identical config (K=5, top_k=2, frozen, raw router, 327K params). The 3 in-paper results exactly match the paper's Table R. The 3 missing datasets are from the same pipeline run.

| Dataset | Moirai+RR-MoA | Best Fixed | DLinear | Gap vs DLinear | RR-MoA vs Best Fixed |
|---------|--------------|------------|---------|----------------|---------------------|
| ETTh1 | 0.471+/-0.002 | 0.664 | 0.416 | +13.1% | **-29.1%** |
| ETTh2 | 0.446+/-0.015 | 0.481 | 0.341 | +30.7% | **-7.2%** |
| ETTm1 | 0.396+/-0.041 | 0.471 | 0.322 | +23.1% | **-15.8%** |
| ETTm2 | 0.250+/-0.014 | 0.297 | 0.200 | +25.0% | **-15.9%** |
| Weather | 0.209+/-0.004 | 0.238 | 0.208 | **+0.4%** | **-12.3%** |
| Electricity | 0.206+/-0.001 | 0.300 | 0.158 | +30.6% | **-31.1%** |

**Action:** Expand Appendix R table. Add main-text sentence: "On Moirai, RR-MoA wins 18/18 across 6 datasets, narrowing the DLinear gap from 40-75% (MOMENT-small) to 0.4-31%."

**Why this matters for the paper:** The #1 reviewer attack is the DLinear gap. Showing it's 0.4-31% on a better backbone (vs 40-75% on MOMENT-small) reframes the gap as a backbone quality issue, not a method limitation. This directly addresses the paper's weakest dimension (significance).

---

## Audited Experiment List

### KEEP: Directly relevant to current paper claims

**E1. LoRA + Unfreezing Ablation** — RELEVANT
- Addresses open reviewer gap C5
- Paper currently has 108-run LoRA sweep but only frozen; no unfreezing ablation
- Directly tests whether co-adaptation generalizes beyond AdaMix to LoRA
- Cost: ~1 GPU-hour. Script exists.

**E3. 50-Epoch Moirai+RR-MoA on Weather** — RELEVANT
- Paper already shows 50-epoch improves MOMENT-small by 8-14% (Appendix)
- Moirai Weather is at parity (0.209 vs 0.208). 50 epochs could beat DLinear.
- Would create strongest possible counter to DLinear criticism
- Cost: ~1 GPU-hour. Script exists (modify epochs param).

**E11. Statistics-Only Router** — RELEVANT
- Directly tests Proposition 2's prediction that routing signal is in (mu, sigma)
- Complements existing rawness-vs-bypass ablation (Table 4)
- Simple implementation: add `router_input_mode="stats"` branch
- Cost: ~30 min GPU + trivial code change.

**E12. Load-Balancing Loss on AdaMix** — RELEVANT
- Shows standard MoE fix fails, proving diagnosis is non-trivial
- Directly relevant to related work section (ReMoE, Switch Transformer)
- Cost: ~30 min GPU + small code change.

### DEMOTE: Lower relevance to current paper narrative

**E2. Moirai Multi-Horizon** — DEMOTED
- Interesting but secondary. Multi-horizon on MOMENT-small (24/24 wins) is already strong.
- The DLinear-gap-narrows-with-horizon story is already told on MOMENT-small.
- Cost: ~2 GPU-hours. Not worth the effort given deadline pressure.

**E4. Routing Signal Ratio on Moirai** — DEMOTED
- Moirai doesn't use RevIN, so the signal ratio framework doesn't directly apply.
- The rho=-0.96 is already validated across 7 datasets on MOMENT-small.
- Would be interesting but tangential to the main narrative.

**E5. Direct MI Estimation** — DEMOTED
- Technically elegant but the rho=-0.96 proxy is already compelling enough.
- Adds complexity without changing the conclusion.
- CPU-only but ~2 hours of coding with diminishing returns.

**E6. DishTS-Style Router** — REMOVED
- Novel but extends beyond the paper's current scope.
- Would dilute the "diagnosis + simple fix" narrative.
- Better as a follow-up paper contribution.

**E7. Expert Diversity Analysis** — DEMOTED
- Nice-to-have but the routing-vs-ensemble comparison (uniform routing 30-75% worse, independent ensemble 36-46% worse) already addresses this.
- Marginal value over existing evidence.

### REMOVE: Not relevant or infeasible before deadline

**E8. Moirai-MoE Internal Routing** — REMOVED
- Requires significant new code for a different model.
- Moirai-MoE doesn't use RevIN, so the finding would likely be negative (no collapse).
- Better as future work (already listed as Limitation 2).

**E9. Scaling Behavior** — REMOVED
- MOMENT-large results already exist for 3 datasets in the cross-backbone table.
- Full scaling study requires too much compute for 4 weeks.

**E10. Learned Router Normalization** — REMOVED
- This is a follow-up paper, not an experiment for this submission.

**E13. Gradient Stopping** — REMOVED
- Interesting mechanistic test but doesn't address any reviewer critique.
- The Frozen Paradox is already established with 5 extended FT configs.
- Adds complexity without strengthening the core claims.

---

## FINAL PRIORITY ORDER

| Priority | Experiment | Cost | Addresses |
|----------|-----------|------|-----------|
| **1** | Add Moirai 6-dataset table to paper | ZERO | DLinear gap (#1 weakness) |
| **2** | E3: 50-epoch Moirai Weather | 1 GPU-hr | Could beat DLinear (headline) |
| **3** | E11: Stats-only router | 30 min | Validates Proposition 2 directly |
| **4** | E1: LoRA + unfreezing | 1 GPU-hr | Closes reviewer gap C5 |
| **5** | E12: Load-balancing loss | 30 min | "Obvious fix fails" evidence |

**Total GPU cost for all 5: ~3 GPU-hours + ~$1 Modal cost**

### Schedule
- **This week:** Item 1 (add Moirai table — no computation) + Items 3, 5 (fast GPU experiments)
- **Next week:** Items 2, 4 (medium GPU experiments)
- **Week 3-4:** Integrate results, revise paper, polish

**Expected acceptance improvement: 35-45% -> 45-55% if items 1-3 succeed.**
