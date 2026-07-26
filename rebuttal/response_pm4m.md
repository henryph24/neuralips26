**Response to Reviewer Pm4m**

We thank the reviewer for the careful, constructive review, for the close reading down to the router parameter count and α initialization, and for noting that addressing these concerns would raise the score. We address every point below; all supporting results are in the submitted paper unless tagged [NEW].

**W1 (framing / DLinear).** This is the reviewer's central concern, and we take it seriously. The reviewer is correct that our abstract's headline numbers (54/54; 12-79% over full fine-tuning) are all comparisons within the frozen-adapter/PEFT regime. Those are the right baselines for the frozen-adapter question we ask, and we agree a from-scratch DLinear belongs beside them as a benchmark from outside that regime. Against that benchmark, the reviewer's point holds for RR-MoA-proper, the minimal diagnostic probe whose only role is to isolate the router input. But it does not apply to the variant we deploy. Residual-IA+, reported in the submitted Appendix P, matches or beats DLinear rather than trailing it:

- 65 significant wins vs 11 losses (5.9:1); 107/123 match-or-beat (87%) across 6 backbones × 6 datasets × 4 horizons.
- 6/6 at H=192; MOMENT-large 23/24 with zero losses (16 significant wins); Moirai-MoE 15/15 (100%).
- It dominates on non-stationary data at long horizons (ETTm2 -27.0% at H=720; ETTh2 wins at all four horizons).
- Even at the single least-favorable case cited (MOMENT-small, H=96), the net mean gap is -2.8%, a win, with ETTh1 the only loss.

We acknowledge that the reviewer is right about our framing; the issue is how we presented the evidence, not the evidence itself. We led with "5/6 match-or-beat" and titled the appendix "Closing the DLinear Gap," which highlighted the single losing cell (ETTh1/H=96). We will fix the presentation in three ways:

- Promote the full grid, the 65:11 win-count, and the backbone-contribution analysis (App. P.3) into the main text, as the reviewer asks.
- Retitle the section to the outperformance the numbers actually show.
- Reframe the abstract's DLinear comparison as a matter of scope, not a weakness, with wording such as: "Because the router isolates the collapse mechanism rather than maximizing accuracy, the minimal RR-MoA probe trails a from-scratch DLinear on linear-dominated datasets; our deployment adapter Residual-IA+ removes this gap, matching or beating DLinear across six backbones and four horizons."

We will present RR-MoA-proper as a diagnostic probe and Residual-IA+ as the deployed method that clears the DLinear bar; the overselling concern is thus one of framing, which we fix.

**W2 (Figure 2).** We agree the figure is too small and cluttered; we will redraw it larger with non-overlapping modules so it clearly shows the raw-router and frozen-expert paths and the core contribution, raw-input routing as the causal fix.

**W3 (Moirai gains without collapse).** The reviewer is right, and we should have made this clearer. On Moirai there is no collapse, so its 12-29% gain is not collapse-repair. We read it as two results. (1) It confirms the mechanism's prediction: Moirai has no learnable-affine input normalization, so the co-adaptation that drives collapse cannot occur, and it does not collapse. (2) Raw routing still helps, for a different reason: Moirai's non-learnable instance normalization (Kim et al. 2022) also strips per-window location-scale from the hidden states, and the raw input lets the router recover it. This is a routing effect, not the raw-signal predictiveness the reviewer links to App. P.3, because in RR-MoA the raw signal feeds only the router while the experts use frozen hidden states, never the raw input. So the gain cannot come from the Pure Raw-MLP mechanism, where experts read the raw signal directly. We agree the cross-backbone wins should be shown as two regimes: collapse-repair on learnable-affine RevIN backbones, and routing-signal recovery elsewhere, not uniform proof of collapse. The causal chain therefore stands, and we will add the Moirai citation.

**W4 (heavy tuning; generalization beyond six).** On "heavy tuning": the 546-run sweep the reviewer cites (Appendix P) was a one-time search to find the recipe, not a per-dataset cost; the recipe is then fixed, and we agree it should not be presented as a SOTA search. The result is a win over DLinear (65:11), not parity. [NEW] Applied unchanged (gate-init b=-2, 5-epoch warmup, shared NLinear, validation early-stop) to three unseen datasets, Residual-IA+ matches or beats DLinear on 8/9 cells: Solar 3/3 (-9.5%), Traffic 3/3 (parity), Exchange 2/3, no collapse. A recipe that transfers to unseen data without re-tuning is hard to see as dataset-specific engineering, and it speaks directly to "generalization beyond these six is unclear."

On "the raw branch does the work; the TSFM is dead weight (App. P.3)": P.3's "dead weight" is measured against the intermediate Dual-Stream (ETTh2 0.452, ETTm2 0.252), not the final Residual-IA+ (ETTh2 0.346, ETTm2 0.188), and that intermediate itself loses to DLinear. P.3's own backbone-free Raw-MLP MoE loses to DLinear on all six datasets (mean +11.2%): remove the TSFM and you fall behind. [NEW] We ran the clean test: Residual-IA+ vs a same-size backbone-free Raw-MLP MoE, raw branch held fixed so only the backbone differs, at H=192 (where the paper says the backbone helps most). Residual-IA+ wins 18/18 (mean -11.8%), including ETTh2/ETTm2/Electricity, the datasets called dead-weight at H=96, and still matches or beats a 49K DLinear. The backbone's contribution is horizon-dependent: small at H=96, decisively positive at H≥192.

**W5 ((μ,σ) tension).** We understand why this looks like a contradiction, and re-scoping Observation 1 resolves it. R(D) is a ratio. The top is the routing signal in the part that normalization removes (location-scale); the bottom is the routing signal in the part that survives (shape). R(D) is not a claim that (μ,σ) is the whole routing signal. When the ratio is high, stripping (μ,σ) removes most of the routing signal and routing collapses; when it is low (Traffic, R=0.14) the signal lives in the surviving shape, so stripping does not hurt and RR-MoA correctly does not improve. This is consistent with the ablations cited: [μ,σ]-only routing degrades +48% (SSR) and temporal shuffling moves performance ≤3% on 4/6 (the other two improve). We agree the routing signal is broader than (μ,σ): the order-invariant shape that survives normalization is R(D)'s surviving component. We will re-scope Observation 1 explicitly as a vulnerability predictor / lower bound, not a claim that (μ,σ) alone is enough.

**W6 (minor).** The router is 853 params for the reported K=5 model, consistent across the main text, Fig 2, Table A.1, and the code. The K-way head grows with K, so over the expert-count sweep (Table I.3, up to K=10) the count reaches ~1,100 at the high end; the Checklist's "~1.1K" reflects that larger-K count, not the reported K=5, and we will correct it to 853. On the SR-MoA figure, the 13-42% is measured over RR-MoA, not the baselines (SR-MoA beats RR-MoA on all six datasets, Table F.2). Rechecking, the exact range is 23-38%, which we will gladly correct. We will also standardize "fine-tuning" and recheck the flagged R(D) parenthesis.

**Q1 (input length 512 / horizon 96).** The input length is indeed fixed at 512, and the reviewer is right to question it. Multi-horizon results, though, are already in the submitted paper (Table C.1: RR-MoA 12/12 vs best fixed over H∈{96,192,336,720}; Residual-IA+ reaches 6/6 at H=192 in Table P.6); we may not have presented them prominently enough. [NEW] Input-length ablation (L∈{96,192,336,512}, 6 datasets, 3 seeds): no collapse at any L (per-seed entropy 0.89-1.53, far above the 0.00 collapse floor), raw beats normalized routing at all 24 configs (-17% to -84%), and RR-MoA beats the best fixed adapter at 70/72.

*Routing entropy per seed (each cell = seeds 42/43/44; H=96):*

| dataset | L=96 | L=192 | L=336 | L=512 |
|---|---|---|---|---|
| ETTh1 | 1.31/1.39/1.39 | 1.41/1.48/1.42 | 1.46/1.51/1.46 | 1.45/1.45/1.49 |
| ETTh2 | 1.19/1.23/1.28 | 1.28/1.16/1.17 | 1.32/1.06/0.92 | 0.92/0.89/1.13 |
| ETTm1 | 1.34/1.39/1.45 | 1.36/1.42/1.39 | 1.25/1.52/1.39 | 1.48/1.46/1.37 |
| ETTm2 | 1.16/1.31/1.26 | 1.12/1.22/1.14 | 1.16/1.13/1.23 | 1.21/1.26/1.08 |
| Electricity | 1.40/1.38/1.44 | 1.53/1.44/1.43 | 1.48/1.48/1.38 | 1.50/1.48/1.40 |
| Weather | 1.47/1.46/1.47 | 1.37/1.41/1.46 | 1.37/1.32/1.42 | 1.47/1.44/1.30 |

Conclusions hold on both axes.

**Q2 (α inconsistency).** Thank you for catching this; it is a wording slip. α is initialized at σ(0)=0.5 (the logit is a parameter at 0.0); 0.492 is the mean α at the end of the first logged epoch, after one epoch has moved it below 0.5, not the initialization (trajectory 0.5 → 0.492 → 0.468). We will correct the wording.

**On significance.** The lasting contribution is the diagnosis and its predictor, and the reframing above sharpens rather than shrinks it. R(D) is a practical, training-free tool: computed from raw statistics before training, it predicts whether raw routing will help on a new dataset (ρ=-0.88 across nine datasets), and where it will not (Traffic, R=0.14, predicted in advance and confirmed). It transfers beyond our six benchmarks and our specific router, since two distinct variants (SR-MoA and Residual-IA+) confirm it. The diagnosis, the R(D) predictor, and the raw-routing intervention are the contribution we ask the reader to take away.

Due to length constraints we report per-seed results here, and would be glad to add the full per-cell tables to the revised manuscript if accepted. Given the corrected DLinear framing (Residual-IA+ outperforms DLinear 65:11, already in App. P), the re-scoped R(D) explanation, and the added input-length ablation, together with the reviewer's kind indication that addressing the concerns would raise the score, we would be grateful for a reconsideration of the rating.
