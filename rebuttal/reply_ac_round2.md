**Response to the Area Chair**

We appreciate the AC's numbered meta-review, the explicit priority on points 3 to 8, and the credit given to the causal analyses and the signal ratio's practical value. That priority told us where to spend the window. We tag the AC's points by number in each rebuttal: 1 and 2 for Reviewer Pm4m, 3 to 5 for Reviewer jemj, 5 to 8 for Reviewer 8b2Z. The AC also asked us to clarify factual errors. Where a reviewer's reading differed from what we intended, we traced the cause to our own writing, and we answered each case in that reviewer's thread. Reviewer 8b2Z withdrew two weaknesses, resolved three, narrowed one, left one open and raised their score; we accept that item and their two further requests for the revision. The discussion has turned to points 1 and 2, and we report below where those landed, with one correction of our own. `[NEW]` marks results not in the submitted paper.

We set those two points beside the paper's three contributions: the diagnosis of routing collapse as a failure of the router's input, Observation 1 and R(D) as a training-free predictor of it, and RR-MoA as the intervention that follows. The AC credits the first two, and the discussion has not contested the evidence behind either. What it has changed is the accuracy claim attached to the third.

**Point 1 (reframing).** We accept this in full and we have acted on it. Our rebuttal answered it with a 65:11 win count against a from-scratch DLinear. Reviewer Pm4m then observed that this anchor is a plain single Linear, while Residual-IA+'s own raw branch is NLinear, the stronger model on level-drifting data. We agree. `[NEW]` We trained a from-scratch NLinear under that anchor's protocol, at n=10 across nine datasets, and our anchor reproduces the published numbers on all six of the paper's datasets.

Against the stronger anchor, Residual-IA+ matches or beats it on four of the six, and is 0.9% better in MSE on average. It wins all 20 cells on Solar (−8.7% and −11.2%) and holds parity on Traffic, which R(D)=0.14 predicts before any training. On Exchange the margin comes from level anchoring, so we now report that cell as parity. We report every cell in our reply to Reviewer Pm4m.

In the revision we will name the anchor accurately wherever the 65:11 count and the match-or-beat claim appear, we will add the NLinear anchor to the grid and lead with the count against the stronger of the two, and we will move the DLinear comparison and the backbone-contribution analysis into the main text.

**Point 2 ((M,Σ) and the ablations).** In our rebuttal we re-scoped Observation 1 as a lower-bound predictor of vulnerability. R(D) is a ratio, and the [μ,σ]-only and temporal-shuffle ablations measure its denominator, the part normalization leaves behind.

**Points 3 to 8.**

- **Task diversity (3).** `[NEW]` Classification and anomaly runs keep routing entropy healthy in every cell, 1.38–1.61 against a log 5 ≈ 1.61 ceiling. With imputation that is four task types, and we will add the normalized-router arm Reviewer 8b2Z asks for.
- **Frozen backbone (4).** Table 1 and App. G have RR-MoA beating the best fixed adapter frozen and at Last-2 and Last-4 unfrozen (ETTh1 −44% / −29% / −32%), frozen strongest on 4/6.
- **Memory (5).** The router reads one univariate window, about 0.26 MB at batch 128 and L=512, roughly 64× smaller than the hidden states a conventional router reads.
- **Theory (6).** Observation 1 covers what normalization destroys, Proposition 1 how collapse unfolds, Proposition 2 when it starts, and R(D) which datasets are vulnerable. Dataset-specific structure remains open, as Reviewer 8b2Z notes.
- **Expert scaling (7).** `[NEW]` At K=15 and K=20 MSE stays stable and entropy holds at 95–97% of its log K ceiling.
- **Beyond IN (8).** BatchNorm (0.62→0.004), GroupNorm (0.51→0.000) and `[NEW]` an input-position LayerNorm (→0.000) collapse alike, with the no-norm control healthy at 0.82 and a vision replication in App. H.1.

**A correction to our rebuttal.** One sentence described Moirai's non-learnable normalization as stripping the (M,Σ) statistics from the hidden states. §4.1 states the opposite. The paper's claim is the correct one, and we withdraw that sentence. We had read Moirai through MOMENT, where RevIN strips the statistics and raw routing restores them, and that reasoning does not transfer. §4.1, Appendix H and Proposition 1 all state the same criterion, and the revision will say so in one place.

Every meta-review point now has an answer, with two carried into the revision: the normalized-router arm under point 3, and the dataset-specific structure under point 6. No experiment was retracted. What changed is the anchor the accuracy claim is measured against, and one sentence we withdraw; the diagnosis and the predictor stand as submitted. We thank the AC for the care given to this thread, and we can answer anything further while it is open.
