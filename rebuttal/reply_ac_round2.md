**Response to the Area Chair**

We are grateful to the AC for the numbered meta-review, for the explicit priority on points 3 to 8, and for crediting the causal analyses and the signal ratio's practical value. That priority told us where to spend the window. Each rebuttal quotes the AC's wording and tags the points by number: 3 to 5 in our reply to Reviewer jemj, 5 to 8 in our reply to Reviewer 8b2Z. Reviewers jemj and 8b2Z have since replied with no points left open, and 8b2Z raised their score. The discussion has turned to points 1 and 2, and we report below where those landed, with one correction to a sentence of our own. `[NEW]` marks discussion-phase work.

It helps to set those two points beside the paper's three contributions: the diagnosis of routing collapse as a failure of the router's input, Observation 1 and R(D) as a training-free predictor of it, and RR-MoA as the intervention that follows. The first two are the ones the AC credits, and the discussion has left both untouched. What it has changed is the accuracy claim attached to the third.

**Point 1 (reframing).** We accept this in full and have acted on it. Our rebuttal answered it with a 65:11 win count against a from-scratch DLinear. Reviewer Pm4m then observed that this anchor is a plain single Linear, while Residual-IA+'s own raw branch is NLinear, the stronger model on level-drifting data. The observation is correct and the control was worth running. `[NEW]` We ran a from-scratch NLinear under that anchor's protocol, at n=10 across nine datasets; our anchor reproduces the published numbers on all six of the paper's datasets.

Against the stronger anchor, Residual-IA+ matches or beats it on four of the six, and is 0.9% better in MSE on average. It wins all 20 cells on Solar and holds parity on Traffic, which R(D)=0.14 predicts before any training. On Exchange the margin comes from level anchoring, so we now report that cell as parity. The full grid is in our reply to Reviewer Pm4m.

In the revision we will name the anchor accurately wherever the 65:11 count and the abstract sentence appear, add the NLinear anchor to the grid and lead with the count against the stronger of the two, and move the DLinear comparison and the backbone-contribution analysis into the main text. The scope the AC asks us to claim is the one the paper already claims in its title, a causal intervention for routing collapse.

**Point 2 ((M,Σ) and the ablations).** In our rebuttal we re-scoped Observation 1 as a lower-bound predictor of vulnerability. R(D) is a ratio, and the [μ,σ]-only and temporal-shuffle ablations measure its denominator, the part normalization leaves behind. That answer has not been contested.

**Points 3 to 8.**

- **Task diversity (3).** `[NEW]` Classification and anomaly runs keep routing entropy healthy in every cell (1.46–1.59 and 1.52–1.61, against a log 5 ≈ 1.61 ceiling). With imputation that is four task types.
- **Frozen backbone (4).** Table 1 has RR-MoA beating the best fixed adapter at Last-2 and Last-4 unfrozen as well as frozen (ETTh1 −44% / −29% / −32%), frozen strongest on 4/6.
- **Memory (5).** The router reads one univariate window, about 0.26 MB at batch 128 and L=512, roughly 64× smaller than the hidden states a conventional router reads.
- **Theory (6).** Observation 1 covers what normalization destroys, Proposition 1 how collapse unfolds, Proposition 2 when it starts, and R(D) which datasets are vulnerable. Dataset-specific structure remains open, as Reviewer 8b2Z notes.
- **Expert scaling (7).** `[NEW]` At K=15 and K=20 MSE stays stable and entropy holds at 91–97% of its log K ceiling.
- **Beyond IN (8).** BatchNorm (0.62→0.004), GroupNorm (0.51→0.000) and `[NEW]` an input-position LayerNorm (→0.000) collapse alike, with the no-norm control healthy at 0.82 and a vision replication in App. H.1.

**A correction to our rebuttal.** One sentence described Moirai's non-learnable normalization as stripping the (M,Σ) statistics from the hidden states. §4.1 states the opposite. The paper's claim is the correct one and holds throughout, and we withdraw that sentence. §4.1, Appendix H and Proposition 1 all state the same criterion, and the revision will say so in one place. The submission itself needs no change here.

Every meta-review point now has an answer. Both updates are changes of naming and framing rather than of evidence: every experiment stands, and the diagnosis and the predictor stand as submitted. We would be glad to see the paper weighed on the diagnosis and the predictor, which this round has left intact. We are grateful for the care the AC has given this thread, and can answer anything further while it is open.
