**Response to the Area Chair**

We thank the AC for numbering the meta-review points and for holding the discussion to them. Those numbers gave us a clear structure to work against, and they told us where to spend the window. We report two updates to our rebuttal: one bears directly on meta-review point 1, and one corrects a sentence of our own. `[NEW]` marks discussion-phase work.

**Meta-review point 1 (reframing).** We take this point as fair, and we have acted on it. Our rebuttal defended the reframing with a 65:11 count of Residual-IA+ against a from-scratch DLinear. That anchor is the simplified single-Linear variant, while Residual-IA+'s raw branch is NLinear, the stronger from-scratch model on level-drifting data; both facts are in the submission, in App. A and App. P. `[NEW]` We have since run the control that isolates the anchor effect: a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer, epochs and batch), at n=10 across nine datasets. We reproduce the published DLinear anchor on all six.

We find the anchor effect is real on three of the nine, all level-drifting, with DLinear the equal or stronger anchor on the other six. Against the stronger anchor on each dataset, we match or beat it on four of the six at a net mean gap of −0.9%, win all 20 cells on Solar, and hold parity on Traffic, which R(D)=0.14 predicts before any training. On Exchange we attribute the margin to level anchoring, and we now report that cell as parity.

We scored the 65:11 count against DLinear alone. In the revision we will name the anchor as the undecomposed single-Linear variant of Zeng et al. wherever that count and the abstract sentence appear, add the NLinear anchor to the grid, lead with the count against the stronger of the two, and move the DLinear comparison and the backbone-contribution analysis into the main text.

On the reframing itself, we draw the scope the AC asks for from the paper. The title calls RR-MoA a causal intervention for routing collapse, the abstract reaches the method as the prescription that follows, and two of three contributions are the diagnosis and R(D). We will state the accuracy claim at the scope the controls support, beside them.

**Meta-review point 2 ((M,Σ) and the ablations).** We re-scoped Observation 1 in our rebuttal as a lower-bound vulnerability predictor, with the [μ,σ]-only and temporal-shuffle ablations populating the denominator of R(D). That answer stands as given.

**The other meta-review points.** We answered points 3 to 8 in the rebuttals and tagged them by number: 3 to 5 in our reply to jemj, and 5 to 8 in our reply to 8b2Z.

**A correction to our rebuttal.** One sentence in our rebuttal described Moirai's non-learnable normalization as stripping the (M,Σ) statistics from the hidden states. §4.1 states the opposite, that non-learnable I/O scaling leaves those statistics in the hidden states the router reads. We confirm the paper's claim is the correct one and holds throughout, and we withdraw that sentence. §4.1, Appendix H and Proposition 1 describe one account, and we will make that relationship explicit. The submission itself needs no change here.

The diagnosis stands as submitted: normalization-induced routing collapse, the 720-run rescue sweep that leaves 89% of the gap to an input-side fix, and R(D) as a training-free predictor that still calls the Traffic null before training. What changes is the accuracy claim reported beside them.

Both updates came out of the discussion, and the paper is more accurate for them. We appreciate the time the AC has given this thread, and we can answer anything further while it is open.
