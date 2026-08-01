**Response to the Area Chair**

We thank the AC for keeping the discussion focused. Reviewer Pm4m narrowed his review to two points, and both are now answered. We summarise where they landed, since one bears directly on meta-review point 1. `[NEW]` marks discussion-phase work.

**Meta-review point 1 (reframing).** Pm4m observed that our from-scratch calibration anchor is the simplified single-Linear DLinear, while Residual-IA+'s raw branch is NLinear, the stronger from-scratch model on level-drifting data. The observation is correct, and both facts are in the submission, in App. A and App. P. `[NEW]` We ran the control he asked for: a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer, epochs and batch), at n=10 across nine datasets. Our DLinear reproduces the published anchor on all six.

The anchor effect is real on three of the nine, all level-drifting, and DLinear is the equal or stronger anchor on the other six. Against the stronger anchor on each dataset, Residual-IA+ matches or beats it on four of the six at a net mean gap of −0.9%, wins all 20 cells on Solar, and holds parity on Traffic, which R(D)=0.14 predicts before any training. On Exchange the margin is level anchoring, and we now report that cell as parity.

Our earlier 65:11 count was scored against DLinear alone. The revision will name the anchor as the undecomposed single-Linear variant of Zeng et al. wherever that count and the abstract sentence appear, add the NLinear anchor to the grid, lead with the count against the stronger of the two, and move the DLinear comparison and the backbone-contribution analysis into the main text, as Pm4m asked in W1.

On the reframing itself, the scope the AC asks for is the paper's own. The title calls RR-MoA a causal intervention for routing collapse, the abstract reaches the method as the prescription that follows, and two of the three contributions are the diagnosis and R(D). The revision will state the accuracy claim at the scope the controls support, beside them.

**Meta-review point 2 ((M,Σ) and the ablations).** Our earlier response re-scoped Observation 1 as a lower-bound vulnerability predictor, with the [μ,σ]-only and temporal-shuffle ablations populating the denominator of R(D). Pm4m did not pursue it, and listed only the two points above as remaining.

**The mechanism question.** Pm4m found a mismatch between §4.1, which states that non-learnable I/O scaling leaves the (M,Σ) statistics in the hidden states the router reads, and one sentence in our first response describing the opposite. The paper's claim is the correct one and holds throughout, and that sentence is withdrawn. §4.1, Appendix H and Proposition 1 describe one account, and the revision will make that relationship explicit. The submission itself needs no change here.

The diagnosis stands as submitted: normalization-induced routing collapse, the 720-run rescue sweep that leaves 89% of the gap to an input-side fix, and R(D) as a training-free predictor that still calls the Traffic null before training. What changes is the accuracy claim reported beside them.

We appreciate Pm4m raising both points, and the paper is more accurate for them. We can answer anything further during the discussion.
