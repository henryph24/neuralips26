**Response to Reviewer Pm4m**

We thank the reviewer for the time this review took. Both remaining points came from reading our rebuttal against the paper's own appendices, and both were right: the first caught a sentence of ours that contradicts §4.1, and the second identified a confound that our own architecture creates, since Residual-IA+'s raw branch is NLinear. We are grateful too for the pass back through the minor issues to close them. On the first point we stand corrected, and offer our reasoning; on the second we took the recommendation, ran the experiment and report the results.

**1. Which claim is correct: the paper's.**

We can see why the two quotations read as a reversal. We clarify conclusively that the paper's claim is the correct one and holds throughout: §4.1, Table N.2's caption and Appendix H agree. One sentence in our W3 answer described what normalization does to its input, but phrased it as a claim about the encoder's representation, and beside §4.1 it could hardly read otherwise. We withdraw it, and want to show how we got there. Our reasoning was that the raw input helps the router on Moirai for the reason it does on MOMENT, where RevIN removes the statistics and raw routing restores them, and we applied it without asking whether it transferred. It does not. However, the two-regime reading is genuinely sharper than uniform collapse-repair, and we restate it correctly below; we should have flagged that the clause carrying it rewrote the mechanism.

Only that clause fails. The diagnosis predicts collapse wherever routing-relevant statistics are stripped from what the router reads; Moirai's routing stays healthy, so they do reach its router and the gain has another source. App. P.3's Pure Raw-MLP effect is a different mechanism, where the experts read the raw signal directly; in RR-MoA it reaches only the router, and the experts read frozen hidden states. The 12–29% figure compares RR-MoA with the best fixed adapter (Table N.1), so it measures what a raw-routed mixture buys over a single one.

The rule we gave Reviewer 8b2Z is shorthand for the paper's own criterion, stated directly in Appendix H: whether normalization causes or prevents collapse depends on whether the stripped statistics overlap with the routing signal. The learnable-affine clause in §4.1 separates the normalizers among the backbones we compare, so §4.1, Appendix H and Proposition 1 describe one account.

**2. The NLinear control: the confound is real, and Residual-IA+ still wins.**

We agree with the reviewer on both facts, and built the control: a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer, epochs and batch), differing only by the level anchor, at n=10. Our DLinear reproduces the published anchor on all six (ETTh1 0.4187 against 0.419±0.005), as does Residual-IA+.

The effect is real. Across nine datasets at H=96, n=10, NLinear beats DLinear on three, all level-drifting: Exchange (−28.9%), ETTh2 (−3.8%) and ETTm2 (−7.9%). On the other six DLinear is the equal or stronger anchor, so the substitution cuts both ways.

At H=96, n=10, Residual-IA+ matches or beats the stronger anchor on four of the six: ETTm2 (−3.2%), Electricity (−1.6%) and Weather (−4.9%), parity on ETTm1 within 0.1%; ETTh1 loses to DLinear, ETTh2 to NLinear.

Beyond the six, on data the recipe never saw, Solar wins all 20 cells against the stronger anchor (−8.7% at H=96, −11.2% at H=192), the R(D) outlier at R=0.06. Traffic sits at −0.3% and −0.9%, winning 15 of 20, as R(D)=0.14 predicts. On Exchange we agree: the −26.3% over DLinear is level anchoring, and against NLinear the cell is parity at H=96 and behind at H=192, so we will report it as parity.

The two effects separate: level anchoring accounts for most of the margin on Exchange and ETTh2 and much on ETTm2, and the margin holds where anchoring is inert, on Electricity, Weather and Solar. On the backbone's contribution, the reviewer is right that the grid cannot settle it; App. P.3 takes it up and reports it as dataset-dependent.

These results are MOMENT-small, and the 65:11 count used DLinear alone. The revision will score against both anchors, lead with the count against the stronger, move the DLinear comparison and backbone-contribution analysis into the main text, and make the criterion relationship explicit.

We want to be explicit that the fault here is our writing, not the reviewer's reading. Both concerns focus on the accuracy claim rather than on the paper's intended diagnostic contribution because our own presentation put that claim in front: we led with "5/6 match-or-beat" and titled the appendix "Closing the DLinear Gap". The reviewer read the paper as we wrote it. With that said, we would respectfully ask that the paper also be weighed on the strength of the diagnostic contribution, which stands as submitted. That is the paper's own scope: the title calls RR-MoA a causal intervention for routing collapse, and two of its three contributions are the diagnosis and R(D). We name a failure mode the standard MoE toolkit does not repair: over 720 runs the best rescue cuts MSE by 10.9%, 2.7× worse than RR-MoA. We see collapse for a ViT under InstanceNorm1d and none for a ResNet under InstanceNorm2d, as the criterion predicts. We compute R(D) before training, and it says where the fix will help and where it will not.

We have answered both points, reporting every cell including those against us. Again, we thank the reviewer. We know the effort a reading at this depth takes, and both have made the paper more accurate. If the reviewer finds them met, we would truly appreciate a reassessment of the score, and can run anything further.
