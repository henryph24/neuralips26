**Response to Reviewer Pm4m**

We thank the reviewer for reading the paper, its appendices and our rebuttal so closely. We believe it takes a great deal of time and effort to do that. We are sorry for the slight delay: we wanted the requested control finished first. On the first we stand corrected; on the second we report every cell below.

**1. Which claim is correct: the paper's.**

The paper's claim is correct and holds throughout: §4.1, Table N.2's caption and Appendix H agree. One sentence in our W3 answer described what normalization does to its input, but phrased it as a claim about the encoder's representation. We withdraw it. We had reasoned that raw input helps Moirai's router for the reason it does on MOMENT, where RevIN removes the statistics and raw routing restores them. It does not transfer.

Only that clause fails. The diagnosis predicts collapse wherever routing-relevant statistics are stripped from what the router reads. Moirai's routing stays healthy, so those statistics do reach its router, and the gain has the source Table N.2's caption already states: raw-signal routing is information-richer than hidden-state routing even when the hidden states retain enough signal to avoid collapse. The two regimes are therefore collapse-repair where the statistics are stripped, and a richer router input where they are not. This is not App. P.3's Pure Raw-MLP effect: in RR-MoA the raw signal reaches only the router, and the 12–29% figure is RR-MoA against best fixed (Table N.1).

The rule we gave Reviewer 8b2Z is shorthand for the paper's criterion, stated in Appendix H: whether normalization triggers or prevents collapse depends on whether the stripped statistics overlap with the routing signal. The learnable-affine clause in §4.1 separates the normalizers among our backbones, so §4.1, Appendix H and Proposition 1 state one criterion.

**2. The NLinear control: the confound is real, and the result holds.**

We agree on both facts, including that our anchor is the simplified single-Linear variant rather than DLinear's trend-seasonal form, and we built it: a from-scratch NLinear under that anchor's protocol (Linear(512→96), 49K, same optimizer, epochs, batch), differing only by the level anchor, n=10. Our anchor reproduces the published values on all six (ETTh1 0.4187 against 0.419±0.005), as does Residual-IA+.

The effect is real. Across nine datasets at H=96, NLinear beats DLinear on three, all level-drifting: Exchange (−28.9%), ETTh2 (−3.8%), ETTm2 (−7.9%). Elsewhere DLinear is stronger on four and the anchors sit within 0.2% on ETTm1 and Weather, so the substitution cuts both ways.

MOMENT-small, H=96, n=10 per arm, Welch two-sided against the stronger anchor:

| Dataset | Residual-IA+ | DLinear | NLinear | Stronger | Gap | p |
|---|---|---|---|---|---|---|
| ETTh1 | 0.4288 | 0.4187 | 0.4289 | DLinear | +2.4% | 0.008 |
| ETTh2 | 0.3456 | 0.3519 | 0.3385 | NLinear | +2.1% | 0.173 |
| ETTm1 | 0.3259 | 0.3259 | 0.3255 | NLinear | +0.1% | 0.923 |
| ETTm2 | 0.1880 | 0.2109 | 0.1943 | NLinear | −3.2% | 0.075 |
| Weather | 0.1967 | 0.2070 | 0.2068 | NLinear | −4.9% | 0.001 |
| Electricity | 0.1564 | 0.1590 | 0.1627 | DLinear | −1.6% | 0.004 |

Residual-IA+ matches or beats the stronger anchor on four of six, mean gap −0.9%. ETTh1 is the only significant loss; ETTh2 does not reach significance.

The margin holds where level anchoring is inert: on Electricity and Weather above, and on Solar, which the recipe never saw, where the anchors differ by 0.3% and Residual-IA+ wins all 20 cells at n=10 (−8.7% at H=96, −11.2% at H=192). On Exchange we agree: the −26.3% over the single-Linear anchor is level anchoring, and against NLinear the cell is parity at H=96 and behind at H=192; we will report it as parity. On the backbone, the reviewer is right that this comparison cannot separate it from the raw branch; App. P.3 takes that up and reports it as dataset-dependent.

The revision will score against both anchors, lead with the count against the stronger, name the anchor accurately wherever it appears, including Figure P.1's caption, which calls the raw branch DLinear-equivalent, and move the anchor comparison and backbone analysis into the main text.

To be explicit: the confusion here comes from our writing, not the reviewer's reading. Both concerns focus on the accuracy claim rather than the diagnostic contribution because our presentation put it in front: we led with "5/6 match-or-beat" and titled the appendix "Closing the DLinear Gap". The reviewer read the paper as we wrote it. That said, we would respectfully ask that the paper also be weighed on the diagnostic contribution, which stands as submitted; the title and two of three contributions name it.

Again, we thank the reviewer. We know the effort a reading at this depth takes, and both points have made the paper more accurate. If the reviewer finds them met, we would truly appreciate a reassessment of the score.
