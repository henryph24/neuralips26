**Response to Reviewer Pm4m**

We thank the reviewer for raising the score to 3, and we read it as a fair account of where the paper stood. We agree with both items that held it there.

We should have included both in our last reply. We had the anchor sweep at all four horizons and the probe in hand when we wrote it. We reported the H=96 slice, and we left the probe out.

**1. We have now scored the full grid against both anchors.**

Six backbones, six datasets, four horizons: the same 123 cells the 65:11 count is drawn from. We take Residual-IA+ from the paper's own runs, and both anchors from the n=10 sweep we ran during the rebuttal period, on 29 July. We count a cell as match-or-beat when the mean is lower or the gap is within 2%, the same rule the paper uses.

| Backbone | cells | vs DLinear | vs NLinear | vs stronger, per cell |
|---|---|---|---|---|
| MOMENT-small | 24 | 20 | 18 | 16 |
| MOMENT-large | 24 | 23 | 21 | 21 |
| Moirai | 24 | 11 | 14 | 10 |
| Moirai-MoE | 15 | 15 | 14 | 14 |
| Chronos | 24 | 21 | 18 | 17 |
| Timer-XL | 12 | 12 | 11 | 11 |
| **Pooled** | **123** | **102** | **96** | **89** |

Against NLinear the grid gives **96/123**. By horizon that is 27/33 at H=96, 25/30 at H=192, 25/30 at H=336, and 19/30 at H=720. The claim holds through H=336 and weakens at H=720.

We had disclosed only one long-horizon cell where DLinear and NLinear differ materially, Exchange at H=192, and it was behind. We have now scored all of them. In the 34 cells where the two linear anchors differ by 5% or more, all at H≥192 and concentrated on ETTh2 and ETTm2, we match or beat the stronger of them in 15. Where level anchoring matters most, our margin is smallest. On MOMENT-small, ETTh2 at H=720 moves from −39.6% against DLinear to +17.7% against NLinear. We report the third column only for completeness. It scores every cell against whichever of the two linear models turned out stronger there, and that choice can only be made after seeing the results.

**2. We have measured what §4.1 asserts.**

We accept the methodological point in full. Entropy cannot carry this claim: frozen MOMENT does not collapse either, so non-collapse does not discriminate, and Table N.2's uniform routing is not evidence of specialization.

We ran this probe on 29 July, in the same batch. We take exactly the tensor a hidden-state router consumes, the mean-pooled frozen hidden states, and we ridge-regress the window's own (μ, log σ) out of it. Held-out R², ETTh1, 2,000 train and 1,000 test windows:

| Router input | R² for μ | R² for log σ |
|---|---|---|
| Moirai, non-learnable I/O scaling | 0.996 | 0.961 |
| MOMENT, learnable-affine RevIN | −0.31 | 0.600 |
| MOMENT, RevIN disabled | 0.999 | 0.951 |

We can recover the window mean from Moirai's hidden states. We cannot recover it from MOMENT's, and disabling RevIN inside MOMENT restores it. We read that as the discriminating test the reviewer asked for, since it is a within-backbone intervention that does not rest on collapse. We will report it at §4.1 rather than lean on Table N.2's caption.

**3. Figure 2.** We accept this without reservation, and will redraw it completely, large enough that the raw-router and frozen-expert paths are its subject rather than annotations on it.

On reflection, we think this is the right trade. The accuracy claim gets smaller, because the stronger linear anchor takes cells the old count kept: against NLinear the grid gives 96/123, and the long-horizon ETT cells go the other way. The diagnosis gets better evidence, because §4.1's "preserves" is now a measurement of what survives in the router's input, with RevIN switched off inside MOMENT as the control. We will likely report that as a ninth causal control alongside the eight. That is the paper we meant to write, a diagnosis of routing collapse and a training-free predictor of it, with Residual-IA+ as the intervention that follows. We should have written that far more explicitly, and we did not. We led with "5/6 match-or-beat", we titled the appendix "Closing the DLinear Gap", and we put 107/123 in the implication paragraph. Each of those put the accuracy claim in front of the diagnosis. We should have written it more carefully, and we will in the revision.

With the discussion closing in a few hours, we want to thank the reviewer properly for all three rounds. Both corrections above exist because of them, and the paper is more accurate for each one. We leave the reading to the reviewer, and would be thankful if what is now on the record warrants another look.
