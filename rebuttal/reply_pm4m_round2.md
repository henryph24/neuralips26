**Response to Reviewer Pm4m**

Thank you for reading the paper and our rebuttal so closely. Putting the two quotations side by side was the right check. We can settle the first here. On the second you named the right control, and we report it below.

**1. Which claim is correct: the paper's.**

§4.1, Table N.2's caption and the regime taxonomy all say the same thing, so on this point the paper needs no revision. The problem is a single sentence in our rebuttal that we should have scoped more carefully. We meant it about what normalization does to its input, but wrote it as a claim about the hidden states, and that is where the contradiction you found comes from.

It may help to explain how that sentence came about, because your question raised a fair point. You asked why raw routing still helps on a backbone that does not collapse, and read the gain as App. P.3's raw-signal predictiveness. We wanted to show it was a routing effect instead. Per-instance scaling subtracts each window's mean and divides by its scale, so the encoder never sees the location-scale, and we carried that straight through to the router. That last move is a claim about the **representation** the encoder builds rather than the **operation**, and it needed measuring.

The answer we should have given was already in the paper, in Table N.2's caption: raw-signal routing is information-richer than hidden-state routing *even when the hidden states retain enough signal to avoid collapse*. Nothing has to go missing for the Moirai gain to be a routing effect. We should have quoted it. We do not stand behind the sentence we wrote instead, or the framing that called the change a strengthening.

We measured it directly. Probing the tensor a hidden-state router reads, the window mean is recoverable from Moirai's hidden states (R²=0.996) but not from MOMENT's under RevIN (−0.311). Switching RevIN off restores it to 0.999, which calibrates the probe. That is the quantity in question, what the router can actually read, and on Moirai the statistics are there, as §4.1 says.

That leaves the substance of our W3 answer unchanged: in RR-MoA the raw signal reaches only the router, and the experts read frozen hidden states. So the gain cannot be App. P.3's Pure Raw-MLP effect, whatever the hidden states hold. The rule we gave Reviewer 8b2Z was shorthand for the same criterion, which needs two conditions together: the statistics removed from what the router reads, and a learnable affine map that can co-adapt (Prop. 1).

**2. The NLinear control, at n=10.**

We built the control you describe: a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer, epochs, batch), differing only by the level anchor, n=10. Our DLinear reproduces the published anchor on all six (ETTh1 0.4187 vs 0.419±0.005), as does Residual-IA+.

The effect you identify is real, and confined to three of nine datasets. NLinear beats DLinear only where the level drifts: Exchange (−28.9%), ETTh2 (−3.8%, −19.0% at H=192) and ETTm2 (−7.9%). On the other six it is neutral or worse, losing 0/10 seeds on Traffic at both horizons.

Residual-IA+ at H=96, n=10: **5/6 against DLinear** (ETTh1 loses) and **5/6 against NLinear** (ETTh2 loses). Swapping in the stronger baseline changes which dataset fails, not how many. Where level anchoring offers no advantage, the backbone's contribution is easier to see: it beats both anchors on Weather (10/10) and Electricity (9/10).

On generalization beyond the six, we re-ran the three datasets never used to build the recipe, n=10 across two horizons, against the stronger anchor. On Solar, where the level anchor gives nothing, it wins all 20 cells (−8.7% at H=96, −11.2% at H=192). On Traffic it is −0.3% and −0.9% across 15/20 cells, and this is the case R(D)=0.14 flags in advance as unlikely to gain.

On Exchange you were right about the cause: the −26.3% over DLinear is level anchoring, not the backbone. Exchange is close to a random walk, where the last value is near the best predictor, so a level anchor is favoured there. Against NLinear it holds parity at H=96 and is behind on the mean at H=192, splitting 5/10 seeds at both, so we will report it as parity rather than a win.

In revision we will score against both anchors throughout and state the collapse criterion as two conditions that must both hold. Both points have made the work more accurate, and we appreciate you raising them.
