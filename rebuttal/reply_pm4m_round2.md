**Response to Reviewer Pm4m**

Thank you for reading the paper and our rebuttal so closely. Putting the two quotations side by side was the right check. We can settle the first here. On the second you named the right control, and we report it below.

**1. Which claim is correct: the paper's.**

The paper's claim is correct and has been consistent throughout, as has the rule we gave Reviewer 8b2Z. We can see why the two quotations read as a reversal, though, and the confusion is ours: one sentence in our W3 answer described what normalization does to its input but phrased it as a claim about the hidden states. Beside §4.1 it could hardly read as anything else. The two-regime reading is genuinely sharper than uniform collapse-repair; we should have flagged that we were rephrasing the mechanism alongside it.

How it came about is worth explaining, since your question was a fair one. You asked why raw routing helps on a backbone that does not collapse, and read the gain as App. P.3's raw-signal predictiveness. We wanted to show it was a routing effect instead. Per-instance scaling subtracts each window's mean and divides by its scale, so the encoder never sees the location-scale, and we carried that through to the router. That last move is a claim about the **representation** the encoder builds rather than the **operation**, and it needed measuring.

The account we should have given was already there, in Table N.2's caption: raw-signal routing is information-richer than hidden-state routing *even when the hidden states retain enough signal to avoid collapse*. Nothing has to go missing for the Moirai gain to be a routing effect.

We measured it directly. Probing the tensor a hidden-state router reads, the window mean is recoverable from Moirai's hidden states (R²=0.996) but not MOMENT's under RevIN (−0.311); switching RevIN off restores it to 0.999, calibrating the probe. That is the quantity in question, and on Moirai the statistics are there, as §4.1 says.

That rule was shorthand for the same criterion, and the two classify every case in App. H identically. Stated fully, collapse needs two conditions: the statistics removed from what the router reads, and a learnable affine map that can co-adapt (Prop. 1).

Only that clause fails; the rest holds. There is no collapse on Moirai, so the gain is not collapse-repair. Co-adaptation cannot occur without a learnable affine map. And the gain cannot be App. P.3's Pure Raw-MLP effect, because in RR-MoA the raw signal reaches only the router while experts read frozen hidden states.

§4.1, Table N.2's caption and Proposition 1 are one account: routing is endangered when normalization removes the router's signal and can co-adapt, and raw routing helps where it does not. Moirai meets neither, so it neither collapses nor needs restoring.

**2. The NLinear control, at n=10.**

Both facts you cite are correct: our raw branch is NLinear, and the anchor a single Linear. So we built the control you describe, a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer, epochs, batch), differing only by the level anchor, n=10. Our DLinear reproduces the published anchor on all six (ETTh1 0.4187 vs 0.419±0.005), as does Residual-IA+.

The effect you identify is real and confined to three of nine datasets. NLinear beats DLinear only where the level drifts: Exchange (−28.9%), ETTh2 (−3.8%, −19.0% at H=192) and ETTm2 (−7.9%). On the other six it is neutral or worse, losing 0/10 seeds on Traffic.

Against each anchor at H=96, n=10, Residual-IA+ is **5/6** (ETTh1 loses to DLinear, ETTh2 to NLinear), so swapping the anchor changes which dataset fails, not how many. On Weather and Electricity, where anchoring gives nothing, it beats both on 9 of 10 cells.

Beyond the six, on datasets never used to build the recipe, Solar wins all 20 cells against the stronger anchor (−8.7% at H=96, −11.2% at H=192), and Traffic is −0.3% and −0.9% across 15/20, the case R(D)=0.14 flags in advance as unlikely to gain. On Exchange you were right about the cause: the −26.3% over DLinear is level anchoring, not the backbone, on data close to a random walk where the last value is near the best predictor. Against NLinear it holds parity at H=96 and is behind at H=192, so we will report it as parity rather than a win.

Net, the control separates the two effects rather than merging them: level anchoring accounts for the margin on drifting data, the backbone accounts for it where anchoring is inert. The confound is local to three datasets, not a property of the comparison.

These results are MOMENT-small. The 65:11 count was computed against DLinear alone, so we are extending the same anchor to the other backbones and will score against both throughout the revision, alongside stating the collapse criterion as two conditions. Both points have made the work more accurate, and we appreciate you raising them.
