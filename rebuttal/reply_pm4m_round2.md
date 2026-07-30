**Response to Reviewer Pm4m**

We thank the reviewer for reading the paper and our rebuttal so closely. Putting the two quotations side by side was the right check. We settle the first here, and report the control named in the second below.

**1. Which claim is correct: the paper's.**

We clarify that the paper's claim is correct, and that it has been consistent throughout the submission, as has the rule we gave Reviewer 8b2Z. We can see why the two quotations read as a reversal, and the confusion is ours. One sentence in our W3 answer described what normalization does to its input, but phrased it as a claim about the representation the encoder builds. Beside §4.1 it could hardly read otherwise, and we withdraw it. The question behind it was a fair one, and in answering it we carried that argument through to the router without measuring whether the statistics survived there.

So we measured it. Probing the tensor a hidden-state router reads, the window mean is recoverable from Moirai's hidden states (R²=0.996, and log σ at 0.961) and not from MOMENT's under RevIN (−0.311 and 0.600). Switching RevIN off restores them to 0.999 and 0.951, calibrating the probe. The statistics are present on Moirai, as §4.1 says.

Only that clause fails, and the rest holds. There is no collapse on Moirai, so the gain is not collapse-repair. Nor can it be App. P.3's Pure Raw-MLP effect, because in RR-MoA the raw signal reaches only the router while the experts read frozen hidden states. The 12–29% figure compares RR-MoA with the best fixed adapter (Table N.1), so it measures what routing buys where routing is possible, and nothing has to go missing for it. That is the account the paper already offers: raw-signal routing is information-richer than hidden-state routing even where the hidden states retain enough signal to avoid collapse.

On the rule we gave Reviewer 8b2Z, it is shorthand for the paper's own criterion rather than a replacement. Appendix H states that criterion directly: whether normalization causes or prevents collapse depends on whether the stripped statistics overlap with the routing signal. The learnable-affine clause in §4.1 is what separates the normalizers among the backbones we compare. On Moirai the statistics reach the router and the backbone is strictly frozen, so §4.1, Appendix H and Proposition 1 have always been one account rather than three.

**2. The NLinear control: NLinear is the stronger anchor on three of nine datasets.**

We agree with the reviewer's reading on both facts cited, so we built exactly that control: a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer, epochs and batch), differing only by the level anchor, at n=10. Our DLinear reproduces the published anchor on all six datasets (ETTh1 0.4187 against 0.419±0.005), as does Residual-IA+, so the comparison runs on the paper's own footing.

The effect identified is one part of the picture, and we can now say how far it reaches. Across nine datasets at n=10, NLinear beats DLinear on three, all of them level-drifting: Exchange (−28.9%), ETTh2 (−3.8%, and −19.0% at H=192) and ETTm2 (−7.9%). On the remaining six it is neutral or worse, winning 0/10 seeds on Traffic at both horizons and 2/10 on Electricity. DLinear is the stronger anchor more often, so the substitution does not run one way.

At H=96, n=10, Residual-IA+ beats both anchors on ETTm2 (−3.2%), Electricity (−1.6%) and Weather (−4.9%), and is at parity with both on ETTm1, within 0.1% either way. Of the remaining two, ETTh1 is the loss against DLinear and ETTh2 against NLinear, so swapping the anchor changes which dataset fails rather than how many.

Beyond the six, on datasets never used to build the recipe, Solar wins all 20 cells against the stronger anchor (−8.7% at H=96, −11.2% at H=192), and Traffic is at −0.3% and −0.9% across 15 of 20, which R(D)=0.14 predicts in advance. On Exchange we agree with the reviewer about the cause. The −26.3% over DLinear is level anchoring rather than the backbone, on data close to a random walk. Against NLinear it holds parity at H=96 and is behind at H=192, so we will report it as parity rather than a win.

So the control does what was asked of it, and the two effects separate. Level anchoring accounts for most of the margin on Exchange and ETTh2, and for much of it on ETTm2; where anchoring is inert the margin holds against the stronger anchor. On the backbone's own contribution, the grid was never the evidence for it, and App. P.3 addresses that question directly, reporting the contribution as dataset-dependent.

These results are MOMENT-small, and the 65:11 count was computed against DLinear alone. We will score against both anchors throughout the revision, lead with the count against the stronger of the two, and make the criterion relationship above explicit. Both points have made the work more accurate, and we appreciate the reviewer raising them.
