**Response to Reviewer Pm4m**

We thank the reviewer for reading the paper and our rebuttal so closely. Putting the two quotations side by side was the right check. We settle the first below, and report the control after it.

**1. Which claim is correct: the paper's.**

We clarify that the paper's claim is correct, and that it has been consistent throughout the paper: §4.1, Table N.2's caption and Appendix H all say the same thing. On our own W3 answer we stand corrected. One sentence there described what normalization does to its input, but phrased it as a claim about the representation the encoder builds, and beside §4.1 it could hardly read otherwise. We withdraw it.

How it came about is worth saying, since the question was fair. We wanted to show the Moirai gain was a routing effect rather than App. P.3's raw-signal predictiveness, and for that we needed a reason the raw input helps the router there. We reached for the reason that holds on MOMENT, where RevIN removes the statistics and raw routing restores them, and applied it to Moirai without asking whether it transferred. It does not. The two-regime reading is sharper than uniform collapse-repair, but the clause carrying it rewrote the mechanism, and we should have said so then.

Only that clause fails. If the statistics were stripped from what the router reads on Moirai, the diagnosis would predict collapse there. Moirai does not collapse. So the gain is not collapse-repair. It is also not App. P.3's Pure Raw-MLP effect. In RR-MoA the raw signal reaches only the router, and the experts read frozen hidden states. The 12–29% figure compares RR-MoA with the best fixed adapter (Table N.1). So it measures the value of routing itself, not the repair of a broken router.

On the rule we gave Reviewer 8b2Z, it is shorthand for the paper's own criterion rather than a replacement. Appendix H states that criterion directly: whether normalization causes or prevents collapse depends on whether the stripped statistics overlap with the routing signal. The learnable-affine clause in §4.1 is what separates the normalizers among the backbones we compare. On Moirai the statistics reach the router and the backbone is strictly frozen, so §4.1, Appendix H and Proposition 1 are one account rather than three.

**2. The anchor control: NLinear beats DLinear on three of nine datasets, and DLinear beats NLinear on three.**

We agree with the reviewer's reading on both facts cited, so we built exactly that control: a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer, epochs and batch), differing only by the level anchor, at n=10. Our DLinear reproduces the published anchor on all six (ETTh1 0.4187 against 0.419±0.005), as does Residual-IA+, so the comparison runs on the paper's footing.

The effect identified is one part of the picture, and we can say how far it goes. Across nine datasets at n=10, NLinear beats DLinear on three, all level-drifting: Exchange (−28.9%), ETTh2 (−3.8%, and −19.0% at H=192) and ETTm2 (−7.9%). On the other six it is neutral or worse, so the substitution does not run one way.

At H=96, n=10, Residual-IA+ beats both anchors on ETTm2 (−3.2%), Electricity (−1.6%) and Weather (−4.9%), and is at parity with both on ETTm1, within 0.1%. Of the other two, ETTh1 loses to DLinear and ETTh2 to NLinear, so swapping the anchor changes which dataset fails rather than how many.

Beyond the six, on unseen data, Solar wins all 20 cells against the stronger anchor (−8.7% at H=96, −11.2% at H=192), and Traffic is at −0.3% and −0.9% across 15 of 20, which R(D)=0.14 predicts in advance. On Exchange we agree about the cause: the −26.3% over DLinear is level anchoring rather than the backbone. Against NLinear it holds parity at H=96 and is behind at H=192, so we will report it as parity rather than a win.

The control does what was asked, and the two effects separate. Level anchoring accounts for most of the margin on Exchange and ETTh2, and for much of it on ETTm2; where anchoring is inert the margin holds against the stronger anchor. On the backbone's own contribution, the grid was never the evidence for it, and App. P.3 addresses that question directly, reporting the contribution as dataset-dependent.

These results are MOMENT-small, and the 65:11 count was computed against DLinear alone. We will score against both anchors in the revision, lead with the count against the stronger, and make the criterion relationship explicit.

Both points bear on the accuracy claim reported beside the diagnosis, not on the diagnosis itself. R(D) is the part we would most want read: computed from the data before any training, it predicts a negative as readily as a positive. At R=0.14 it says Traffic should not benefit, and Traffic does not, which is why the parity above is reported rather than worked around. Both readings have made the work more accurate, and we appreciate them.
