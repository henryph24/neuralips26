**Response to Reviewer Pm4m**

We thank the reviewer for a second careful reading. Setting the two quotations side by side was the right check, and both points are well taken. We settle the first below; the second names the right control for the anchor question, which we have run.

**1. Which claim is correct: the paper's.**

We clarify that the paper's claim is correct and consistent on this point, as is the rule we gave Reviewer 8b2Z. We can see why the two read as a reversal, though, and the confusion is ours. One clause in our W3 answer described what normalization does to its **input**, but phrased it as a claim about the **representation** the encoder builds. Next to §4.1 it could hardly read otherwise.

We should say how it happened. The reviewer asked why raw routing helps on a backbone that does not collapse, reading the gain as App. P.3's raw-signal predictiveness. We wanted to show it was a routing effect instead. Per-instance scaling subtracts each window's mean and divides by its scale, so the encoder never sees the location-scale, and we assumed the same for the router without checking. The two-regime reading is genuinely sharper, but we should have said clearly that we were also rewording the mechanism.

We have since measured it, taking the tensor a hidden-state router reads and recovering each window's mean. On Moirai the mean is recoverable (R²=0.996); on MOMENT under RevIN it is not (−0.311). Switching RevIN off restores it to 0.999, showing the probe works. The statistics are present in what the router reads, which is §4.1's claim. What we measured is decodability there, not the scaling mechanism. The account we should have given was already in Table N.2's caption: raw-signal routing is information-richer than hidden-state routing *even when the hidden states retain enough signal to avoid collapse*. Nothing need go missing for the gain to be a routing effect.

That clause aside, the rest of our W3 answer stands. There is no collapse on Moirai, so the gain is not collapse-repair. Nor can it be App. P.3's Pure Raw-MLP effect: in RR-MoA the raw signal reaches only the router, while experts read frozen hidden states. The rule we gave 8b2Z was shorthand for the same criterion; both give the same verdict on every App. H case: collapse needs the statistics removed from what the router reads **and** a learnable affine map that can co-adapt (Prop. 1). Moirai has neither, so §4.1, Table N.2's caption and Prop. 1 are one account, not three.

**2. The NLinear control: NLinear is the stronger anchor on three of nine datasets.**

We are convinced that both facts cited are correct, and that the proposed control is the right test of the anchor effect, so we have run it: a from-scratch NLinear under the DLinear anchor's protocol (Linear(512→96), 49K, same optimizer and schedule), differing only by the level anchor, n=10. Our DLinear reproduces the published anchor on all six (ETTh1 0.4187 vs 0.419±0.005), as does Residual-IA+, so the comparison uses the paper's baseline. The control does separate the anchor effect from the rest: level anchoring explains the margin where the level drifts, and elsewhere the margin survives the stronger anchor.

We agree the effect is real, and can say how far it reaches. Across nine datasets at n=10, NLinear beats DLinear on three, all with drifting levels: Exchange (−28.9%), ETTh2 (−3.8%, −19.0% at H=192) and ETTm2 (−7.9%). On the other six it is equal or worse, losing 0/10 seeds on Traffic. DLinear is more often the stronger anchor, so changing it does not favour one side consistently.

At H=96, n=10, Residual-IA+ is **5/6** against each anchor, ETTm1 within 0.1% of NLinear (ETTh1 loses to DLinear, ETTh2 to NLinear): changing the anchor changes which dataset fails, not how many. On Weather and Electricity, where the anchor does not help, it beats both on 9 of 10.

Beyond the six, on datasets never used to build the recipe, Solar wins 20 of 20 against the stronger anchor (−8.7% and −11.2%), and Traffic 15/20 at −0.3% and −0.9%, as R(D)=0.14 predicts. We agree with the reviewer on Exchange: the −26.3% over DLinear is accounted for by level anchoring alone, on data close to a random walk where the last value is near the best predictor. Against NLinear it is at parity at H=96 and behind at H=192, so we now read it as parity.

Beating NLinear does not by itself isolate the backbone from routing or the residual path. A backbone-free Raw-MLP MoE does lose to Residual-IA+ on 18 of 18 cells at H=192 (mean −11.8%), on the cells App. P.3 called negligible, but its raw branch is an MLP rather than our NLinear, so it does not isolate the backbone either; our first response called it a backbone toggle, which was imprecise. A control varying only that path would, and that is future work.

These results are MOMENT-small, and the 65:11 count is a DLinear comparison. We are extending the anchor to the other backbones, a good addition we owe to this exchange.
