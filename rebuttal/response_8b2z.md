**Response to Reviewer 8b2Z**

Thank you for the careful assessment. We are grateful the causal mechanism and the signal-ratio metric came through as valuable. Many of these concerns are already answered by experiments in the paper, and we will highlight them in the revision: memory, expert scaling, normalizers, raw routing on an existing MoE, and a vision replication. Where a concern needed something new, we ran it and tag it [NEW].

**W1 (memory of raw routing).** *(Meta-review point 5.)* We can see why routing on raw inputs sounds memory-heavy. In practice it is the opposite. The router reads one short univariate window (B×L, ~0.26 MB). That is about 64× smaller than the hidden states a normal router reads (B×T×d, ~17 MB at batch 128). And the window is the model's own input, already in memory, so it is a re-read, not a new allocation. The gate itself is a tiny 853-parameter Conv1d, and it adds under 0.1 ms to the 52.9 ms forward pass. Peak memory is set by the frozen backbone that every adapter already loads. The three variants add +5.9% latency and at most +6 MB on a 359 MB backbone (Table B.1). So raw routing adds no meaningful memory, and we will show this in the revision, as the reviewer asked.

**W2 (dataset-distributional / feature-asymmetry factors).** *(Meta-review point 6: theory limited to IN–router interaction.)* We appreciate this. The theory already reaches well beyond the IN–router interaction. Observation 1 is a four-part mutual-information decomposition: an exact identity, a bound, and a data-processing corner where normalization destroys the whole routing distribution (I(S;E)=0). Appendix D.1 adds a rate-distortion reading: instance normalization is lossy compression, and R(D) is its rate-distortion slope. Proposition 2 then pins the phase transition, with an SNR onset at α*=R(D)/(R(D)+1) and a softmax collapse at α=1. Proposition 1 is a separate, dynamical result: gradient co-adaptation drives the entropy collapse over training, confirmed in the full 8-block Transformer. R(D) is also a falsifiable predictor. It predicts vulnerability before training (ρ=-0.88, n=9), is right on the boundary cases (Traffic R=0.14, Solar R=0.06), and rules out a generic SNR account. It even holds beyond instance normalization: under BatchNorm1d, GroupNorm, and in a vision model (App. H, H.1). What is genuinely open is formalizing the dataset-specific structure and feature asymmetries beyond (μ,σ). Our ablations already show the router uses more than (μ,σ) (App. F, SSR +48%).

**W3 (scaling experts → router-input dimensionality / bandwidth).** *(Meta-review point 7.)* The concern is fair for a standard MoE, where the router input grows with the expert count. Ours does not, by design. The router input is a fixed-size raw window, the same at K=5 and K=20. And Top-2 routing runs two experts per sample at any K (2 of 5 at K=5, 2 of 20 at K=20), so active compute and bandwidth do not grow with K. Only the tiny K-way logit layer (64→K) grows. Table I.3 (K∈{1,2,3,5,7,10}) confirms this: routing stays stable for K≥2, so expert count creates no router-input or bandwidth bottleneck. [NEW] We extended the sweep to K=15 and K=20 (ETTh1, 3 seeds). MSE stays stable (0.67-0.73), entropy stays at 91-97% of its log K ceiling (no collapse), and RR-MoA still beats the best fixed adapter by ~40% at every K. We also varied the pool composition (richer conv/residual/gated, deeper MLP, a ten-way pool) and took a first step on learned pools, where a hypernetwork generates each expert. The tables below show routing stays healthy and beats the best fixed adapter. A rigorous learned-pool method (larger generators, NAS) is left to future work.

*Pool composition (ETTh1, H=96, per seed; K=5, large-diverse K=10):*

| pool | seed | MSE | entropy | vs fixed |
|---|---|---|---|---|
| canonical | 42 | 0.646 | 1.452 | -50.3% |
| canonical | 43 | 0.721 | 1.452 | -38.1% |
| canonical | 44 | 0.677 | 1.493 | -42.2% |
| macro | 42 | 0.752 | 1.304 | -42.3% |
| macro | 43 | 0.701 | 1.324 | -47.4% |
| macro | 44 | 0.701 | 1.421 | -34.9% |
| large-diverse | 42 | 0.730 | 2.201 | -41.2% |
| large-diverse | 43 | 0.687 | 2.164 | -29.2% |
| large-diverse | 44 | 0.769 | 2.146 | -43.3% |
| deep-mlp | 42 | 0.893 | 1.292 | -31.0% |
| deep-mlp | 43 | 0.804 | 1.310 | -33.0% |
| deep-mlp | 44 | 0.847 | 1.398 | -31.3% |

*Learned/generated pool: hyper-gen vs canonical (H=96, per seed):*

| dataset | pool | seed | MSE | entropy | vs fixed |
|---|---|---|---|---|---|
| ETTh1 | canonical | 42 | 0.646 | 1.452 | -50.3% |
| ETTh1 | canonical | 43 | 0.721 | 1.452 | -38.1% |
| ETTh1 | canonical | 44 | 0.677 | 1.493 | -42.2% |
| ETTh1 | hyper-gen | 42 | 0.848 | 1.519 | -27.9% |
| ETTh1 | hyper-gen | 43 | 0.876 | 1.528 | -29.7% |
| ETTh1 | hyper-gen | 44 | 0.820 | 1.456 | -19.5% |
| Weather | canonical | 42 | 0.292 | 1.468 | -44.4% |
| Weather | canonical | 43 | 0.251 | 1.435 | -51.6% |
| Weather | canonical | 44 | 0.269 | 1.304 | -51.0% |
| Weather | hyper-gen | 42 | 0.341 | 1.476 | -38.0% |
| Weather | hyper-gen | 43 | 0.309 | 1.562 | -39.6% |
| Weather | hyper-gen | 44 | 0.307 | 1.539 | -40.9% |

**W4 (other normalizers: BatchNorm / LayerNorm / RMSNorm).** *(Meta-review point 8: causal intervention beyond IN.)* This is a reasonable question, and the submitted App. H already tests it. Swapping RevIN for BatchNorm1d (entropy 0.62→0.004) or GroupNorm (0.51→0.000) gives the same collapse, while removing normalization keeps it healthy (0.82). [NEW] For LayerNorm, we swap one into MOMENT's input-normalization position (same App. H protocol). It collapses the same way (entropy → 0.000), with the no-norm control healthy at 0.825. What about our LayerNorm/RMSNorm negative controls (Chronos, Timer-XL)? The difference is where the norm sits. At the input position it strips per-window statistics, so it collapses. Encoder-internal LayerNorm/RMSNorm do not strip the router's input, so they do not. The rule is simple: a normalizer causes collapse when it strips routing-relevant statistics at the router's input. So the mechanism is not IN-specific.

**W5 (only forecasting / imputation).** We already go beyond forecasting on six datasets and imputation (Table J.1, 7/8 wins). [NEW] A classification check on three UEA datasets keeps entropy healthy (1.46-1.59), so the mechanism is not forecasting-specific (frozen features cap accuracy). [NEW] An input-length ablation (L∈{96,192,336,512}, 72 runs) shows no collapse at any L, with raw beating normalized routing at all 24 cells. [NEW] A reconstruction-based anomaly check on SMD adds a fourth task type: entropy stays healthy (1.52-1.61, no collapse) and detection is frozen-capped (ROC-AUC ~0.55). So the mechanism holds across four task types, and generative adaptation is a clear next step.

**W6 / Q3 (few epochs / convergence).** We understand why 15 epochs may seem few. [NEW] So we trained to 200 epochs on the four datasets slowest to converge: ETTh1, ETTm1, Weather, and Electricity (3 seeds each). RR-MoA never collapses (seed-mean entropy 1.02-1.34, min 0.87 across 12 runs), and its MSE is stable-to-improved versus 15 epochs (ETTh1 0.680→0.646, ETTm1 0.564→0.493). So the reported numbers are conservative. Extended full fine-tuning (50-100 epochs) still loses to 15-epoch frozen RR-MoA by 33-71% (App. O). The dual-stream Residual-IA+ overfits by 50 epochs, so we report its 15-epoch result.

**W7 (statistical corrections may be too conservative).** You are right that an over-conservative correction risks false negatives. We address both sides. [NEW] (i) Our effect is correction-immaterial. The pooled Wilcoxon signed-rank on the RR-MoA-vs-best-fixed cells gives p ≈ 3×10⁻¹¹ uncorrected. Bonferroni over every comparison still leaves it below 10⁻⁹, and Holm/BH only lower it further. A per-dataset Wilcoxon with 3-7 seeds is limited by seed count, not effect size (n=3 floors at p=0.125), so our headline is the pooled test plus 54/54, not per-dataset p-values. (ii) The boundary null is genuine. On Traffic (R=0.14), RR-MoA does not beat the best fixed adapter even under the most powerful uncorrected test (+2.9%, p=0.88). We will report uncorrected and Holm/BH alongside Bonferroni.

**Q1 (apply raw routing to an existing MoE, e.g. AdaMix).** We appreciate this suggestion, and it is exactly what our AdaMix-Raw experiment does. It swaps only AdaMix's hidden-state router for a raw-input router, and leaves its experts and training loop unchanged. Routing entropy recovers from near-collapse to near-uniform (0.49→1.55), and MSE improves 42-88% across all 12 cells (Table G.4). This isolates the router input as the dominant cause, which is the clarification requested.

**Q2 (generality to image / other domains).** Yes. App. H.1 replicates the mechanism in vision. A ViT-B/16 with InstanceNorm1d on patch embeddings collapses (0.000), while ResNet-18 does not: its InstanceNorm2d strips statistics that do not carry the routing signal. This matches the theory across domains.

**On significance.** We appreciate that the review recognizes the signal-ratio's value ("predicting dataset vulnerability prior to training... practical value beyond the specific method"). This is what R(D) provides. It is training-free and computed a priori (ρ=-0.88, n=9). So a practitioner can decide before training whether raw routing will help, and where it will not (Traffic, R=0.14). With the collapse diagnosis, that is a reusable result for the TSFM-adapter community, whichever backbone wins. It is also general, not LTSF-specific: it holds under BatchNorm, GroupNorm, and in a vision model (App. H, H.1), and extends to any instance-normalized backbone with a mixture head.

The memory, expert-scaling, normalizer-generality, task-diversity, convergence, and statistical concerns are now addressed, several of them already in the submission. We would be grateful if the reviewer would reconsider the score.
