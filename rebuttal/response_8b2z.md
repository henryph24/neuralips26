**Response to Reviewer 8b2Z**

We sincerely thank the reviewer for the careful assessment, and are glad the evaluation was found diverse across the six LTSF benchmarks, the causal mechanism plausible, the intervention simple and targeted, and the signal-ratio valuable beyond the method itself. Several points are already addressed by experiments in the paper, which we will foreground in the revision (memory, expert scaling, other normalizers, raw routing on an existing MoE, a vision replication); where a concern needed something new (a LayerNorm swap, 200-epoch convergence, a significance re-analysis), we ran it and tag it [NEW].

**W1 (memory of raw routing).** We can see why "routing on raw inputs" sounds memory-heavy; in practice it is small. The router reads a single univariate length-L window and is an 853-parameter Conv1d gate (0.002% of the ~40M backbone, 57× smaller than a 49K DLinear), adding under 0.1 ms to the 52.9 ms forward pass. Overhead across all three variants is +5.9% latency and +6 MB on a 359 MB backbone (Table B.1); peak memory is set by the frozen backbone every adapter already loads.

**W2 (dataset-distributional / feature-asymmetry factors).** We appreciate this, and you are right that the formal theory is intentionally narrow. Observation 1 isolates the single removable quantity, per-sample location-scale, and that narrowness is what makes R(D) usable before training: a training-free predictor of which datasets collapse (ρ=-0.88, n=9), which correctly abstains on the boundary cases (Traffic R=0.14, Solar R=0.06). We broaden it two ways. First, Appendix D.1 recasts the same loss through rate-distortion and the information bottleneck. Rate-distortion treats instance normalization as lossy compression that discards each window's mean and scale, so R(D) measures how much routing signal lives in that discarded part, rather than being a hand-built ratio. The information-bottleneck view explains why collapse happens: the normalize-then-encode pipeline keeps only what predicts the target and drops the rest, and "the rest" includes what the router needs. Second, the mechanism holds well beyond instance normalization: it reproduces under BatchNorm1d, GroupNorm, and a vision modality (App. H, H.1, W4). We do not yet formalize dataset-specific distributional structure or feature asymmetries beyond mean and scale; those are genuine open theory, and our ablations already show the router relies on more than (μ,σ) (App. F, SSR +48%).

**W3 (scaling experts → router-input dimensionality / bandwidth).** This is a natural worry, since in a standard MoE the router input grows with the expert count. In ours it does not: the router input is the raw window, independent of K; only the K-way logit layer (64→K) grows, which is negligible. Because routing is Top-2, exactly two experts run per sample regardless of K, so compute and memory traffic stay at O(2), not O(K). Table I.3 (K∈{1,2,3,5,7,10}) confirms performance is stable for K≥2. So expert count creates no router-input or bandwidth bottleneck. [NEW] Extending the published sweep to K=15 and K=20 (ETTh1, 3 seeds), MSE stays stable (0.67-0.73), routing entropy stays at 91-97% of its log K ceiling (no collapse), and RR-MoA still beats the best fixed adapter by ~40% at every K. Beyond pool size, we varied the pool composition (richer conv/residual/gated and deeper MLP experts, a ten-way pool) and took a first step on learned pools (a hypernetwork generates each expert). Both stay healthy and beat the best fixed adapter; the canonical five stay strongest, and a rigorous learned-pool method (larger generators, NAS search) is future work.

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

**W4 (other normalizers: BatchNorm / LayerNorm / RMSNorm).** This is a natural question, and the submitted App. H already tests it: swapping RevIN for BatchNorm1d (entropy 0.62→0.004) and GroupNorm (0.51→0.000) yields identical collapse, while removing normalization keeps it healthy (0.82). [NEW] For LayerNorm, we swap one into MOMENT's input-normalization position (same App. H protocol): it collapses identically (entropy → 0.000), no-normalization control healthy at 0.825. The tension with our LayerNorm/RMSNorm negative controls (Chronos, Timer-XL) comes down to where the norm sits: at the input position it strips per-window statistics and collapses, while encoder-internal LayerNorm/RMSNorm do not strip the router's input, so they do not collapse. The rule is whether the normalizer strips routing-relevant statistics at the router's input, so the mechanism is not specific to instance normalization.

**W5 (only forecasting / imputation).** Beyond forecasting on six datasets and imputation (Table J.1, 7/8 wins), [NEW] a classification check on three UEA datasets keeps routing entropy healthy (1.46-1.59, no collapse), so the mechanism is not forecasting-specific (scoped to mechanism-generalization, since frozen features cap accuracy). [NEW] An input-length ablation (L∈{96,192,336,512}, 72 runs) shows no collapse at any L, raw beating normalized routing at all 24 cells. [NEW] A reconstruction-based anomaly check on SMD adds a fourth task type: entropy stays healthy (1.52-1.61, no collapse), detection frozen-capped (ROC-AUC ~0.55).

**W6 / Q3 (few epochs / convergence).** The complex dual-stream Residual-IA+ overfits by 50 epochs; the simpler RR-MoA handles long training well. Extending full fine-tuning to 50 and 100 epochs (90-config sweep) still loses to 15-epoch frozen RR-MoA by 33-71% (App. O). [NEW] We train RR-MoA to 200 epochs on ETTh1, ETTm1, Weather and Electricity (flagged as slowest to converge; 3 seeds each): routing never collapses (seed-mean entropy 1.02-1.34, min 0.87 across 12 runs), and MSE is stable-to-improved versus 15 epochs (ETTh1 0.680→0.646, ETTm1 0.564→0.493). The 15-epoch numbers are conservative, not a convergence artifact.

**W7 (statistical corrections may be too conservative).** You are right that an over-conservative correction risks false negatives; we address both directions. [NEW] (i) Our claimed effect is correction-immaterial: the pooled Wilcoxon signed-rank on the RR-MoA-vs-best-fixed cells gives p ≈ 3×10⁻¹¹ uncorrected, so Bonferroni over every comparison still leaves it below 10⁻⁹, and Holm/BH only lower it. A per-dataset Wilcoxon with 3-7 seeds is limited by seed count, not effect size: an n=3 signed-rank floors at p=0.125, so our headline is the pooled test plus the 54/54 win-count, not per-dataset p-values. (ii) The boundary null is genuine: on Traffic (R=0.14) RR-MoA does not beat the best fixed adapter even under the most powerful uncorrected test (+2.9%, p=0.88). We will report uncorrected and Holm/BH values alongside Bonferroni.

**Q1 (apply raw routing to an existing MoE, e.g. AdaMix).** We appreciate this suggestion, and it is exactly what our AdaMix-Raw experiment does: it swaps only AdaMix's hidden-state router for a raw-input router, leaving its experts and training loop unchanged. Routing entropy recovers from near-collapse to near-uniform (0.49→1.55) and MSE improves 42-88% across all 12 cells (Table G.4), isolating the router input as the dominant cause, exactly the clarification requested.

**Q2 (generality to image / other domains).** Yes, App. H.1 replicates the mechanism in vision: a ViT-B/16 with InstanceNorm1d on patch embeddings collapses (0.000), while ResNet-18 does not (its InstanceNorm2d strips statistics that do not carry the routing signal), matching the theory across modality.

**On significance.** We are glad the review recognizes the signal-ratio's value ("predicting dataset vulnerability prior to training... practical value beyond the specific method"). This is the paper's most transferable contribution: R(D) is training-free and computed a priori (ρ=-0.88, n=9), so a practitioner can decide before training whether raw routing will help on a new dataset, and it abstains where it will not (Traffic, R=0.14). With the collapse diagnosis it explains, that is a reusable result for the TSFM-adapter community, independent of which backbone wins on a dataset. The diagnosis is also general, not LTSF-specific: it reproduces under BatchNorm, GroupNorm, and a vision modality (App. H, H.1), so it extends to any instance-normalized backbone with a mixture head.

Given that the memory, expert-scaling, normalizer-generality, task-diversity, convergence, and statistical concerns are now addressed, several of them from evidence already in the submission, we would be grateful if the reviewer would consider re-evaluating the score.
