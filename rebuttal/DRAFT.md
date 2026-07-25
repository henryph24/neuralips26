# RR-MoA (Submission 16168) — Rebuttal draft

Structure: (0) Common response to AC + all reviewers, organized around the AC's
priority points 3–8 + factual clarifications; (1) Pm4m; (2) jemj; (3) 8b2Z.
Numbers verified against the submitted PDF (= root main.tex) on 2026-07-24.
`[NEW]` = evidence added during rebuttal. `[pending]` = run in progress.

---

## 0. Common Response (Area Chair & all reviewers)

We are grateful to the AC and all three reviewers for the careful reading and constructive
feedback. We are encouraged that the reviews found the problem important and practically
relevant, the causal analysis rigorous, and the training-free signal-ratio of practical
value beyond the method itself. **Following the AC's guidance, we (A) prioritize points
3–8, (B) clarify several factual points — noting where we ourselves could have made the
evidence more prominent — and (C) reframe per points 1–2.** During the
rebuttal we ran several new experiments (input-length; 200-epoch convergence including
Electricity; a TSFM-contribution study at H=192; and a LayerNorm-swap control); every number
below is verified against the submitted PDF.

### A. Priority points 3–8

| # | Concern | Resolution (evidence) |
|---|---------|-----------------------|
| **3** | forecasting/imputation only | `[NEW]` input-length ablation (72 runs, no collapse at any L) + classification mechanism-check; horizons already span 96–720 (Table 14) |
| **4** | requires strictly frozen backbone | RR-MoA also *wins* at last-2 and last-4 (Table 3) — frozen is best, not required |
| **5** | raw routing is memory-heavy | 853-param router, <0.1 ms, +5.9% latency / +6 MB total (Table B.1) |
| **6** | theory limited to IN↔router | scoped by design; App D.1 adds rate-distortion / info-bottleneck; generalizes empirically (pt 8) |
| **7** | expert scaling / memory bandwidth | router input is K-independent; Top-2 keeps active compute O(2); K→10 stable (Table I.2) |
| **8** | causal intervention limited to IN | App H: BatchNorm1d & GroupNorm collapse identically to RevIN; + vision replication (App H.1) |

**Point 3 — The evaluation covers only forecasting (primary) and imputation (secondary).**
*The reviewer is right that our primary evaluation is forecasting and imputation, and that
classification and anomaly detection are the natural next tasks.* What we can show is that
the diagnosis is robust across four axes — task, horizon, input length, and backbone — and
that the anti-collapse mechanism already reaches beyond forecasting: **(Task)** forecasting on
6 datasets and imputation (Table J: 7/8 wins), plus a `[NEW]` **classification
mechanism-check** — on 3 UEA datasets routing entropy stays healthy (1.46–1.59 — near the log 5 ≈ 1.61 uniform maximum, far above the 0.00 collapse floor), so the
anti-collapse phenomenon is not forecasting-specific; we scope this to
mechanism-generalization since frozen features cap absolute accuracy; we are grateful for the
pointer to anomaly detection and will add it in future work. **(Horizon)** H∈{96,192,336,720}: RR-MoA 12/12 vs best
fixed (Table 14). **(Input length)** `[NEW]` L∈{96,192,336,512}, 6 datasets × 3 seeds = 72
runs: routing never collapses (entropy 0.98–1.48 at *every* L), raw routing beats the
normalized-routing control in all 24 (dataset,L) cells, and beats the best fixed adapter
in 70/72. **(Backbone)** six backbones across three normalization regimes. The method and
the mechanism are invariant across all four axes.

**Point 4 — The method requires a strictly frozen backbone.**
*We can see how our presentation gave this impression: we lead with the frozen setting
throughout and prove Proposition 1 for it, so "frozen" reads as a hard requirement. It is not
a requirement, it is simply the best-performing configuration (frozen wins on 4/6 datasets; light unfreezing wins by ≤13% on the other two).* Table 3 shows RR-MoA beating the best fixed adapter at
**Frozen, Last-2, and Last-4** alike (ETTh1 −44% / −29% / −32%), so the method applies
whenever backbone adaptation is genuinely needed. Frozen is also a deployment *feature*,
not just a constraint: one shared backbone serves many tasks via hot-swappable adapters.
Proposition 1 explains *why* frozen is best — a gradient co-adaptation feedback loop — and
a Router-Detached Gradient Flow ablation isolates exactly that mechanism (+35–100%).

**Point 5 — Routing on raw inputs seems memory-consuming.**
*We understand why this looks memory-heavy: the phrase "routing on raw inputs" can suggest
feeding a large tensor to the router. In our design it is the opposite.* The "raw input" is a single
univariate length-L window (not a high-dimensional tensor), which the router pools to 64
features. To calibrate (Table B.1): the router is **853 parameters — 0.002% of the ~40M
backbone, and 57× smaller than the 49K-parameter DLinear** it is benchmarked against —
adding **<0.1 ms to the backbone's 52.9 ms forward pass** and **~0.003% of its FLOPs**.
End-to-end, all three variants stay within **+5.9% latency (56.0 vs 52.9 ms)** and **+6 MB
on a 359 MB backbone (+1.7%)**. Peak memory is set by the frozen backbone every adapter
method already loads, so the extra memory for the router and the experts is negligible next
to it.

**Point 6 — The theoretical analysis is limited to the interaction between instance
normalization and router inputs.**
*This is a fair characterization of the formal theory; we can be clearer about what it
delivers and how we broaden it.* Observation 1 is intentionally narrow — it isolates the
one removable quantity, per-sample location–scale — because that narrowness is exactly what
makes the result *usable before training*: R(D), a **training-free predictor** that says
which datasets will collapse before any training is run (ρ=−0.88), and correctly abstains on
the Traffic null (R=0.14, +2.9%) before any training. We broaden the account in two concrete ways: **(i)** App. D.1 recasts the
same loss through **rate–distortion and the information bottleneck** — a more general
information-theoretic picture than the raw MI argument. The rate–distortion view treats
instance normalization as *lossy compression* that deliberately throws away each window's
mean and scale, and shows that R(D) measures how much of the routing signal sits in exactly
that thrown-away part — so R(D) is a principled estimate of the routing information lost to
normalization, not a hand-built ratio. The information-bottleneck view explains *why* the
collapse happens at all: the normalize-then-encode pipeline is trained to keep only what
helps predict the target and to drop everything else, but "everything else" includes exactly
what the router needs — so collapse is the expected side effect of compressing for one
objective (prediction) while starving another (routing). And **(ii)** empirically the
mechanism holds well beyond the IN↔router pair — BatchNorm1d/GroupNorm and a vision modality
all collapse as predicted (point 8). What we do *not* yet formalize — dataset-specific
distributional structure and feature asymmetries beyond location–scale (raised by 8b2Z) — we
see as genuine open theory, and our own ablations already show the signal that actually
drives routing is broader than (μ,σ) (App. F: SSR +48%).

**Point 7 — How does the approach scale to more experts without a memory-bandwidth
bottleneck?**
*This is a natural concern: in many MoE designs the router input and the memory traffic both
grow with the number of experts, so a bottleneck would be expected. In ours they do not.* The
router reads the raw input window, whose dimensionality is **independent of the expert count K**;
the only K-dependent parameter is the final logit head (64→K plus bias — 325 params at K=5, 650 at
K=10), which is negligible. Because routing is **Top-2**, exactly two experts execute per
sample regardless of K, so **K=10 costs the same per-sample compute and memory traffic as
K=2** (O(2), not O(K)). Empirically, Table I.2 sweeps
K∈{1,2,3,5,7,10} and performance is stable for K≥2 (within ±10%).

**Point 8 — The causal intervention is limited to instance normalization (RevIN).**
*This is a fair reading of the main text, where our headline experiments all use RevIN, so it
is reasonable to read the intervention as RevIN-specific. It is not: the diagnosis and the
intervention already generalize across normalizer families and across modalities.* In App. H
we swap MOMENT's RevIN for **BatchNorm1d** (entropy 0.62→0.004) and
**GroupNorm** (0.51→0.000): both collapse identically — routing entropy runs from log 5 ≈
1.61 (uniform over 5 experts) down to 0.00 (one expert takes everything), so 0.004/0.000
are effectively full collapse — while removing normalization keeps it healthy (0.82). **LayerNorm/RMSNorm used *inside the encoder*** (Chronos, Timer-XL) are
non-collapsing negative controls, exactly as the theory predicts, because they do not strip the
router's input (a LayerNorm swapped into the *input-normalization* position does collapse; `[NEW]`, §3 W4). A vision replication
(App. H.1: ResNet-18 and ViT-B/16) reproduces the mechanism in images. The rule that decides collapse
is *whether the normalizer strips routing-relevant statistics* — not which normalizer it
is.

### B. Clarifying reviewer factual points (per the AC's request)

We are glad to clarify the following. In each case the supporting evidence is already in
the paper — mostly in the appendices — so we take these as a sign that we did not make the
evidence prominent enough, and we will state each one in the main text of the revision:
1. **Raw routing on an existing MoE (8b2Z Q1).** We do exactly this — **AdaMix-Raw**
   (Table `adamix_raw`) swaps only AdaMix's router input, recovering entropy 0.49→1.55
   with −42% to −88% MSE across all 12 cells. This isolates the router *input* as the cause.
2. **BatchNorm / LayerNorm / RMSNorm (8b2Z W4 & Limitations).** As the reviewer notes in
   Q2, our controls already include a BatchNorm swap: App. H shows **BatchNorm1d
   (0.62→0.004) and GroupNorm (0.51→0.000) collapse identically** to RevIN; encoder-internal
   LayerNorm and RMSNorm are our non-collapsing negative controls (Chronos, Timer-XL), while a
   LayerNorm swapped into the *input* position collapses like RevIN (`[NEW]`, §3 W4).
3. **"Router input scales with experts" (8b2Z W3).** The router reads the raw window and
   is **independent of K**; only the K-way logit head grows (64→K), and Top-2 keeps
   *active* compute at O(2).
4. **Image-domain generality (8b2Z Q2).** Already included — a vision replication on
   ResNet-18 and ViT-B/16 (App. H.1).
5. **"13–42% should be over baselines" (Pm4m W6.3).** The referent is over RR-MoA, not the
   baselines (SR-MoA beats RR-MoA on all 6 datasets, Table F.2). Rechecking that table on the
   reviewer's prompt, though, the *range itself* is imprecise: the per-dataset improvements are
   **23–38%** (ETTh1 33%, ETTh2 27%, ETTm1 32%, ETTm2 38%, Weather 23%, Electricity 33%), not
   13–42%. We will correct "13–42%" → "23–38%" everywhere it appears and keep the over-RR-MoA
   referent explicit.
6. **"Almost only horizon 96" (Pm4m Q1).** Horizons span **96–720** (Table 14: RR-MoA
   12/12; Residual-IA⁺ across four horizons).

### C. Corrections we are making
- Router parameter count: the exact figure is **853** — Conv1d(1→16, k=32) [528] +
  Linear(64→5) [325] — consistent across the main text, Fig 2, Table A.1, and the released
  code (`param_count`). "~1.1K" was an early-draft estimate that we later refined to 853 in
  the main text and figures but missed in the checklist; corrected.
- Learnable-α: initialization is **α=σ(0)=0.5** (logit ℓ=0); 0.492 is the first-logged-epoch
  mean, not the initialization (trajectory 0.5 → 0.492 → 0.468).
- Stray parenthesis in R(D) removed; "fine-tuning" standardized.
- Multi-horizon table (Table `residual_ia_plus_multihorizon`): the ETTh2 H=720 entry reads
  −39.6%, but its own reported MSEs (0.849→0.584) and the surrounding prose both give **−31.2%**;
  we cite −31.2% here and will correct the table in the revision.

### D. Points 1–2 (contribution framing and the R(D)/(μ,σ) account)

**Point 1 — reframing the contribution.** We agree the contribution should be reframed, and we
adopt the reviewer's framing. *RR-MoA-proper* is a *diagnostic* intervention and does trail
DLinear in absolute MSE on the smallest/linear-dominated datasets. But the strong form of the
criticism (that the method sits below a trivial linear baseline) does not survive our own
submitted evidence. Our *deployment* method **Residual-IA⁺ outperforms DLinear: 65 significant
wins vs. 11 losses (5.9:1), 107/123 match-or-beat (87%)** of the 6-backbone × 6-dataset ×
4-horizon grid (123 rather than 144 because Timer-XL and Moirai-MoE cover fewer combinations),
with **6/6 at H=192**, **27/27 at H=96 on the five non-MOMENT-small backbones** (ETTh1 included:
MOMENT-large −9.5%, Chronos −11.0%), MOMENT-large **23/24 (zero losses)**, and **100%** on
Timer-XL (12/12) and Moirai-MoE (15/15) (Tables `residual_ia_plus_multihorizon`, `_backbone`,
`_mhcb`). **All of this was already in the submitted paper (App. P).** What we got wrong was the
framing: "match-or-beat" and the section title "Closing the DLinear Gap" foregrounded the one
losing cell (ETTh1 at H=96) that the reviewers understandably anchored on. We will (a) promote
the full grid and the win-count into the main text, (b) retitle to the outperformance the data
show, and (c) state the abstract's DLinear comparison as scope rather than deficit (wording in
the Pm4m W1 response below). The three main contributions stand: the *diagnosis* together with
the training-free R(D) predictor (ρ=−0.88), the *raw-routing intervention*, and the *frozen
shared-backbone deployment regime* (App. B.1), where running a separate DLinear for every tenant
is not practical.

**Point 2 — The (μ,σ)-centric R(D) is in tension with our ablations.** *The reviewer identifies
a real tension, and re-scoping Observation 1 resolves it: R(D) is the **ratio** of
routing-discriminative variance in the instance-norm-**removable** component (location–scale)
to that in the **surviving** component (shape), not a claim that (μ,σ) is the whole routing
signal.* When
that ratio is high, stripping (μ,σ) removes most of the routing signal and routing collapses
(vulnerable); when it is low — Traffic, R=0.14 — the signal lives in the shape that survives
normalization, so stripping does not hurt and RR-MoA correctly does *not* improve (our
falsification case). This is exactly consistent with the ablations showing that the signal
which actually drives routing is broader than (μ,σ): [μ,σ]-only routing degrades +48% (App. F, SSR) and temporal
shuffling moves performance ≤3% (order-invariant, not temporal). We concede Observation 1's
pure-(μ,σ) argument is a **first-order / lower-bound** account — the full collapse also
involves the encoder homogenizing the representation (App.) plus the gradient co-adaptation
of Proposition 1 — and we will re-scope it explicitly as a predictor/lower bound, not a
sufficiency claim.

---

## 1. Response to Reviewer Pm4m

We thank the reviewer for the detailed and fair reading, and for the constructive spirit of
the review. We address every point.

**W1 (framing / DLinear).** This is the reviewer's central point and we treat it as such, but
one correction changes the conclusion. "Absolute forecasting quality below a trivial linear
baseline" is true of **RR-MoA-proper**, the *minimal diagnostic probe* whose only role is to
isolate the router input; it is **not** true of the method we deploy. **Residual-IA⁺, already in
the submitted paper (App. P), does not trail DLinear: it beats it**, and that evidence was in
front of the reviewer at review time:

- **65 significant wins vs. 11 losses (5.9:1)**; **107/123 match-or-beat (87%)** across the
  6-backbone × 6-dataset × 4-horizon grid (Table `residual_ia_plus_mhcb`).
- **6/6 at the H=192 deployment horizon**; **zero losses on MOMENT-large (23/24, 16 sig wins)**;
  **15/15 (100%) on Moirai-MoE**.
- On non-stationary data it *dominates* at long horizons: **ETTh2 −31.2%, ETTm2 −27.0% at H=720**
  (Table `residual_ia_plus_multihorizon`).
- Even at the single least-favorable slice the reviewer cites (MOMENT-small, H=96), the net mean
  gap is **−2.8%, a win**, with ETTh1 the only loss.

We therefore read the criticism as fair about our **framing, not our evidence**. We led with "5/6
match-or-beat" and titled the section "Closing the DLinear Gap," which foregrounded the weakest
cell and invited exactly this reading. We are fixing the presentation, not running to new
experiments to rescue the claim: (a) we promote the full grid and the 65:11 win-count from App. P
into the main text; (b) we retitle away from "gap-closing" to the outperformance the numbers show;
(c) we revise the abstract so the DLinear comparison reads as scope, not deficit. Proposed
wording: *"Because the router isolates the collapse mechanism rather than maximizing accuracy, the
minimal RR-MoA probe trails a from-scratch DLinear on linear-dominated datasets; our deployment
adapter Residual-IA⁺ removes this gap, matching or beating DLinear across six backbones and four
horizons."* We also **accept the narrow, true version** of the point: RR-MoA-proper does trail
DLinear in absolute MSE, sharpest on ETTh1 at H=96, and we will introduce it plainly as a
diagnostic instrument rather than an accuracy claim.

Two scope notes we keep honest. **(i)** Residual-IA⁺ is a larger model than DLinear (hundreds of K
adapter parameters on a frozen backbone vs. 49K), so the comparison is method-level, not
parameter-matched; its shared raw branch is deliberately DLinear-equivalent. **(ii)** The frozen
backbone's marginal contribution is dataset- and horizon-dependent (App. P.3), a separate question
from "do we beat DLinear," which we answer directly in W4 with a controlled experiment showing the
backbone *does* earn its keep.

**W2 (Figure 2).** We will redraw Figure 2 at larger size with non-overlapping modules
and a clearer depiction of the raw-router / frozen-expert paths.

**W3 (Moirai gains without collapse).** A fair distinction, which we can now support with
Moirai's own text. Moirai applies "**(non-learnable) instance normalization (Kim et al.,
2022) … to inputs/outputs**" [Woo et al., 2024] — i.e. it strips per-window location–scale
using the *same* instance-normalization family we study (Kim et al., 2022 is RevIN), minus
the learnable affine part. So raw routing recovers the stripped signal there **by the same
input-side mechanism**, while the *absence* of learnable affine is exactly why AdaMix does
**not** collapse on Moirai: the sudden collapse, which is driven by gradient co-adaptation,
only happens when the norm has a learnable affine part. The unified claim is therefore "raw
routing helps whenever per-window statistics are stripped from the router's input," and
collapse is the sudden, severe failure mode that *additionally* requires learnable affine. We will present the cross-backbone results as
*consistent with the diagnosis* rather than uniform collapse-repair, and add the Moirai
citation.

**W4 (heavy tuning to a linear baseline; generalization beyond the six benchmarks).**

*"Heavy tuning to approach a linear baseline."* The 546-run sweep was development effort to find
the recipe, and the reviewer is right that we should not dress it up as a SOTA search. Two facts
reframe the conclusion. First, the outcome is not "approaching" DLinear, it is beating it (65 sig
wins vs. 11 losses; W1). Second, the recipe is **fixed, not per-dataset tuned**: `[NEW]` applied
**unchanged** (gate-init b=−2, 5-epoch warmup, shared NLinear, val-early-stop) to **three datasets
it had never seen**, Residual-IA⁺ **matches or beats DLinear on 8/9 (dataset, seed) cells**: Solar
3/3 (−9.5%), Traffic 3/3 (−1.5%, parity), Exchange 2/3 (net −19.5%), with no collapse (entropy
0.95–1.51). A recipe that transfers to unseen data without re-tuning is the opposite of
dataset-specific engineering, and it directly answers "its generalization beyond these six
benchmarks is unclear."

*"The raw/NLinear branch does the work; the TSFM is dead weight (App. P.3)."* This is the sharpest
form of the concern, and we answer it head-on, starting with a clarification about P.3 that we
should have made explicit in the paper. P.3's "dead weight" verdict is measured against the
**intermediate Dual-Stream** architecture (ETTh2 0.452, ETTm2 0.252), **not** the final
Residual-IA⁺ (ETTh2 0.346, ETTm2 0.188), and that intermediate architecture itself *loses* to
DLinear. P.3's own backbone-free Raw-MLP MoE in fact **loses to DLinear by +11.2% on all six
datasets** (Table `raw_mlp_moe`, Δ-vs-DL column): remove the TSFM and you fall *behind* DLinear;
it is the TSFM-inclusive Residual-IA⁺ that gets *ahead* of it. `[NEW]` We then ran the clean
isolation the reviewer asks for: Residual-IA⁺ vs. an identical-budget backbone-free Raw-MLP MoE,
holding the raw branch fixed and toggling only the backbone, at **H=192** (where the paper reports
the backbone contributes maximally). **Residual-IA⁺ wins 18/18** (mean −11.8%, range −0.3% to
−34.3%), including ETTh2/ETTm2/Electricity, the very datasets P.3 flagged as dead-weight at H=96,
while still matching or beating a 49K DLinear (seed-mean −6.6% to +1.8%). The corrected, honest
statement is therefore that the frozen backbone's contribution is **horizon-dependent**: small at
H=96 on linear-dominated data, decisively positive at H≥192, and "dead weight" holds only for the
pre-fix architecture at the shortest horizon.

*Generalization of the diagnosis.* Separately from the method, the **diagnosis is already
validated on nine datasets, not six**: beyond the primary six, RR-MoA is evaluated on Exchange
(R=2.17 → **−66.5%**), Solar (R=0.06 → **−32.9%**, the low-R outlier), and Traffic (R=0.14 →
correct **null**, +2.9%) (Table `exchange_solar`), with R(D) correlating with benefit at ρ=−0.88
(n=9). R(D) is an **a-priori** tool, computed from raw statistics before training, so it predicts
whether raw routing will help on any new dataset, Traffic being the predicted-and-observed null (a
falsification, not a failure).

*What we concede.* At H=96 on linear-dominated datasets the backbone adds little over a strong raw
branch, and ETTh1 at H=96 stays DLinear's on MOMENT-small (+2.2%, closing to parity by H=192). The
routing-collapse diagnosis and the training-free R(D) predictor, our core contributions, hold
regardless of which backbone is best on a given dataset.

**W5 ((μ,σ) tension).** See Common Response (point 2): R(D) is a predictor of
vulnerability, not a sufficiency claim; we will scope Observation 1 accordingly.

**W6 (minor).** *Router parameter count:* the exact figure is **853** (Conv1d 528 + Linear
325), reproducible from the released code and matching the main text, Fig 2, and Table A.1;
the Checklist's "~1.1K" was a stale rounding, now corrected. The remaining minors are also
fixed: the stray parenthesis in R(D); the SR-MoA gain is *over RR-MoA* (the intended referent,
not the baselines), and on rechecking Table F.2 we correct its range from "13–42%" to the exact
**23–38%**;
and "finetuning" → "fine-tuning".

**Q1 (input length 512 / horizon 96).** Horizon is already varied over
H∈{96,192,336,720} (12/12 vs best fixed; Residual-IA⁺ 6/6 at H=192). `[NEW]` We add an
input-length ablation (L∈{96,192,336,512}, 6 datasets, 3 seeds): routing never collapses
(entropy 0.98–1.48 at every L — vs a log 5 ≈ 1.61 maximum and a 0.00 collapse floor),
RR-MoA (raw) beats the normalized-routing control at all 24 cells, and beats the best fixed
adapter at 70/72 cells. Conclusions hold across both axes.

**Q2 (α inconsistency).** Thank you — this is a wording slip. The coefficient is
initialized at **α=σ(0)=0.5** (the logit is `nn.Parameter(0.0)`); **0.492 is the mean α
at the end of the first logged epoch**, after one epoch has moved it below 0.5, not the
initialization. Trajectory: 0.5 → 0.492 (epoch 0) → 0.468 (converged). Corrected.

---

## 2. Response to Reviewer jemj

We thank the reviewer for the positive and careful assessment.

**Q1 (classification / anomaly).** `[NEW]` We ran RR-MoA on three UEA classification
datasets (3 seeds): routing entropy stays healthy on all three (1.46–1.59, near the 1.61
maximum, no collapse), so the **anti-collapse mechanism is not forecasting-specific**. Because
frozen-backbone features cap absolute accuracy, we frame this as mechanism-generalization, not
a classification-performance claim. We thank the reviewer for raising anomaly detection and
will extend the study to it in future work.

**Q2 (raw routing + selective unfreezing).** Yes — and encouragingly, this already works. Table 3 shows
RR-MoA (raw routing) wins over the best fixed adapter not only frozen but also with
**Last-2 and Last-4 blocks unfrozen** (ETTh1 −44%/−29%/−32%). Raw routing composes with
partial backbone adaptation; frozen is the strongest configuration on most datasets (4/6; Proposition 1).

**Q3 (H∈{1000,2000}).** We test up to H=720, and within that range the frozen backbone does
*not* become limiting. Our deployment method Residual-IA⁺ still significantly beats DLinear at
H=720 on the non-stationary datasets (ETTh2 −31.2%, ETTm2 −27.0%; Table
`residual_ia_plus_multihorizon`), while DLinear keeps its edge on the short-seasonality
datasets (ETTh1, ETTm1, Electricity), so long-horizon behavior is dataset-dependent rather
than a uniform degradation. The TSFM's contribution is in fact largest at intermediate
horizons (H=192: 6/6 match-or-beat), with the raw branch carrying more of the load as H grows.
Whether the frozen representation becomes limiting *beyond* 720 is genuinely open; we thank the
reviewer and will add the H∈{1000,2000} regime in future work.

**Q4 (training memory, raw + normalized paths).** Training keeps one frozen backbone plus
tiny trainable adapters (354–466K, ~1% of the ~40M backbone) and the 853-param raw router
(0.002% of the backbone); the raw path is a univariate window, so peak training memory is
set by frozen-backbone activations — the same floor as any frozen-backbone adapter method.
At inference the whole stack adds only +5.9% latency and +6 MB (+1.7% of the 359 MB
backbone; Table B.1).

---

## 3. Response to Reviewer 8b2Z

We thank the reviewer for the constructive assessment and for highlighting the practical
value of the signal-ratio metric. Several of the concerns are addressed in the appendices,
which we agree we should have presented more prominently; we clarify each below and will do
so in the revision.

**W1 (memory of raw routing).** *We understand the worry — "routing on raw inputs" sounds
like it should be memory-heavy; in fact it is the opposite.* The router reads a single
univariate window and is an **853-parameter** Conv1d gate —
**0.002% of the ~40M backbone, 57× smaller than 49K-param DLinear** — adding **<0.1 ms** to
the backbone's 52.9 ms forward (**~0.003% of FLOPs**). Total overhead across all three
variants is **+5.9% latency (56.0 vs 52.9 ms)** and **+6 MB on a 359 MB backbone (+1.7%)**
(Table B.1); memory is set by the frozen backbone every adapter shares.

**W2 (dataset-distributional / feature-asymmetry factors).** This is a fair ask. R(D) is in
fact designed as a dataset-distributional quantity — computed from raw dataset statistics,
predicting per-dataset vulnerability (ρ=−0.88), with Traffic (R=0.14) and Solar (R=0.06) as
the boundary/outlier cases — and App. D.1 adds rate-distortion and information-bottleneck
framings toward a broader account (what each framing says is spelled out under Point 6). We do agree that feature asymmetries beyond location–scale
are not yet formalized, and we see formalizing them as a valuable direction for future theory
(see Point 2).

**W3 (scaling experts → router-input dimensionality / bandwidth).** This is a natural worry,
since in a standard MoE the router input does grow with the expert count. Here it does not:
the router input is the raw window and is **independent of K**; only the K-way logit layer (64→K) grows,
which is negligible. Because routing is **Top-2, exactly two experts execute per sample
regardless of K**, so both active compute *and* memory traffic are **O(2), not O(K)** — bandwidth
does not grow with the expert count by construction. Table I.2 (K∈{1..10}) is the empirical
performance-stability check (within ±10% for K≥2); larger learned or dynamically-generated
pools beyond K=10 are future work. So expert count does not create a router-input or bandwidth
bottleneck.

**W4 (other normalizers: BatchNorm/LayerNorm/RMSNorm).** App. H directly tests this:
swapping RevIN for **BatchNorm1d** and **GroupNorm** yields identical collapse
(entropy→0.004 / 0.000 — i.e. one expert, vs the log 5 ≈ 1.61 uniform maximum), while
removing normalization keeps it healthy (0.82). `[NEW]` To match this interventional rigor for
LayerNorm specifically, we also swap in a **LayerNorm at MOMENT's input-normalization position**
(same App H protocol: AdaMix, last-4 unfreeze, ETTh1, seed 42): it **collapses identically
(entropy → 0.000)**, with the no-normalization control healthy at 0.825 (reproducing App H's 0.817). The apparent tension
with our **LayerNorm/RMSNorm negative controls (Chronos, Timer-XL)** is resolved by *where* the
norm sits: at the input position LayerNorm strips per-window statistics and collapses, whereas an
**encoder-internal** LayerNorm/RMSNorm does not strip the router's input and does not collapse.
The rule is *whether the normalizer strips routing-relevant statistics at the router's input*.

**W5 (only forecasting/imputation).** See Common Response (point 3): + input-length
ablation `[NEW]` and a classification mechanism check `[NEW]`.

**W6 / Q3 (few epochs / convergence / 200 epochs).** The complex dual-stream Residual-IA⁺
reaches its validation optimum early (logged early-stop 7–20 epochs) and *overfits* by 50;
the simpler RR-MoA is robust to longer training (below). Extending the *baseline* full-FT
to 50 and 100 epochs (90-config sweep) still loses to 15-epoch frozen RR-MoA by 33–71%
(App. O). `[NEW]` We now train RR-MoA to **200 epochs** on ETTh1, ETTm1, Weather **and
Electricity** (the dataset flagged as slowest to converge; 3 seeds each): routing **never
collapses** (seed-mean entropy 1.02–1.34, minimum 0.87 across all 12 runs, vs a 0.00 collapse
floor), with **Electricity the healthiest at 1.34**. MSE is stable-to-improved versus the
15-epoch numbers (ETTh1 0.680→0.646, ETTm1 0.564→0.493, Weather 0.276→0.271; Electricity 0.314).
Long training therefore neither destabilizes routing nor erodes the advantage: the reported
15-epoch MSEs are conservative, not a convergence artifact.

**W7 (statistical corrections may be too conservative).** The reviewer is right that an
over-conservative correction risks *false negatives*. `[NEW]` We re-analyzed the archived
per-run MSEs and address both directions. **(i) Our claimed effect is correction-immaterial.**
Re-running the *pooled* Wilcoxon signed-rank on the RR-MoA-vs-best-fixed cells gives
**p ≈ 3×10⁻¹¹ uncorrected**, so Bonferroni over every comparison in the paper still leaves it
below 10⁻⁹, and the more powerful Holm/BH only lower it further; no correction hides it. (A
per-*dataset* Wilcoxon with 3–7 seeds is limited by seed count, not effect size, since an n=3
signed-rank floors at p=0.125, which is precisely why the headline significance is the pooled
test plus the 54/54 win-count rather than per-dataset p-values.) **(ii) The boundary null is
genuine.** On Traffic (R=0.14), RR-MoA does *not* beat the best fixed adapter even under the
most powerful *uncorrected* test (+2.9%, **p=0.88**), so the null is a real falsification, not a
strict-correction artifact. Solar (R=0.06) is by contrast not a null but the low-R *outlier*
where RR-MoA improves (−32.9%, Table `exchange_solar`). We will report these uncorrected and
Holm/BH values alongside Bonferroni in the revision.

**Q1 (apply raw routing to existing MoE, e.g. AdaMix).** We did exactly this. **AdaMix-Raw**
(App., Table `adamix_raw`) swaps only AdaMix's hidden-state router for a raw-input router,
leaving its experts and training loop unchanged: routing entropy recovers from near-collapse
to near-uniform (0.49→1.55, of a 1.61 maximum) and MSE improves **42–88%** across all 12 cells. This isolates the **router input** as the
dominant cause, exactly the clarification requested.

**Q2 (generality to image / other domains).** App. H.1 replicates the mechanism in vision:
ViT-B/16 with InstanceNorm1d on patch embeddings collapses (0.000), while ResNet-18 does
not (its InstanceNorm2d strips stats that do not carry the routing signal) — matching the
theory across modality.
