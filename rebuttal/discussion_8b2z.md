# 8b2Z discussion-phase reply kit (NOT the rebuttal; post during discussion)

Use when 8b2Z acknowledges the rebuttal, or the AC asks reviewers to confirm resolution.
Reply within the discussion window: a fast, complete answer is often what converts a
satisfied 4 into a 5. Keep it short and non-argumentative; point to tables, do not re-litigate.

---

## Primary reply (post if 8b2Z indicates concerns are addressed, or stays positive-but-no-score-change)

We thank the reviewer for engaging with our responses. To recap what is now in place: memory and expert-scaling are settled by the 853-parameter, K-independent, Top-2 design (active traffic is O(2)); normalizer generality by the BatchNorm/GroupNorm and [NEW] input-position LayerNorm controls plus the vision replication; task diversity by the [NEW] classification and input-length results; and convergence and statistics by the [NEW] 200-epoch and significance re-analyses. Several of these were already in the submission, and we will surface them in the main text.

Underlying all of them is one general finding: normalization-induced routing collapse is a property of instance-normalized backbones paired with a mixture head, not an LTSF-specific effect (it reproduces under BatchNorm, GroupNorm, and in vision), and R(D) predicts it a priori (ρ=-0.88). Given the concerns are resolved, we would be grateful if the reviewer would consider raising the score to accept, and we are happy to address anything further during the discussion.

---

## If 8b2Z raises a NEW point: open with one line of thanks, then answer from this fact-sheet

- **Memory:** 853-param router, +6 MB on a 359 MB backbone (+1.7%), under 0.1 ms; the frozen backbone is the floor every adapter already loads (Table B.1).
- **Expert scaling:** router input is K-independent; only the 64→K logit head grows; Top-2 keeps compute and memory traffic at O(2); stable across K∈{1..10} (Table I.2).
- **Normalizer generality:** RevIN→BatchNorm 0.62→0.004, GroupNorm 0.51→0.000, [NEW] input-position LayerNorm →0.000; encoder-internal LayerNorm/RMSNorm (Chronos, Timer-XL) do not collapse. Rule: collapse iff the normalizer strips per-window statistics at the router's input.
- **Task diversity:** [NEW] classification entropy 1.46-1.59 (no collapse); imputation 7/8; [NEW] input-length 72 runs, no collapse at any L, raw beats normalized routing at all 24 cells.
- **Convergence:** [NEW] 200 epochs (incl. Electricity, the slowest to converge), entropy 1.02-1.34, MSE stable-to-improved vs 15 epochs.
- **Statistics:** [NEW] pooled Wilcoxon p ≈ 3×10⁻¹¹ uncorrected (Bonferroni still below 10⁻⁹); Traffic null genuine even uncorrected (+2.9%, p=0.88).
- **Raw routing on an existing MoE:** AdaMix-Raw recovers entropy 0.49→1.55, MSE improves 42-88% across all 12 cells, isolating the router input as the cause.
- **Vision:** ViT-B/16 + InstanceNorm1d collapses (0.000); ResNet-18 does not (App. H.1).
- **Significance / generality:** general property of instance-normalized backbones (App. H/H.1); R(D) is training-free and a-priori (ρ=-0.88, n=9) and abstains on the Traffic null.

---

## Tactics
- The 4→5 move for a conf-3 reviewer usually happens in discussion, not the written rebuttal, so be present.
- If the AC asks all reviewers to confirm resolution, that is the moment to post the primary reply.
- Consensus pulls a low-confidence 4 upward: Pm4m moving up and jemj holding at 5 both help 8b2Z. Keep those alive.
- Keep replies short; link to tables; never re-argue a point already conceded or answered.
