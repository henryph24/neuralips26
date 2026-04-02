# NeurIPS 2026 Paper TODO

**Deadline: May 4, 2026 (abstract) / May 6, 2026 (full paper)**
**Status: 21/21 wins, 4 backbones, 72 refs, draft complete**

---

## Experiments

### Running on RACE VM
- [ ] ETTh1 H=336 (in progress)
- [ ] ETTm1 H=336
- [ ] ETTh1 H=720
- [ ] ETTm1 H=720
- [ ] Electricity H=96

### After current batch
- [ ] Update paper Table 1 with H=336, H=720 results
- [ ] Update paper with Electricity results
- [ ] Update abstract claim count (21 → 26+ experiments)

### Nice-to-have (if time)
- [ ] ETTm2, Weather multi-seed (seeds 43, 44)
- [ ] H=336/720 on ETTh2, Weather
- [ ] Electricity on MOMENT-large

---

## Paper Writing

### Framing (CRITICAL)
- [ ] Reframe: sell the FINDING not the METHOD
  - Finding: "adapter architecture is an overlooked design dimension that matters 11%"
  - Method is simple (random search + evolution) — that's a feature, not a bug
  - Message: "even simple search discovers consistent improvements — the design space is far richer than exploited"
- [ ] Don't oversell LLM contribution — LLM is part of ablation story, not the winning method
- [ ] Address reviewer concern: "isn't this just hyperparameter tuning?"
  - No: we search computational graphs (Conv1d+BN, depthwise conv), not scalar configs
  - The winning architectures literally cannot be expressed as hyperparameter combinations

### Sections to polish
- [ ] Abstract: tighten to emphasize finding over method
- [ ] Introduction: the 3 reasons why nobody searched adapter architecture — already written, verify flow
- [ ] Related work: adapter architecture review — already written, verify completeness
- [ ] Method: clarify template generator is one instantiation, not the core contribution
- [ ] Results: add H=336/720 rows when available
- [ ] Discussion: add "why simple search works" paragraph
- [ ] Conclusion: end with impact statement about changing practitioner behavior

### Tables and Figures
- [ ] Table 1: expand with H=336, H=720, Electricity
- [ ] Table 2: cross-backbone table is complete (4 backbones)
- [ ] Figure 2: update bar chart if adding more datasets
- [ ] Consider: convergence curve figure (MSE vs generation)
- [ ] Consider: adapter code example in appendix

### Citations
- [x] 72 references — sufficient for NeurIPS
- [x] All from rigorous academic sources
- [x] Adapter architecture gap justified with 3 reasons + FM Struggle paper
- [ ] Final pass: check all citations render correctly in compiled PDF

---

## Reviewer Anticipation

### Expected questions and our answers
1. **"Isn't this just hyperparameter tuning?"**
   → No. We search computational graphs. Depthwise conv + learned positional weights cannot be expressed as a hyperparameter combination.

2. **"The template vocabulary is hand-designed with LLM-discovered patterns — circular?"**
   → The contribution is proving adapter architecture matters (21/21, p=0.031). The template generator is one instantiation. Even naive templates (mean/max/last + MLP) beat discrete search.

3. **"Why not compare to published PatchTST/iTransformer numbers?"**
   → Different paradigm. We compare within the frozen-backbone adapter paradigm. Our MSEs are higher because we freeze the backbone — but evolved adapters beat fixed adapters consistently.

4. **"Only one primary backbone (MOMENT-small) for full horizon coverage"**
   → Standard practice (MSFT also does this). We additionally show H=96 wins on 4 backbones for generalization.

5. **"The method is simple — random search + evolution"**
   → Simplicity is a strength. 30 min, $0, single GPU. The finding that simple search yields 11% improvement proves the design space is undertapped.

6. **"Statistical significance with only 3 seeds?"**
   → p=0.031 (code vs discrete, Wilcoxon), p=0.003 (LLM patterns, Mann-Whitney). Both significant despite modest sample size.

---

## Pre-submission Checklist
- [ ] NeurIPS paper checklist (checklist.tex) — fill in all items
- [ ] Anonymize: remove author names, institution, RACE VM references
- [ ] Code: prepare clean release repo
- [ ] Supplementary: full adapter code listings, additional tables
- [ ] LaTeX: compile on Overleaf, fix any warnings
- [ ] Proofread: full pass for clarity, grammar, consistency
- [ ] Abstract: ≤1500 characters for OpenReview submission
