# Final Paper Outline: Evolving Adapter Programs for TSFMs

## Abstract (~200 words)
- Problem: TSFM adapter design is manual/discrete
- Method: Code-level evolutionary search over PyTorch adapter programs
- Key finding 1: Code space >> discrete (p=0.031, d=0.82, 7/7 datasets)
- Key finding 2: LLM-discovered patterns are significantly better (p=0.003, d=0.67, 19.6%)
- Key finding 3: Validity-quality tradeoff explains when LLM helps vs templates
- Self-improving loop resolves tradeoff: 100% validity + LLM-quality patterns
- Practical: $3.30/run, 55 min, discovers novel architectures (depthwise conv, attention pooling)

## §1 Introduction (~1.5 pages)
- TSFMs need adapters; current approach: discrete config search
- Gap: no code-level architecture search for TSFM adapters
- Our contribution: framework + thorough empirical study + self-improvement
- Preview: code space is key (p=0.031); LLM patterns individually better (p=0.003) but lose on aggregate due to validity; self-improving loop fixes this

## §2 Related Work (~1.5 pages)
- TSFMs: MOMENT, TimesFM, Chronos, Timer-XL, Moirai-MoE
- TSFM adaptation: MSFT (NeurIPS 2025), Prune-then-Finetune (NeurIPS 2025), in-context fine-tuning
- NAS for adapters: NAS-LoRA, Shears, ONE-NAS
- LLM-guided code search: FunSearch, PartEvo (NeurIPS 2025), SOAR (ICML 2025), EvoTune, SEA-TS
- AI Research Agents (NeurIPS 2025 Spotlight): search strategy matters

## §3 Method (~2 pages)
### 3.1 Adapter Interface Contract
- (B, 512, d_model) → (B, H) mapping
- ≤500K params, torch/nn/F/math only
### 3.2 Code-Space Evolutionary Search
- Population of PyTorch nn.Module code strings
- Local validation → GPU evaluation → elitism → offspring generation
- Pluggable code generators: templates, static LLM, self-improving LLM
### 3.3 Self-Improving Loop
- Collect winning adapters from evolution
- QLoRA fine-tune open LLM (Qwen 2.5 3B) on winners
- Evolve with fine-tuned LLM → 100% validity + LLM-quality patterns
- Iterate

## §4 Experiments (~3.5 pages)

### 4.1 Setup
- MOMENT backbone, 7 standard LTSF datasets, H=96, MSE
- 3 seeds (42-44), ~240 evals/run, Modal A10G + RACE VM

### 4.2 Code Space vs Discrete Space (Table 1)
- All code methods >> Evo-3epoch on 7/7 datasets
- p=0.031 (Wilcoxon), d=0.82 (large), mean 16.5% improvement
- Evolved adapters beat manual adaptation (linear probe, MLP) by 11-20%

### 4.3 What Drives Improvement? (Table 2, Figure 3)
Factorial decomposition:
- Search space (code vs discrete): STRONG (7/7)
- Evolution (with vs without): MODERATE (5/7)
- LLM guidance (LLM vs templates): COMPLEX (see §4.4)

### 4.4 The Paradox: LLM Quality vs Template Reliability (Table 3, Figure 4)
**The key section.** Two-part finding:
1. Random templates win on AGGREGATE (5/7 datasets) due to 100% validity
2. But LLM-discovered patterns produce INDIVIDUALLY better adapters:
   - 19.6% lower MSE (p=0.003, d=0.67)
   - 38% fewer parameters
   - Pareto-optimal: best MSE under 100K params are ALL from LLM
   - Cross-domain transfer: depthwise conv from MobileNet, attention from NLP

**The resolution:** The LLM wastes 25% of evals on invalid code → 33% fewer real evaluations. Templates win not on quality but on quantity.

### 4.5 Self-Improving Code Evolution (Table 4)
- Fine-tuned Qwen achieves 100% validity (matches templates)
- Retains LLM-exclusive patterns (BatchNorm+Conv1d)
- Comparison with random on MOMENT-small: [from RACE run]

### 4.6 Additional Results
- **Variance reduction:** 6-7× lower std across seeds (reliability)
- **Multi-horizon:** H=96-336 works; H=720 limited by param budget
- **Cross-dataset transfer:** Adapters transfer across ETT datasets
- **LLM model:** gpt-4o doesn't help (lower validity)
- **Design principles:** Conv1d + attention + BatchNorm + compact = best

### 4.7 Architecture Analysis (Figure 5)
- Discovered patterns: depthwise conv, attention pooling, BatchNorm gating, learned feature weighting
- Cross-domain origin: MobileNet (vision), attention (NLP)
- Data-driven design guidelines for TSFM adapter practitioners

## §5 Discussion (~1 page)
- The search space hypothesis: representation > strategy for structured problems
- LLM as architect vs LLM as optimizer: different modes of value
- When LLM guidance matters: complex datasets, efficiency constraints
- Limitations: single backbone, statistical power for some sub-analyses

## §6 Conclusion (~0.5 page)
- Code-level adapter search is practical and effective for TSFMs
- LLM-discovered patterns are genuinely better but validity is the bottleneck
- Self-improving loop closes the gap
- Framework contribution: pluggable generators, reproducible, $3.30/run

## Key Statistics for Reviewers
- Code >> discrete: **p=0.031, d=0.82**
- LLM patterns >> basic: **p=0.003, d=0.67**
- 7 datasets, 3 seeds, 90 validated adapters analyzed
- 5 ablation conditions + 2 LLM model comparisons
- $3.30/run, 55 min wall time
