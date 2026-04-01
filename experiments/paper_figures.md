# Proposed Paper Figures

## Figure 1: Framework Overview
Pipeline diagram: LLM/Template → Code → Validate → GPU Eval → Evolution → Best Adapter
Show the code-space search loop with pluggable generators.

## Figure 2: Code Space vs Discrete Space (Bar Chart)
Bar chart: 7 datasets, 2 bars each (best code method vs Evo-3epoch).
Shows consistent improvement across all datasets.
Data: Table 1 from results.

## Figure 3: The Validity-Quality Tradeoff (Scatter)
X-axis: validity rate, Y-axis: best MSE per method
Points: Code Evo (75%, 0.5345), Random+Evo (100%, 0.4775), LLM No Evo (40%, 0.5575)
Arrow showing: "templates win on aggregate but LLM wins per-adapter"
Annotate LLM-exclusive patterns on the best individual adapters.

## Figure 4: Pareto Front — MSE vs Parameter Count
Scatter of all validated adapters (30 Code Evo + 35 Random + 35 LLM No Evo)
Color by method. Show Pareto front.
LLM adapters dominate the efficient frontier (low params + low MSE).
Key: 51K-param depthwise conv adapter at MSE=0.1956.

## Figure 5: Discovered Architecture Examples
Side-by-side: 3 novel adapter architectures discovered by Code Evolution
- Conv1d + BatchNorm (ETTh1) — vision-style feature extraction
- Depthwise Conv + Learned Weights (Electricity) — MobileNet-inspired
- Attention Feature Selection (Weather) — soft channel weighting
Show code + architecture diagram + MSE.

## Figure 6: LLM Pattern Analysis
Grouped bar: percentage of adapters using each pattern, by method
(Conv1d, AttentionPool, DepthwiseConv, BatchNorm, LearnedParam, etc.)
Highlight LLM-exclusive patterns.

## Figure 7: Variance Reduction Across Seeds
Box plot: 3 seeds × 5 datasets
Code Evo vs Evo-3epoch
Shows 6-7× tighter spread for Code Evo.

## Figure 8: Self-Improving Loop
Line plot: validity rate and MSE across self-improvement iterations
Iter 0 (static LLM): 75% validity
Iter 1 (fine-tuned): 100% validity
Show progression.

## Table 1: Main Results (7 datasets)
Code Evo vs Random+Evo vs LLM No Evo vs Evo-3epoch

## Table 2: Factorial Decomposition (2×2)
Evolution × Generator design

## Table 3: Design Principles
Which architectural patterns correlate with lower MSE
(Conv1d ↓, AttentionPool ↓, BatchNorm ↓, Deep MLP ↑, Last Token ↑)

## Table 4: Adaptation Method Comparison
Linear probe vs MLP vs evolved adapter (direct MSE comparison)
