# Architectural Pattern Analysis: LLM vs Template Adapters

## Key Finding

**LLM-discovered architectural patterns produce 13.4% lower MSE** when they appear in winning adapters, even though random templates beat LLM on aggregate (due to validity waste).

## Analysis (from Modal results, MOMENT-large, seed 42)

### LLM-Exclusive Patterns
Patterns that appear in LLM-generated adapters but NOT in random template output:
- **DepthwiseConv** (groups= in Conv1d) — 20% of LLM adapters
- **LearnedParam** (nn.Parameter) — 17% of LLM adapters
- **BatchNorm** — 10% of LLM adapters
- **Residual connections** — 3% of LLM adapters

### MSE Comparison: WITH vs WITHOUT LLM Patterns

| Metric | WITH LLM patterns | WITHOUT | Δ |
|--------|-------------------|---------|---|
| n | 15 | 15 | |
| Mean MSE | **0.4390** | 0.5069 | -13.4% |
| Best MSE | **0.1956** | 0.2239 | -12.6% |

### Per-Dataset Best Adapter Pattern Analysis

| Dataset | Best MSE | Patterns | LLM-exclusive? |
|---------|----------|----------|----------------|
| ETTh1 | 0.5345 | BatchNorm | YES |
| ETTm1 | 0.3660 | LearnedParam, AttentionPool | YES |
| ETTh2 | 0.6714 | basic | no |
| ETTm2 | 0.5419 | basic | no |
| Electricity | 0.1956 | DepthwiseConv, LearnedParam | YES |
| Weather | 0.3258 | LearnedParam, AttentionPool | YES |

**Best adapter uses LLM-exclusive patterns on 4/6 datasets.**

### Interpretation

The LLM's per-adapter quality is genuinely superior. It discovers architectural patterns (depthwise convolution, learned positional weighting, batch normalization before pooling) that simple templates cannot produce. However, the LLM wastes ~25% of its evaluation budget on invalid code (wrong shapes, too many parameters), giving random templates a 33% real-evaluation advantage.

**This resolves the apparent paradox:** random templates win on aggregate MSE because they get more valid evaluations, but the LLM's individual adapter quality is better when it produces valid code.

### Self-Improving LLM Retains LLM Patterns

The fine-tuned Qwen 2.5 3B produces LLM-exclusive patterns (BatchNorm+Conv1d) with 100% validity:
- 4/5 top adapters use Conv1d
- Best adapter uses BatchNorm + Conv1d
- Validity: 100% from generation 1 onward

This suggests the self-improving loop can unlock the LLM's quality advantage while eliminating the validity penalty.
