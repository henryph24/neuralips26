# Draft Abstract

Adapting time series foundation models (TSFMs) to downstream tasks typically involves selecting from a small menu of adapter configurations — LoRA ranks, bottleneck dimensions, prediction heads. We propose searching over arbitrary PyTorch adapter programs instead, treating the adapter architecture as code to be evolved rather than hyperparameters to be tuned. Through systematic experiments on MOMENT across seven standard forecasting benchmarks, we establish three findings. First, evolutionary search in the unbounded code space consistently outperforms discrete configuration search (p=0.031, Cohen's d=0.82), with improvements up to 37.8%. Second, a careful factorial ablation reveals an unexpected pattern: while random template-based code generators win on aggregate MSE (5/7 datasets), adapters containing LLM-discovered architectural patterns — depthwise convolutions, learned feature weighting, batch normalization gating — achieve 19.6% lower MSE when they appear (p=0.003, d=0.67) and require 38% fewer parameters. The aggregate gap is explained entirely by validity: static LLMs waste 25% of evaluations on invalid code, while templates achieve 100% validity. Third, we show that fine-tuning an open-source LLM (Qwen 2.5 3B) on evolutionary discoveries restores 100% code validity while retaining LLM-quality architectural patterns, resolving the validity-quality tradeoff. Our framework completes a full architecture search in 55 minutes at $3.30 per run, discovering novel adapter designs that beat standard adaptation methods by 11-20% and transfer across datasets.

# Draft Introduction Key Points

1. TSFMs (MOMENT, TimesFM, Chronos, Moirai) are powerful but need task-specific adapters
2. Current practice: LoRA, bottleneck, linear probing — all from a FIXED menu
3. This menu can't discover novel architectures (depthwise conv, attention pooling)
4. We propose: search over PyTorch PROGRAMS, not configurations
5. Connection to FunSearch/PartEvo/SOAR — but for FM adaptation, not combinatorics
6. Key question: what drives improvement? Search space? Evolution? LLM guidance?

## Our findings (preview):
- Search space is the dominant factor (code >> discrete, p=0.031)
- LLM discovers architecturally superior adapters (p=0.003, 19.6% better)
- But LLM wastes budget on invalid code → templates win on aggregate
- Self-improving loop fixes validity → unlocks LLM quality advantage
- Discovered architectures import cross-domain patterns (MobileNet → time series)

## Contributions:
1. First code-level adapter search framework for TSFMs
2. Rigorous factorial decomposition with two significant findings
3. Self-improving loop that resolves the validity-quality tradeoff
4. Data-driven adapter design guidelines and novel architectures
