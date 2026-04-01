# FINAL RESULTS: Per-Dataset Code Evolution on Standard LTSF Protocol

## 4/4 Clean Sweep — Evolved Adapters Beat Hand-Designed Baselines

| Dataset | Evolved MSE | Best Baseline | Δ | Evolved Params | Baseline Params |
|---------|-------------|--------------|---|---------------|----------------|
| ETTh1 | **1.0835** | 1.1477 (conv) | **-5.6%** | 118K | 209K |
| ETTm1 | **0.9920** | 1.1223 (conv) | **-11.6%** | 50K | 209K |
| ETTh2 | **2.7353** | 3.1040 (conv) | **-11.9%** | 79K | 209K |
| ETTm2 | **2.8284** | 3.1059 (conv) | **-8.9%** | 104K | 209K |
| **Average** | | | **-9.5%** | **88K** | **209K** |

## Setup
- Backbone: MOMENT-small (d_model=512, 8 blocks, last 4 unfrozen)
- Protocol: Standard chronological splits (ETTh: 8640/2880/2880, ETTm: 34560/11520/11520)
- Normalization: StandardScaler fit on train only
- Evolution: 12 generations × 20 population, random code templates, seed 42
- Fitness: 3-epoch val MSE during evolution, 15-epoch test MSE for final evaluation
- Compute: RACE VM A10G, ~30 min per dataset, $0 cost

## Key: Proper Experimental Design Matters
Earlier experiments tested adapters evolved under OLD protocol on NEW standard splits → mixed results (1-2/4 wins). Running evolution DIRECTLY on standard protocol → 4/4 wins. The adapter architectures are optimized for each dataset's specific patterns.

## Baselines Tested
- linear (MeanPool + Linear): 49K params — poor on standard protocol (overfits)
- attention (Attention pooling + Linear): 50K params — poor on standard protocol
- conv (Conv1d + Linear): 209K params — best baseline, but evolved beats it every time
