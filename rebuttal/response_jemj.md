**Response to Reviewer jemj**

We are very grateful for the reviewer's generous assessment, and especially for valuing the paper's novelty and the signal-ratio metric. Below are brief answers to the four questions, folding in the two limitations noted.

**Task diversity / Q1 (classification, anomaly).** *(Meta-review point 3: "experimental setup focusses only on forecasting (primary) and imputation (secondary).")* Thank you for raising this. The routing signal does not change with the task: it comes from the raw input window, and instance normalization strips the same per-window statistics whether the objective is forecasting, classification, or anomaly detection. So the collapse and its raw-routing fix carry over, and we confirmed it directly. [NEW] We ran RR-MoA on three UEA classification datasets and a reconstruction-based anomaly check on SMD (3 seeds each): routing entropy stays healthy in every cell (near the log 5 ≈ 1.61 maximum, far from the 0.00 collapse floor). Frozen-backbone features cap absolute accuracy and detection, so we read both as mechanism-generalization, not performance claims.

*Classification (RR-MoA K5, Top-2, per seed):*

| dataset | seed | entropy | accuracy |
|---|---|---|---|
| BasicMotions | 42 | 1.575 | 0.896 |
| BasicMotions | 43 | 1.596 | 0.792 |
| BasicMotions | 44 | 1.605 | 0.906 |
| EthanolConcentration | 42 | 1.600 | 0.257 |
| EthanolConcentration | 43 | 1.593 | 0.267 |
| EthanolConcentration | 44 | 1.587 | 0.330 |
| JapaneseVowels | 42 | 1.462 | 0.277 |
| JapaneseVowels | 43 | 1.526 | 0.296 |
| JapaneseVowels | 44 | 1.383 | 0.317 |

*Anomaly (SMD reconstruction, per seed):*

| machine | seed | entropy | ROC-AUC |
|---|---|---|---|
| machine-1-1 | 42 | 1.606 | 0.595 |
| machine-1-1 | 43 | 1.587 | 0.560 |
| machine-1-1 | 44 | 1.555 | 0.593 |
| machine-2-1 | 42 | 1.607 | 0.528 |
| machine-2-1 | 43 | 1.604 | 0.510 |
| machine-2-1 | 44 | 1.597 | 0.532 |
| machine-3-1 | 42 | 1.536 | 0.570 |
| machine-3-1 | 43 | 1.518 | 0.564 |
| machine-3-1 | 44 | 1.521 | 0.573 |

So routing stays healthy on both new task types, confirming the routing signal carries over from forecasting rather than differing from it.

**Frozen-backbone requirement / Q2 (raw routing + selective unfreezing).** *(Meta-review point 4: "the method requires a strictly frozen backbone.")* We appreciate this question, and we can see why it comes up. Our best results use a frozen backbone, and Proposition 1 explains why (the co-adaptation feedback loop). That makes a frozen backbone look like a hard requirement, one that would limit the method when adaptation is genuinely needed. But a strictly frozen backbone is not required: raw routing combines with selective unfreezing and still works well. Table 1 shows RR-MoA (raw routing) beating the best fixed adapter not only frozen but also with Last-2 and Last-4 blocks unfrozen (ETTh1 -44% / -29% / -32%). Frozen is simply the strongest setting on 4/6 datasets, a design choice the method benefits from, not one it imposes. It is also a deployment advantage rather than a constraint: because the backbone stays fixed, one shared TSFM can serve many tasks through small hot-swappable adapters (App. B). So raw routing is compatible with backbone adaptation; freezing is a recommended default, not a requirement.

**Q3 (H∈{1000,2000}).** We test up to H=720, and within that range the frozen backbone is not a bottleneck. Our deployment variant Residual-IA+ still significantly beats DLinear at H=720 on the non-stationary datasets (ETTm2 -27.0%; ETTh2 also a significant win), while DLinear keeps its edge on short-seasonality data. So long-horizon behavior depends on the dataset, with clear gains where non-stationarity dominates and DLinear keeping its edge elsewhere. The TSFM's contribution is in fact largest at intermediate horizons (H=192: 6/6 match-or-beat DLinear), and the raw branch takes on more of the work as the horizon grows. [NEW] We now test H∈{1000,2000} directly. RR-MoA still beats the best fixed adapter at every cell (21-72%, no collapse, entropy 1.03-1.54), and Residual-IA+ matches or beats DLinear on 14/18 cells, decisively on the non-stationary ETTm2 (-39% at H=1000, -42% at H=2000, 3/3) and at parity on Weather and Electricity. So the frozen representation does not become limiting even at H=2000.

*Residual-IA+ vs DLinear at H ∈ {1000, 2000} (per seed; MoB = match-or-beat):*

| dataset | H | seed | RIA+ MSE | DLinear MSE | gap | MoB |
|---|---|---|---|---|---|---|
| ETTm2 | 1000 | 42 | 0.431 | 0.724 | -40.5% | Y |
| ETTm2 | 1000 | 43 | 0.419 | 0.999 | -58.0% | Y |
| ETTm2 | 1000 | 44 | 0.408 | 0.500 | -18.5% | Y |
| Electricity | 1000 | 42 | 0.265 | 0.269 | -1.6% | Y |
| Electricity | 1000 | 43 | 0.266 | 0.259 | +2.7% | N |
| Electricity | 1000 | 44 | 0.263 | 0.265 | -0.8% | Y |
| Weather | 1000 | 42 | 0.368 | 0.371 | -0.7% | Y |
| Weather | 1000 | 43 | 0.373 | 0.370 | +0.8% | N |
| Weather | 1000 | 44 | 0.378 | 0.380 | -0.5% | Y |
| ETTm2 | 2000 | 42 | 0.435 | 0.592 | -26.5% | Y |
| ETTm2 | 2000 | 43 | 0.462 | 0.830 | -44.4% | Y |
| ETTm2 | 2000 | 44 | 0.447 | 1.006 | -55.6% | Y |
| Electricity | 2000 | 42 | 0.329 | 0.334 | -1.4% | Y |
| Electricity | 2000 | 43 | 0.333 | 0.341 | -2.6% | Y |
| Electricity | 2000 | 44 | 0.336 | 0.334 | +0.7% | N |
| Weather | 2000 | 42 | 0.397 | 0.403 | -1.5% | Y |
| Weather | 2000 | 43 | 0.416 | 0.408 | +1.9% | N |
| Weather | 2000 | 44 | 0.404 | 0.413 | -2.2% | Y |

**Q4 (training memory, raw + normalized paths).** The two paths are very asymmetric, so maintaining both does not come close to doubling memory. The normalized path is the frozen ~40M backbone: because it is frozen, no gradients flow into it, so its weights carry no optimizer state and its internal activations need not be retained for backpropagation. The raw path is a single univariate window (batch × length, ~0.26 MB at the standard batch 128, L=512) fed to the 853-parameter router and a small raw branch, so it adds negligible activation memory. The only trainable memory is the adapters and router (354-466K + 853, about 1% of the backbone) plus their optimizer state. So training peak memory is set by the backbone forward pass, the same floor as any frozen-backbone adapter method; at inference, the full router-plus-expert stack adds at most +5.9% latency and +6 MB (Table B.1), a hard ceiling on that overhead.

We thank the reviewer again for the generous and encouraging assessment. We are especially glad the paper's core strengths came through: a novel and important problem, a rigorous causal analysis with the RevIN intervention isolating normalization as the cause, strong experimental validation, the theoretical results, a principle shown to be architecture-agnostic, and the practical value of an 853-parameter router. With the two noted limitations now addressed, task diversity through the new classification and anomaly experiments and the frozen-backbone requirement shown to be a recommended default rather than a hard constraint, we hope the contribution reads as stronger still.
