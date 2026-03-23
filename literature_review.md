# Literature Review: LLM-Guided Code Evolution for TSFM Adaptation

## 1. Time Series Foundation Models (2024–2025)

### 1.1 Core Models

| Model | Authors | Venue | Architecture | Key Contribution |
|-------|---------|-------|-------------|-----------------|
| **MOMENT** | Goswami et al. | ICML 2024 | Transformer encoder, masked reconstruction | Pretrained on Time-series Pile (1B+ time points). Strong on forecasting, classification, anomaly detection, imputation. Our backbone. |
| **TimesFM** | Das et al. | ICML 2024 | Decoder-only transformer | 100B real-world time points, causal LM objective. Near-par zero-shot with specialized models. 17M–200M variants. |
| **Chronos** | Ansari et al. | NeurIPS 2024 Workshop | Encoder-decoder, quantized tokens | Reframes time series as token sequences. Probabilistic forecasting via discrete output tokens. |
| **Timer / Timer-XL** | Liu et al.; Chen et al. | ICML 2024 / ICLR 2025 | Decoder-only, hierarchical attention | Autoregressive pretraining, long-context (up to 10K steps). Timer-XL at ICLR 2025 emphasizes long-horizon forecasting. |
| **Moirai** | Woo et al. (Salesforce) | ICML 2024 | Masked encoder, any-variate attention | Universal forecaster trained on LOTSA (27B observations, 9 domains). Multiple patch sizes for multi-frequency. |
| **Moirai-MoE** | Liu et al. (Salesforce) | ICML 2025 | Sparse MoE Transformer | Replaces frequency-level specialization with token-level MoE routing. Up to 17% improvement over Moirai with 65x fewer activated params. |
| **Moirai 2.0** | Aksu et al. (Salesforce) | 2025 | Decoder-only (transition from encoder) | #1 on GIFT-Eval by MASE (non-leaking). Expanded pretraining data including synthetic series. |
| **FlowState** | Graf et al. (IBM) | NeurIPS 2025 Workshop | State-space model (SSM) | Dynamic forecasting horizons, invariant to sampling rates. #2 on GIFT-Eval zero-shot. Part of IBM Granite family. |
| **UniTS** | Lee et al. | NeurIPS 2024 | Unified encoder-decoder | Single model for denoising, forecasting, classification across 50 datasets. Multi-task pretraining feasibility. |

### 1.2 Relevance to Our Work
Our paper uses MOMENT as the frozen backbone. The rapid proliferation of TSFMs (MOMENT, TimesFM, Chronos, Timer, Moirai, FlowState) strengthens our motivation: with many backbone choices, efficient adaptation becomes critical. Our Code Evolution approach is backbone-agnostic in principle — future work should test on Moirai and Timer-XL.

---

## 2. TSFM Adaptation and Fine-Tuning Methods

### 2.1 Parameter-Efficient Fine-Tuning

| Method | Authors | Venue | Approach | Results |
|--------|---------|-------|----------|---------|
| **LoRA for TSFMs** | Zhang et al. | SIGKDD 2024 | Low-rank updates in attention/FFN layers | Comparable performance with 2% trainable params, 70% memory reduction on MOMENT/Chronos |
| **ChronosX** | Emergent Minds | 2025 | Lightweight adapters + cross-attention covariates | 6% error reduction vs full fine-tuning on multivariate benchmarks |
| **Gen-P-Tuning** | arXiv 2411.12824 | 2024 | Prompt tokens prepended to TSFM inputs | Matches full fine-tuning on classification with <0.5% gap, <1% extra params |

### 2.2 Model Selection and Transferability

| Method | Authors | Venue | Approach | Results |
|--------|---------|-------|----------|---------|
| **TEMPLATE** | Zhang, Chen et al. | **NeurIPS 2025** | CKA-based transferability scores (DL, PL, TA) | 35% improvement in weighted Kendall τ_w over ETran. Selects best pretrained model from pool. |
| **TimeTic** | Yao, Jin et al. | arXiv 2509.23695, 2025 | In-context learning for transferability estimation | Entropy evolution across layers + tabular FM as ICL. τ≈0.6, 30% improvement over zero-shot baseline. |
| **In-Context Fine-Tuning** | Faw et al. (Google) | **ICML 2025** | Multiple time-series examples as prompts at inference | Matches explicit fine-tuning performance. No gradient steps needed. |

### 2.3 Relevance to Our Work
- **TEMPLATE** (NeurIPS 2025) is directly cited in our paper. Confirmed venue: NeurIPS 2025 poster. Authors: Weiyang Zhang, Xinyang Chen, Xiucheng Li, Kehai Chen, Weili Guan, Liqiang Nie (HIT Shenzhen). Official title: "Unified Transferability Metrics for Time Series Foundation Models". Our §4.1 shows TEMPLATE scores don't predict adapter configuration quality (designed for model selection, not adapter search).
- **TimeTic** also addresses model selection but via ICL rather than CKA metrics. Both are passive (select among existing models) while our method is generative (creates new adapters).
- The gap between "selecting models" and "generating adapters" is our key positioning.

---

## 3. Neural Architecture Search (NAS) for Time Series

| Method | Authors | Venue | Search Space | Key Finding |
|--------|---------|-------|-------------|-------------|
| **EMTSF** | Liang & Sun | ACML 2024 | Spatial-temporal GNN modules via genetic algorithms | Evolutionary NAS on STGNNs outperforms handcrafted designs |
| **Hierarchical NAS** | Deng et al. | arXiv 2406.05088, 2024 | Multi-level search over conv/recurrent/attention blocks | Selects lightweight, high-performing forecasting architectures |
| **Chain-Structured NAS** | Levchenko et al. | Int. J. Data Sci. Anal. 2025 | MLP/CNN/RNN/TFT for financial TS | Bayesian and Hyperband outperform RL for small financial datasets |
| **AutoSeries** | Zhao et al. | NeurIPS 2025 | Temporal conv + LSTM + attention cells | Automated NAS under compute constraints for energy/traffic forecasting |

### Relevance
No prior NAS work searches an unbounded code space for foundation model adapters. Existing TS NAS searches discrete cell spaces or fixed architectural blocks. Our Code Evolution operates on arbitrary PyTorch programs — a fundamentally different (and larger) search space.

---

## 4. LLM-Guided Program Synthesis and Architecture Search

### 4.1 Foundational Methods

| Method | Authors | Venue | Approach | Key Contribution |
|--------|---------|-------|----------|-----------------|
| **FunSearch** | Romera-Paredes et al. | Nature 2024 | LLM as mutation operator in evolutionary program search | Discovered novel mathematical constructions (cap sets). Superhuman results on extremal combinatorics. |
| **ELM** | Lehman et al. | arXiv 2206.08896, 2023 | LLMs as intelligent mutation operators | Sodarace robot design. Showed LLMs can generate diverse morphologies. |
| **ReEvo** | Ye et al. | 2024 | LLM-guided reflective evolution for optimization heuristics | LLM proposes heuristic mutations with self-reflection on failures. |

### 4.2 Recent Advances (2025)

| Method | Authors | Venue | Approach | Key Contribution |
|--------|---------|-------|----------|-----------------|
| **EvoTune** | Surina et al. (EPFL) | arXiv 2504.05108, 2025 | RL fine-tuning of LLM within evolutionary search | Closes the loop: LLM generates programs → RL updates LLM policy based on fitness. Outperforms FunSearch on bin packing, TSP, flatpack. |
| **GENESYS** | Allen AI | 2025 | Multi-agent LLM evolution for LLM architectures | "Ladder of Scales" (14M–350M). LLM-guided genetic programming outperforms direct prompting. Searches transformer block designs. |
| **LLMatic** | Nasir et al. | GECCO 2024 | LLM + Quality-Diversity (QD) search | Combines code-generating LLMs with MAP-Elites for diverse NAS. Competitive on CIFAR-10/NAS-bench-201 with only 2K evaluations. |
| **RZ-NAS** | Ji et al. | ICML 2025 | Reflective zero-cost proxies in LLM-guided NAS | Enriches LLM prompts with zero-cost proxy metrics. LLM evaluates architectures at code level without full training. |

### 4.3 Relevance to Our Work
- **FunSearch** is our methodological ancestor. We apply the same evolutionary program synthesis paradigm to adapter architecture search.
- **EvoTune** (2025) extends FunSearch with RL fine-tuning of the LLM — a natural future direction for our work.
- **LLMatic** is the closest NAS work to ours: LLM generates PyTorch code + QD search. Difference: they search full network architectures on CIFAR-10; we search adapter modules on frozen TSFMs.
- **GENESYS** searches foundation model architectures themselves (compute-heavy); we search adapters on existing FMs (compute-light, minutes vs days).
- **RZ-NAS** combines zero-cost proxies with LLM guidance — relevant to our §4.2 negative result on GP-evolved proxies. Our finding that proxies fail when optimized against (Goodhart's Law) is complementary.

---

## 5. Benchmarking and Evaluation (2025)

| Benchmark | Authors | Key Features |
|-----------|---------|-------------|
| **GIFT-Eval** | Salesforce (Woo et al.) | 23 datasets, 7 domains, univariate + multivariate. Contamination-aware. De facto standard leaderboard. Current #1: TimesFM-2.5 (Google), #2: FlowState (IBM), #3: Moirai 2.0. |
| **fev-bench** | Shchur et al. (Amazon) | 100 forecasting tasks, 7 domains, 46 tasks with covariates. Bootstrap confidence intervals. Emphasizes statistical rigor. |
| **FoundTS** | Zhang et al. | KDD 2025. Critiques TSFMs: statistical models outperform on univariate; TSFMs excel on multivariate/cross-domain zero-shot. |

### Relevance
Our evaluation uses ETTh1/h2, ETTm1/m2, Weather, Electricity — standard benchmarks that appear in GIFT-Eval. Future work should evaluate on GIFT-Eval's broader suite and fev-bench to strengthen generalization claims.

---

## 6. Key NeurIPS 2025 Time Series Papers

### Confirmed NeurIPS 2025 Papers
1. **TEMPLATE** — "Unified Transferability Metrics for Time Series Foundation Models" (Zhang et al., HIT Shenzhen). Poster #5412. [Directly cited in our paper]
2. **FlowState** — IBM's SSM-based TSFM. NeurIPS 2025 Workshop on Time Series Foundation Models.
3. **OATS** — Online data augmentation during TSFM pretraining. Reduces forecasting error volatility by 15%.
4. **SymTime** — Series-symbol data generation for pretraining. Outperforms real-world pretrained TSFMs on 5 tasks (Wang et al., OpenReview).

### Related Venue Papers (ICML 2025, ICLR 2025)
5. **Moirai-MoE** — ICML 2025. Sparse MoE for time series FMs.
6. **Timer-XL** — ICLR 2025. Long-context decoder-only forecasting.
7. **In-Context Fine-Tuning** — ICML 2025 (Faw et al., Google). Prompting TSFMs with examples at inference.
8. **RZ-NAS** — ICML 2025. Reflective zero-cost proxies + LLM-guided NAS.

---

## 7. Positioning and Gaps

### Our Unique Contributions (validated by literature review)
1. **No prior work searches unbounded code space for FM adapters.** LLMatic searches full architectures; GENESYS searches FM architectures; NAS-LoRA/Shears search discrete adapter hyperparameters. We search arbitrary PyTorch adapter modules on frozen backbones.
2. **Negative results on transferability metrics and zero-cost proxies are novel.** TEMPLATE (NeurIPS 2025) shows these metrics work for model selection. We show they fail for adapter configuration search — a complementary finding.
3. **The "Goodhart's Law" finding on proxy-guided evolution is practically important.** RZ-NAS (ICML 2025) integrates proxies into LLM prompts (informational use); our work shows they catastrophically fail as optimization targets.
4. **Code Evolution occupies a unique cost-efficiency niche.** GENESYS: days of compute for FM architecture search. Our method: ~55 minutes per run for adapter search.

### Gaps to Address in Paper
- **Missing comparison with LLMatic/QD-based approaches.** Should cite and discuss.
- **EvoTune** as natural extension — RL fine-tuning of the code-generating LLM.
- **Moirai-MoE and Timer-XL** should be mentioned as potential backbone targets for future work.
- **GIFT-Eval and fev-bench** should be cited as standard benchmarks; discuss why our ETT/Weather/Electricity evaluation is sufficient for a methods paper but broader evaluation is future work.
- **In-Context Fine-Tuning (ICML 2025)** is an alternative adaptation paradigm — no architecture search needed. Worth mentioning as orthogonal approach.

---

## 8. Updated Reference List (New Citations to Add)

### Must-Cite (directly relevant, top venues)
- Moirai-MoE (Liu et al., ICML 2025) — MoE for TSFM heterogeneity
- TEMPLATE (Zhang et al., NeurIPS 2025) — Already cited, update venue/authors to match official publication
- EvoTune (Surina et al., 2025) — RL + evolutionary search, extends FunSearch
- LLMatic (Nasir et al., GECCO 2024) — LLM + QD for NAS, closest methodological peer
- In-Context Fine-Tuning (Faw et al., ICML 2025) — Alternative adaptation paradigm
- RZ-NAS (Ji et al., ICML 2025) — Zero-cost proxies in LLM-guided NAS

### Should-Cite (strengthens related work)
- Timer-XL (Chen et al., ICLR 2025) — Latest Timer variant
- Moirai 2.0 (Aksu et al., 2025) — Current GIFT-Eval leader
- FlowState (IBM, NeurIPS 2025 Workshop) — SSM-based TSFM
- GIFT-Eval (Salesforce, 2024) — Standard TSFM benchmark
- fev-bench (Shchur et al., 2025) — Rigorous benchmark with covariates
- FoundTS (Zhang et al., KDD 2025) — Critical evaluation of TSFMs
- TimeTic (Yao et al., 2025) — ICL-based model selection

### Nice-to-Cite (context)
- SymTime (NeurIPS 2025) — Synthetic pretraining data
- UniTS (Lee et al., NeurIPS 2024) — Multi-task TSFM
- EMTSF (Liang & Sun, ACML 2024) — Evolutionary NAS for time series
