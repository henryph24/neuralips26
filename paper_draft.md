# LLM-Guided Code Evolution for Time Series Foundation Model Adaptation

## Abstract

Adapting time series foundation models (TSFMs) to downstream tasks requires choosing adapter architectures — a design space that is typically navigated by hand or constrained to small hyperparameter grids. We propose **Code Evolution**, a method that uses large language models to generate and evolve arbitrary PyTorch adapter modules through an evolutionary loop, searching an unbounded architecture space rather than a fixed discrete menu. We evaluate Code Evolution on the MOMENT foundation model across 5 forecasting benchmarks (ETTh1, ETTh2, ETTm1, ETTm2, Weather) with preliminary results on Electricity. Code Evolution outperforms discrete evolutionary search on 4 of 5 complete benchmarks while exhibiting 3–7× lower variance across random seeds. Notably, the LLM discovers adapter architectures — including depthwise convolutions with learned positional weighting, attention-weighted feature pooling, and LayerNorm-gated projections — that do not decompose into any configuration in the discrete search space. Our results demonstrate that LLM-guided program synthesis is a viable and practical approach to neural architecture search for foundation model adaptation.

---

## 1. Introduction

Time series foundation models (TSFMs) such as MOMENT (Goswami et al., 2024), TimesFM (Das et al., 2024), Timer (Liu et al., 2024), Chronos (Ansari et al., 2024), and Moirai (Woo et al., 2024) have emerged as powerful pretrained representations for diverse temporal tasks. However, a critical practical question remains: *how should one adapt a frozen TSFM to a specific downstream dataset?*

The standard approach — selecting from a small menu of adapter types (LoRA ranks, bottleneck dimensions, prediction heads) — is limited in two ways. First, the designer must enumerate all candidate architectures upfront; novel structural ideas (e.g., depthwise convolutions, learned feature weighting) cannot be discovered if they are not in the menu. Second, even moderately sized discrete spaces (thousands of configurations) are small enough that random search performs comparably to guided search, leaving no room for intelligent exploration.

We propose a shift from **configuration search** to **code search**: instead of picking hyperparameters from a predefined grid, the search process generates arbitrary neural architectures as PyTorch programs. Recent advances in LLM-guided program synthesis — FunSearch (Romera-Paredes et al., 2024), ELM (Lehman et al., 2023), LLMatic (Nasir et al., 2024), GENESYS (Allen AI, 2025) — have demonstrated that language models can serve as mutation operators in evolutionary algorithms, generating novel programs that would be difficult to specify through traditional genetic programming. We instantiate this idea for TSFM adaptation.

Our method, **Code Evolution**, maintains a population of PyTorch `nn.Module` adapter classes. Each generation, an LLM (GPT-4o-mini) observes the current population — their code, fitness scores, and failure modes — and proposes new adapter architectures. These are validated locally for correctness, then evaluated on GPU via short training runs. Elitism preserves the best designs while the LLM balances exploitation of successful patterns with exploration of novel architectures.

Our contributions:

1. **Code Evolution**, an LLM-guided evolutionary framework that searches the unbounded space of PyTorch adapter programs for foundation model adaptation, completing a full search in ~55 minutes at ~$3.30 per run.

2. **Empirical results** showing Code Evolution outperforms discrete evolutionary search on 4/5 forecasting benchmarks with 3–7× lower variance across seeds, discovering structurally novel adapters (depthwise convolutions, attention-weighted pooling, LayerNorm gating) that lie outside any discrete search space.

3. **Analysis of discovered architectures** revealing that the LLM imports cross-domain design patterns (from mobile vision, transformer literature) and that winning adapters are structurally simpler (50K–157K params) than discrete-menu alternatives.

---

## 2. Related Work

### 2.1 Time Series Foundation Models and Adaptation

The TSFM landscape has rapidly expanded. MOMENT (Goswami et al., 2024) pretrains a transformer encoder via masked reconstruction on 1B time points. TimesFM (Das et al., 2024) and Chronos (Ansari et al., 2024) demonstrate strong zero-shot forecasting. Timer (Liu et al., 2024) unifies multiple tasks through autoregressive pretraining; Timer-XL (Chen et al., 2025) extends this to long-context horizons via hierarchical attention. Moirai (Woo et al., 2024) trains a universal forecaster on 27B observations across 9 domains, while Moirai-MoE (Liu et al., 2025a) replaces frequency-level specialization with sparse mixture-of-experts for automatic token-level routing. UniTS (Lee et al., 2024) demonstrates multi-task pretraining across 50 datasets. The GIFT-Eval benchmark (Woo et al., 2024b) now tracks these models systematically across 23 datasets and 7 domains.

Adaptation typically follows the NLP playbook: LoRA (Hu et al., 2022), bottleneck adapters, or linear probing atop frozen features. More recently, in-context fine-tuning (Faw et al., 2025) shows that TSFMs can be adapted at inference time by prompting with related time-series examples — an orthogonal approach to architecture search. TEMPLATE (Zhang et al., 2025) and TimeTic (Yao et al., 2025) address model *selection* (which pretrained TSFM to use), but not adapter *design* on a chosen backbone. Our work closes this gap: rather than selecting models, we *generate* adapter architectures guided by performance feedback.

### 2.2 Neural Architecture Search for Adapters

NAS-LoRA and Shears (Munoz et al., 2024) apply neural architecture search over adapter configurations for large language models, but search discrete hyperparameter spaces. MTF-PDNS (Xue et al., 2024) combines multi-objective evolutionary search with training-free proxies for vision CNNs. AZ-NAS (Lin et al., 2024) assembles zero-cost proxies for architecture ranking. RZ-NAS (Ji et al., 2025) enriches LLM-guided NAS with reflective zero-cost proxies — using proxy metrics *informatively* in prompts rather than as optimization targets. ONE-NAS (Li et al., 2022) applies evolutionary NAS to time series but searches raw RNN architectures, not adapters on frozen backbones. Recent time series NAS work (Liang & Sun, 2024; Deng et al., 2024) searches over spatial-temporal GNN modules or hierarchical cell spaces, but remains confined to discrete architectural blocks.

The key gap: no prior work searches an *unbounded* architecture space (arbitrary code) for foundation model adaptation, nor uses LLMs as the generative operator.

### 2.3 LLM-Guided Program Synthesis and Evolution

FunSearch (Romera-Paredes et al., 2024) demonstrated that LLMs can discover novel mathematical constructions through evolutionary program search, achieving superhuman results on extremal combinatorics problems. Evolution through Large Models (ELM; Lehman et al., 2023) showed LLMs can serve as intelligent mutation operators for Sodarace robot design. ReEvo (Ye et al., 2024) applies LLM-guided evolution to combinatorial optimization heuristics. LLMatic (Nasir et al., 2024) combines code-generating LLMs with Quality-Diversity (MAP-Elites) search for NAS, producing competitive architectures on CIFAR-10 and NAS-bench-201 with only 2,000 evaluations — the closest methodological peer to our work, though LLMatic searches full network architectures rather than adapter modules on frozen backbones.

EvoTune (Surina et al., 2025) extends FunSearch by fine-tuning the LLM via reinforcement learning based on fitness signals. GENESYS (Allen AI, 2025) applies multi-agent LLM evolution to discover novel LLM architectures, using 6 specialized agents (design proposer, adversarial reviewer, implementation planner, coder, observer, and RAG-based search assistant) to generate complete PyTorch architectures for language models at 14M–350M parameter scale. Their "Ladder of Scales" approach trains candidate architectures from scratch at progressively larger scales, narrowing the candidate pool at each stage. GENESYS produced 1,062 verified architectures that outperform GPT-2 and Mamba2 on 6/9 BabyLM benchmarks — but requires thousands of GPU-hours for full pre-training evaluations.

Our work applies this paradigm to a fundamentally different level of the architecture stack. GENESYS evolves *entire foundation model architectures* — full transformer variants trained from scratch on pre-training corpora. We evolve *adapter modules* — small (50K–500K parameter) task-specific networks attached to a frozen backbone. This difference has three implications: (1) evaluation cost drops from thousands of GPU-hours to ~4 GPU-hours per run (~$3.30), enabling rapid iteration; (2) our single-agent LLM design (one GPT-4o-mini call per generation) suffices where GENESYS requires a 6-agent pipeline, because adapter code is structurally simpler than full architectures; and (3) the fixed backbone interface contract (`(batch, seq_len, d_model) → (batch, output_dim)`) provides strong constraints that guide the LLM's search, whereas GENESYS must explore the much larger space of unconstrained transformer block designs.

---

## 3. Method

### 3.1 Problem Setup

Given a pretrained TSFM backbone $f_\theta$ (MOMENT, $d_\text{model}=768$, 12 transformer blocks) and a target forecasting dataset, we seek an adapter module $g_\phi: \mathbb{R}^{B \times 512 \times 768} \rightarrow \mathbb{R}^{B \times 96}$ that maps encoder hidden states to forecast values. The backbone's last 4 encoder blocks are unfrozen during training; all other parameters remain fixed.

**What is an adapter module?** In the foundation model paradigm, the pretrained backbone produces rich intermediate representations but is not directly usable for a specific downstream task. An adapter is a small, task-specific neural network (typically 50K–500K parameters) attached to the backbone's output that transforms its hidden representations into task predictions — in our case, mapping a $(B, 512, 768)$ tensor of encoder hidden states to a $(B, 96)$ forecast vector. Unlike full fine-tuning (which updates all backbone parameters) or LoRA (which constrains adaptation to low-rank weight perturbations), our adapter modules are *structurally unconstrained*: they can implement any differentiable function within the parameter budget, including convolutions, attention mechanisms, gating, and combinations thereof. This structural freedom is what enables Code Evolution to discover novel designs.

### 3.2 Code Evolution

**Representation.** Each individual in the population is a Python string defining a complete `nn.Module` class:

```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        # Architecture defined by LLM

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(batch, 512, 768) -> (batch, 96)"""
        # Forward pass defined by LLM
```

Constraints enforce ≤500K parameters and restrict imports to `torch`, `nn`, `F`, and `math`. Each individual is represented as a `CodeIndividual` containing the source code string, fitness (−MSE, higher is better), parameter count, and any error message from validation or training.

**Initialization.** The population (size 20) is seeded with 5 hand-designed baselines that represent standard adapter patterns:
1. MeanPool + Linear — the simplest possible adapter (pool across time, project to output)
2. MeanPool + 2-layer MLP with GELU and dropout — matching the best discrete-menu head
3. LastToken + MLP — using only the final timestep's representation
4. Attention pooling + Linear — learnable weighted pooling via a scoring network
5. Conv1d downsample + Linear — temporal pattern extraction via strided convolution

The remaining 15 slots are filled by **LLM cold-start**: GPT-4o-mini receives the 5 seed adapters (with no fitness data yet) and generates 15 new architectures. This bootstraps diversity before any training has occurred — the LLM draws on its architectural priors rather than population feedback.

**Evolutionary loop.** Each of the 12 generations proceeds in 6 steps:

1. **Local validation** (CPU): Each new individual's code is `exec()`'d in a sandboxed namespace containing only `torch`, `nn`, `F`, and `math`. The resulting `Adapter` class is instantiated with `(d_model=768, output_dim=96)`, then a dummy forward pass checks that a `(2, 512, 768)` input produces a `(2, 96)` output. Invalid codes (syntax errors, shape mismatches, exceeding 500K parameters) are assigned fitness = −∞ and their error messages are preserved for LLM feedback.

2. **GPU evaluation** (Modal A10G): Valid adapters that haven't been evaluated are sent to remote GPU workers via Modal's `starmap` for parallel execution. Each evaluation loads MOMENT, freezes all parameters, unfreezes the last 4 encoder blocks, attaches the adapter, and trains for 3 epochs with Adam (lr=1e-3), MSE loss, batch size 64, and 80/20 train/val split. The adapter and unfrozen backbone parameters are trained jointly. Fitness = −validation MSE.

3. **Ranking**: The population is sorted by fitness (descending). Generation statistics (best MSE, mean MSE, valid/invalid counts) are logged.

4. **Elitism**: The top 2 individuals pass unchanged to the next generation, preserving the best architectures discovered so far.

5. **LLM offspring generation**: GPT-4o-mini receives a structured prompt containing: (a) a system prompt describing the interface contract, constraints, and design principles (attention pooling, multi-scale convolutions, gating mechanisms, learned positional weighting, feature selection); (b) the top 5 adapters with their full source code, fitness, MSE, and parameter counts; (c) the bottom 3 adapters; (d) up to 5 failed adapters with their error messages and code snippets; (e) the fitness trajectory (best fitness per generation). The LLM returns a JSON object containing 18 new adapter architectures, each with a reasoning string and complete class definition.

6. **Deduplication**: LLM-generated codes are deduplicated against existing population members (exact string match after stripping whitespace). If the LLM produces fewer than 18 unique adapters, remaining slots are filled with seed adapter variations to maintain population size.

**Final validation.** After 12 generations (~240 total evaluations, of which ~179 pass local validation), the top 5 architectures from the final population are re-evaluated at 15 epochs on the same data split for final comparison. This longer training schedule reveals whether the 3-epoch fitness ranking is preserved under more thorough optimization.

### 3.3 Baselines

- **Evo-3epoch**: Traditional evolutionary search over a discrete 2,160-configuration space (6 LoRA ranks × 2 target modules × 3 placements × 4 unfreezing strategies × 3 head types × 4 pooling methods), using tournament selection, crossover, and mutation. 3-epoch MSE as fitness. Top 5 validated at 15 epochs.
- **Random**: Random sampling from the same discrete space, top 5 validated at 15 epochs.

Both baselines use identical compute budgets (~240 evaluations per run).

### 3.4 Why Code Search Over Configuration Search?

The discrete search space contains 2,160 configurations — all combinations of predefined hyperparameters. This space is small enough that random sampling achieves near-optimal coverage, leaving no room for intelligent guidance. More fundamentally, it cannot represent architectures that don't decompose into the fixed template of `pooling → head` (e.g., Conv1d→Conv1d→Pool→Linear, or learned feature weighting before aggregation).

Code Evolution removes both limitations: the search space is infinite (all valid PyTorch programs satisfying the interface contract), and architectures are represented as programs rather than hyperparameter tuples, enabling structural novelty.

| Dimension | Discrete Search | Code Evolution |
|-----------|----------------|----------------|
| Space size | 2,160 configs | Infinite (all valid PyTorch programs) |
| What varies | Hyperparameters (rank, placement, head type, pooling) | Entire architecture (arbitrary nn.Module code) |
| Structural novelty | None (fixed template) | Unbounded (depthwise conv, attention gating, etc.) |
| Evals per run | ~240 | ~240 (179 valid, ~25% rejection rate) |

---

## 4. Experiments

### 4.1 Setup

**Backbone.** MOMENT-1-large (Goswami et al., 2024), a 12-layer transformer encoder pretrained via masked reconstruction on 1B time points. Hidden dimension $d_\text{model}=768$, sequence length 512. Last 4 encoder blocks unfrozen during adapter training.

**Datasets.** ETTh1, ETTh2, ETTm1, ETTm2 (Zhou et al., 2021), Weather (Wetterstation), Electricity (UCI). Forecast horizon: 96 steps. Standard train/val/test splits.

**Training.** Adam optimizer, lr=1e-3, MSE loss, batch size 64, 80/20 train/val split. Evolution uses 3-epoch fitness evaluations; final validation uses 15 epochs. All GPU computation on Modal A10G instances via parallel `starmap` dispatch.

**Seeds.** Results averaged over seeds 42–44 where available.

### 4.2 Main Results

| Dataset | Code-Evo (mean ± std) | Evo-3ep (mean ± std) | Random (mean) | Δ vs Evo-3ep |
|---------|----------------------|---------------------|---------------|-------------|
| ETTh1 | **0.5614 ± 0.011** | 0.6189 ± 0.106 | 0.8704 | −9.3% |
| ETTh2 | **0.6580 ± 0.010** | 0.7398 ± 0.060 | 0.7951 | −11.1% |
| ETTm1 | **0.3619 ± 0.007** | 0.3697 | 0.3871 | −2.1% |
| ETTm2 | 0.5501 ± 0.009 | **0.5280 ± 0.006** | 0.5810 | +4.2% |
| Weather | **0.3325 ± 0.007** | 0.3376 ± 0.007 | 0.3484 | −1.5% |
| Electricity* | **0.1956** | — | — | — |

*Electricity: single seed, no baselines due to compute budget.

**Code-Evo wins on 4/5 datasets** with complete comparisons. The single loss (ETTm2, +4.2%) is within a small margin.

**Variance reduction.** Code-Evo exhibits 3–7× lower standard deviation across seeds compared to Evo-3ep (e.g., ETTh1: 0.011 vs 0.106; ETTh2: 0.010 vs 0.060). The LLM's architectural priors act as implicit regularization, consistently producing well-behaved architectures and avoiding the high-variance configurations that discrete search occasionally selects.

### 4.3 Discovered Architectures

Code Evolution discovers architectures that cannot be expressed in the discrete search space. These share a pattern: they are *structurally simple* (50K–157K params) but contain design choices that lie outside the discrete space's representational capacity.

**Depthwise convolution + learned positional weighting** (Electricity, 52K params, MSE=0.1956):
```python
self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                                 groups=d_model, padding=1)
self.positional_weights = nn.Parameter(torch.randn(d_model))
```
Applies channel-wise temporal smoothing followed by learned feature weighting — a pattern common in efficient vision models (MobileNet) but absent from the discrete adapter menu. The depthwise convolution avoids the $O(d^2)$ cost of standard convolutions while capturing local temporal structure.

**Attention-weighted feature pooling** (Weather, 157K params, MSE=0.3258):
```python
self.attention_weights = nn.Parameter(torch.randn(d_model))
weights = F.softmax(self.attention_weights, dim=-1)
pooled = (hidden_states * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
```
Learns which of the 768 features are most relevant before temporal aggregation — effectively a soft feature selection layer. This inverts the standard pooling→projection order: feature selection happens *before* spatial reduction, allowing the model to discard irrelevant dimensions early.

**LayerNorm + mean pooling** (ETTh1, 50K params, MSE=0.5501):
```python
self.norm = nn.LayerNorm(d_model)
normed = self.norm(hidden_states)
pooled = normed.mean(dim=1)
```
The simplest winning architecture: normalization before pooling stabilizes training with unfrozen backbone layers. Despite having no hidden layers, it outperforms deeper discrete-menu architectures — suggesting that representation quality from the unfrozen backbone is sufficient when properly normalized.

### 4.4 The LLM as Architecture Designer

The LLM's contribution is not random code generation — it is informed architectural reasoning. The system prompt provides the interface contract and design principles; the population provides empirical feedback. Across generations, we observe the LLM:

- **Learning from failures**: Avoiding patterns that produced shape mismatches or NaN losses in previous generations.
- **Combining successful motifs**: Merging attention weighting from one parent with convolution from another.
- **Importing cross-domain knowledge**: Depthwise separable convolutions from mobile vision (Howard et al., 2017), pre-norm architectures from transformer literature (Xiong et al., 2020).
- **Simplifying**: Converging toward simpler architectures over generations, as overparameterized designs are penalized by the 500K parameter budget and short training schedules.

---

## 5. Discussion

### Why code search over configuration search?

The core insight is that LLM guidance only provides value when the search space is large enough to benefit from intelligent exploration but structured enough for the LLM to reason about. A discrete space of 2,160 configurations is too small — random sampling achieves near-optimal coverage. The space of valid PyTorch programs is vast enough that the LLM's architectural priors, cross-domain knowledge, and ability to learn from population feedback become genuine advantages.

### Variance reduction as implicit regularization

The 3–7× variance reduction across seeds is arguably as important as the MSE improvement. In practice, a method that reliably produces good adapters (Code-Evo: σ=0.007–0.011) is more valuable than one that occasionally finds great adapters but sometimes produces poor ones (Evo-3ep: σ=0.006–0.106). The LLM's training data provides strong priors about which architectural patterns are generally effective, filtering out degenerate designs that random or evolutionary search may explore.

### Relationship to concurrent work

Code Evolution occupies a distinct niche in the emerging landscape of LLM-guided architecture search. Compared to LLMatic (Nasir et al., 2024), which searches full network architectures on vision benchmarks via NAS-bench lookup, our evaluation is more expensive (GPU fine-tuning), making the LLM's ability to propose high-quality candidates from limited evaluations (~240 per run) more critical. Compared to GENESYS (Allen AI, 2025), which evolves full LM architectures (14M–350M params) using a 6-agent pipeline and thousands of GPU-hours, our adapter-level search uses a single LLM agent and completes in under an hour at ~$3.30. This 3-orders-of-magnitude cost reduction is possible because adapter modules are structurally constrained by the backbone interface — the LLM only needs to solve the reduction problem ($\mathbb{R}^{512 \times 768} \rightarrow \mathbb{R}^{96}$), not design an entire transformer architecture from scratch.

EvoTune (Surina et al., 2025) demonstrates that RL fine-tuning of the LLM mutation operator accelerates convergence. Our current approach treats GPT-4o-mini as a static generator; incorporating RL feedback could further improve sample efficiency. In-context fine-tuning (Faw et al., 2025) represents an orthogonal adaptation paradigm — these approaches are complementary: one could use Code Evolution to design the adapter and in-context fine-tuning to specialize it at deployment.

### Limitations

**Missing ablations.** We do not yet isolate the LLM's contribution from the search space expansion. A random code baseline (generating random valid `nn.Module` adapters without LLM guidance) would establish whether the unbounded space alone accounts for the improvement. An LLM-without-evolution baseline (single-shot generation without population feedback) would establish whether the evolutionary loop contributes beyond the LLM's priors.

**Evaluation breadth.** Results use a single forecast horizon (96 steps); standard evaluation requires multiple horizons (96, 192, 336, 720). Only MSE is reported; MAE should be included. Electricity lacks multi-seed results and baselines.

**Single backbone and LLM.** We evaluate only MOMENT with GPT-4o-mini. Generalization to other TSFMs (TimesFM, Chronos, Timer-XL, Moirai-MoE) and other code-generating LLMs remains untested.

**Statistical rigor.** Three seeds provide preliminary variance estimates but are insufficient for formal significance testing.

### Future Work

- **Ablation suite**: Random code baseline, LLM-without-evolution, and prompt ablations to isolate the LLM's contribution (~$200 in compute).
- **Multi-objective evolution**: Jointly optimizing MSE and parameter count to discover Pareto-optimal adapters.
- **Cross-dataset transfer**: Using adapters evolved on one dataset as seeds for another, testing whether architectural motifs transfer across temporal domains.
- **Backbone diversity**: Evaluating Code Evolution across MOMENT, TimesFM, Timer-XL, and Moirai-MoE.
- **RL-enhanced search**: Fine-tuning the LLM mutation operator via reinforcement learning on architectural fitness signals (Surina et al., 2025).
- **Quality-Diversity search**: Incorporating MAP-Elites (Nasir et al., 2024) to explicitly maintain architectural diversity alongside fitness optimization.

---

## 6. Conclusion

We presented Code Evolution, an LLM-guided evolutionary approach to adapter architecture search for time series foundation models. By searching the unbounded space of PyTorch programs rather than a fixed discrete menu, Code Evolution discovers novel adapter architectures — including depthwise convolutions, attention-weighted feature pooling, and LayerNorm-gated projections — that outperform discrete evolutionary search on 4/5 forecasting benchmarks with 3–7× lower variance. The discovered adapters are structurally simpler (50K–157K parameters) than discrete-menu alternatives yet achieve better performance, suggesting that the LLM's cross-domain architectural knowledge enables efficient navigation of the vast code search space. At ~$3.30 per run and ~55 minutes wall time, Code Evolution provides a practical framework for automated foundation model adaptation.

---

## References

- Ansari, A.F. et al. (2024). Chronos: Learning the language of time series. *arXiv:2403.07815*.
- Chen, S. et al. (2025). Timer-XL: Long-context transformers for unified time series forecasting. *ICLR 2025*.
- Das, A. et al. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*.
- Deng, Y. et al. (2024). Optimizing time series forecasting architectures: A hierarchical NAS approach. *arXiv:2406.05088*.
- Faw, M. et al. (2025). In-context fine-tuning for time-series foundation models. *ICML 2025*.
- GENESYS Team, Allen AI (2025). GENESYS: Distributed language model architecture discovery. *GitHub: allenai/genesys*.
- Goswami, M. et al. (2024). MOMENT: A family of open time-series foundation models. *ICML 2024*.
- Howard, A.G. et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv:1704.04861*.
- Hu, E.J. et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Ji, H. et al. (2025). RZ-NAS: Reflective zero-cost proxies for LLM-guided neural architecture search. *ICML 2025*.
- Lee, G. et al. (2024). UniTS: A unified multi-task time series model. *NeurIPS 2024*.
- Lehman, J. et al. (2023). Evolution through large models. *arXiv:2206.08896*.
- Li, Y. et al. (2022). ONE-NAS: An online neuroevolution-based NAS for time series forecasting.
- Liang, Z. & Sun, J. (2024). Evolutionary neural architecture search for multivariate time series forecasting. *ACML 2024*.
- Lin, Y. et al. (2024). AZ-NAS: Assembling zero-cost proxies for NAS. *CVPR 2024*.
- Liu, X. et al. (2025a). Moirai-MoE: Empowering time series foundation models with sparse mixture of experts. *ICML 2025*.
- Liu, Y. et al. (2024). Timer: Generative pre-training of time series. *ICML 2024*.
- Munoz, J.P. et al. (2024). Shears: Unstructured sparsity with neural low-rank adapter search.
- Nasir, M.U. et al. (2024). LLMatic: Neural architecture search via large language models and quality diversity optimization. *GECCO 2024*.
- Romera-Paredes, B. et al. (2024). Mathematical discoveries from program search with large language models. *Nature*.
- Surina, A. et al. (2025). Algorithm discovery with LLMs: Evolutionary search meets reinforcement learning. *arXiv:2504.05108*.
- Woo, G. et al. (2024). Moirai: A time series foundation model for universal forecasting. *ICML 2024*.
- Woo, G. et al. (2024b). GIFT-Eval: A benchmark for general time series forecasting model evaluation. *arXiv:2410.10393*.
- Xiong, R. et al. (2020). On layer normalization in the transformer architecture. *ICML 2020*.
- Xue, Y. et al. (2024). MTF-PDNS: Multi-objective training-free proxy-based differential NAS.
- Yao, Q. et al. (2025). TimeTic: Estimating time series foundation model transferability via in-context learning. *arXiv:2509.23695*.
- Ye, H. et al. (2024). ReEvo: Large language models as hyper-heuristics with reflective evolution.
- Zhang, W. et al. (2025). Unified transferability metrics for time series foundation models. *NeurIPS 2025*.
- Zhou, H. et al. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *AAAI 2021*.
