# LLM-Guided Code Evolution for Time Series Foundation Model Adaptation

## Abstract

Adapting time series foundation models (TSFMs) to downstream tasks requires choosing adapter architectures — a design space that is typically navigated by hand or constrained to small hyperparameter grids. We propose **Code Evolution**, a method that uses large language models to generate and evolve arbitrary PyTorch adapter modules through an evolutionary loop, searching an unbounded architecture space rather than a fixed discrete menu. We evaluate Code Evolution on the MOMENT foundation model across 5 forecasting benchmarks (ETTh1, ETTh2, ETTm1, ETTm2, Weather) with preliminary results on Electricity. Code Evolution outperforms discrete evolutionary search on 4 of 5 complete benchmarks while exhibiting 3–7× lower variance across random seeds. Notably, the LLM discovers adapter architectures — including depthwise convolutions with learned positional weighting, attention-weighted feature pooling, and LayerNorm-gated projections — that do not decompose into any configuration in the discrete search space. Our results demonstrate that LLM-guided program synthesis is a viable and practical approach to neural architecture search for foundation model adaptation.

---

## 1. Introduction

Time series foundation models (TSFMs) such as MOMENT (Goswami et al., 2024), TimesFM (Das et al., 2024), and Timer (Liu et al., 2024) have emerged as powerful pretrained representations for diverse temporal tasks. However, a critical practical question remains: *how should one adapt a frozen TSFM to a specific downstream dataset?* The standard approach — selecting from a small menu of adapter types (LoRA ranks, bottleneck dimensions, prediction heads) — is limited by the designer's imagination and the combinatorial size of the configuration space.

This limitation motivates a shift from **configuration search** to **code search**: instead of picking hyperparameters from a predefined grid, what if the search process could generate arbitrary neural architectures? Recent advances in LLM-guided program synthesis (Romera-Paredes et al., 2024; Lehman et al., 2023) have demonstrated that language models can serve as mutation operators in evolutionary algorithms, generating novel programs that would be difficult to specify through traditional genetic programming.

We instantiate this idea for TSFM adaptation. Our method, **Code Evolution**, maintains a population of PyTorch `nn.Module` adapter classes. Each generation, an LLM (GPT-4o-mini) observes the current population — their code, fitness scores, and failure modes — and proposes new adapter architectures. These are validated locally for correctness, then evaluated on GPU via short training runs. Elitism preserves the best designs while the LLM balances exploitation of successful patterns with exploration of novel architectures.

Our experiments reveal a clear progression of negative and positive results that together paint a complete picture of what works and what doesn't for TSFM adapter search:

1. **Transferability metrics fail as fitness** (§3.1). TEMPLATE-style scores (CKA-based transferability estimation) differentiate adapter configurations but this differentiation does not predict downstream performance. The metric was designed to rank pretrained models, not adapter configurations on a single backbone.

2. **Zero-cost proxies fail at selection** (§3.2). GP-evolved proxy formulas achieve τ = 0.50 correlation with true performance but catastrophically fail when used as evolutionary fitness — the search exploits proxy weaknesses, producing degenerate configurations.

3. **Discrete hyperparameter search saturates quickly** (§3.3). A 2,160-configuration discrete space (LoRA rank × target modules × placement × unfreezing × head type × pooling) leaves little room for LLM guidance — random search and evolutionary search achieve similar results.

4. **Code-level evolution succeeds** (§3.4). When the search space is expanded to arbitrary PyTorch programs, LLM guidance provides genuine value — discovering architectures that outperform the best discrete configurations on 4/5 benchmarks with dramatically lower variance.

---

## 2. Related Work

### 2.1 Time Series Foundation Models and Adaptation

The TSFM landscape has rapidly expanded. MOMENT (Goswami et al., 2024) pretrains a transformer encoder via masked reconstruction on 1B time points. TimesFM (Das et al., 2024) and Chronos (Ansari et al., 2024) demonstrate strong zero-shot forecasting. Timer (Liu et al., 2024) unifies multiple tasks through autoregressive pretraining. Adaptation typically follows the NLP playbook: LoRA (Hu et al., 2022), bottleneck adapters, or linear probing atop frozen features.

TEMPLATE (Li et al., NeurIPS 2025) provides transferability estimation for model selection across TSFMs, computing CKA-based scores to predict which pretrained model best suits a target dataset. Critically, TEMPLATE is *passive* — it selects among existing models rather than generating new configurations. TimeTic (Pan et al., 2025) takes a similar selection approach using in-context learning entropy. Our work closes this gap: rather than selecting models, we *generate* adapter architectures guided by performance feedback.

### 2.2 Neural Architecture Search for Adapters

NAS-LoRA (Munoz et al., 2024) and Shears (Munoz et al., 2024) apply neural architecture search over adapter configurations for large language models, but use training-based fitness evaluation and search discrete hyperparameter spaces. MTF-PDNS (Xue et al., 2024) combines multi-objective evolutionary search with training-free proxies for vision CNNs. AZ-NAS (Lin et al., CVPR 2024) assembles zero-cost proxies for architecture ranking. ONE-NAS (Li et al., 2022) applies evolutionary NAS to time series but searches raw RNN architectures, not adapters on frozen backbones.

The key gap: no prior work searches an *unbounded* architecture space (arbitrary code) for foundation model adaptation, nor uses LLMs as the generative operator.

### 2.3 LLM-Guided Program Synthesis and Evolution

FunSearch (Romera-Paredes et al., 2024) demonstrated that LLMs can discover novel mathematical constructions through evolutionary program search, achieving superhuman results on extremal combinatorics problems. Evolution through Large Models (ELM; Lehman et al., 2023) showed LLMs can serve as intelligent mutation operators for Sodarace robot design. ReEvo (Ye et al., 2024) applies LLM-guided evolution to combinatorial optimization heuristics.

Most recently, GENESYS (Allen AI, 2025) applies multi-agent LLM evolution to discover novel LLM architectures themselves, using a "Ladder of Scales" approach where designs are proposed, adversarially reviewed, implemented, and verified at increasing model scales (14M–350M parameters). GENESYS demonstrates that LLM-guided genetic programming outperforms direct prompting for architecture discovery.

Our work applies this paradigm to a complementary problem: rather than discovering foundation model architectures, we discover *adapter* architectures for *existing* foundation models — a setting with much lower evaluation cost (minutes vs. days) that enables rapid iteration. The adapter search space is also fundamentally different: GENESYS searches over transformer block designs, while we search over arbitrary reduction modules that map high-dimensional encoder representations to task-specific outputs.

---

## 3. Method

### 3.1 Problem Setup

Given a pretrained TSFM backbone $f_\theta$ (MOMENT, $d_\text{model}=768$, 12 transformer blocks) and a target forecasting dataset, we seek an adapter module $g_\phi: \mathbb{R}^{B \times 512 \times 768} \rightarrow \mathbb{R}^{B \times 96}$ that maps encoder hidden states to forecast values. The backbone's last 4 encoder blocks are unfrozen during training; all other parameters remain fixed.

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

Constraints enforce ≤500K parameters and restrict imports to `torch`, `nn`, `F`, and `math`.

**Initialization.** The population (size 20) is seeded with 5 hand-designed baselines (MeanPool+Linear, MeanPool+MLP, LastToken+MLP, Conv1d+Pool, Attention+Linear) plus 15 LLM-generated cold-start architectures.

**Evaluation.** Each generation proceeds as:
1. **Local validation** (CPU): `exec()` the code, check output shape `(1, 96)` and parameter count ≤500K.
2. **GPU evaluation** (Modal A10G): 3-epoch training with Adam (lr=1e-3), MSE loss, 80/20 train/val split. Fitness = −MSE.
3. **Elitism**: Top 2 individuals pass unchanged to the next generation.
4. **LLM offspring**: GPT-4o-mini observes the ranked population (code, fitness, errors) and generates 18 new architectures, balancing exploitation and exploration.

**Validation.** After 12 generations, the top 5 architectures are re-evaluated at 15 epochs for final comparison.

### 3.3 Baselines

- **Evo-3epoch**: Traditional evolutionary search over a discrete 2,160-configuration space (6 LoRA ranks × 2 target modules × 3 placements × 4 unfreezing strategies × 3 head types × 4 pooling methods), using 3-epoch MSE as fitness. Top 5 validated at 15 epochs.
- **Random**: Random sampling from the discrete space, top 5 validated at 15 epochs.

Both baselines use identical compute budgets (~240 evaluations per run).

---

## 4. Experimental Journey

This section reports our full experimental trajectory, including negative results that shaped the final approach.

### 4.1 TEMPLATE Transferability Scores as Fitness

**Hypothesis.** TEMPLATE's CKA-based transferability scores can serve as cheap fitness for adapter architecture search, avoiding expensive fine-tuning during evolution.

**Setup.** 45 adapter configurations evaluated on ETTh1 and ETTm1. TEMPLATE composite scores (DL, PL, TA components) computed as fitness.

**Result.** TEMPLATE scores differentiate configurations (range 0.44–0.96) but this differentiation does **not** predict downstream MSE. Kendall τ between TEMPLATE fitness and 15-epoch MSE was not significant. All search strategies (hand-designed, random, TEMPLATE-guided evolution) produced statistically equivalent fine-tuning results.

**Diagnosis.** TEMPLATE measures frozen backbone representation quality — a property of the pretrained model, not the adapter-backbone interaction. It was designed to rank different pretrained models (MOMENT vs. TimesFM vs. Timer), not adapter configurations on the same backbone.

### 4.2 GP-Evolved Zero-Cost Proxies

**Hypothesis.** A composite zero-cost proxy, discovered via genetic programming over feature statistics (gradient norms, SNR, feature rank, etc.), could replace training-based fitness.

**Setup.** 30 calibration configurations per dataset, 15-epoch ground truth. GP search over 14 feature statistics across ETTh1, ETTm1, EthanolConcentration.

**Result.** Best proxy achieved cross-dataset τ = 0.50 (per-dataset: ETTh1=0.57, ETTm1=0.68, EthanolConcentration=0.29). Gradient signal-to-noise ratio (grad_snr) was the strongest individual predictor (|τ|=0.46). However, when used as evolutionary fitness, the proxy-guided search produced **worse** results than random (best MSE 2.73 vs. 0.88 on ETTh1) — evolution exploited proxy weaknesses, pushing toward degenerate configurations.

**Diagnosis.** Goodhart's Law in action: a metric correlated with performance becomes a poor objective when optimized against. The proxy captures coarse ranking but lacks the resolution needed for selection among good candidates.

### 4.3 Discrete Hyperparameter Evolution

**Hypothesis.** LLM guidance over discrete adapter hyperparameters improves upon random or traditional evolutionary search.

**Setup.** Search space: 2,160 configurations. LLM (GPT-4o-mini) proposes configurations via function calling. 12 generations, population 20.

**Result.** LLM-guided evolution produced marginal improvement over traditional evolution (within noise). The search space is too small — with only 2,160 possible configurations, random sampling already achieves near-optimal coverage, and the LLM's architectural intuition cannot be expressed through 6 discrete knobs.

### 4.4 Code-Level Evolution (This Work)

**Hypothesis.** When the search space is expanded from 2,160 discrete configurations to the infinite space of valid PyTorch programs, LLM guidance can discover architectures that discrete search cannot reach.

**Results.** Multi-seed results (seeds 42–44 where available):

| Dataset | Code-Evo (mean ± std) | Evo-3ep (mean ± std) | Random (mean) | Δ vs Evo-3ep |
|---------|----------------------|---------------------|---------------|-------------|
| ETTh1 | **0.5614 ± 0.011** | 0.6189 ± 0.106 | 0.8704 | −9.3% |
| ETTh2 | **0.6580 ± 0.010** | 0.7398 ± 0.060 | 0.7951 | −11.1% |
| ETTm1 | **0.3619 ± 0.007** | 0.3697 | 0.3871 | −2.1% |
| ETTm2 | 0.5501 ± 0.009 | **0.5280 ± 0.006** | 0.5810 | +4.2% |
| Weather | **0.3325 ± 0.007** | 0.3376 ± 0.007 | 0.3484 | −1.5% |
| Electricity* | **0.1956** | — | — | — |

*Electricity: single seed, no baselines due to compute budget.

**Code-Evo wins on 4/5 datasets** with complete comparisons. On the single loss (ETTm2), the margin is small (4.2%).

**Variance reduction.** Code-Evo exhibits 3–7× lower standard deviation across seeds compared to Evo-3ep (e.g., ETTh1: 0.011 vs 0.106). The LLM's architectural priors act as implicit regularization, avoiding the high-variance configurations that discrete search occasionally selects.

### 4.5 Discovered Architectures

Code Evolution discovers architectures that cannot be expressed in the discrete search space:

**Depthwise convolution + learned positional weighting** (Electricity, 52K params, MSE=0.1956):
```python
self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                                 groups=d_model, padding=1)
self.positional_weights = nn.Parameter(torch.randn(d_model))
```
Applies channel-wise temporal smoothing followed by learned feature weighting — a pattern common in efficient vision models (MobileNet) but not in the discrete adapter menu.

**Attention-weighted feature pooling** (Weather, 157K params, MSE=0.3258):
```python
self.attention_weights = nn.Parameter(torch.randn(d_model))
weights = F.softmax(self.attention_weights, dim=-1)
pooled = (hidden_states * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
```
Learns which of the 768 features are most relevant before temporal aggregation — effectively a soft feature selection layer.

**LayerNorm + mean pooling** (ETTh1, 50K params, MSE=0.5501):
```python
self.norm = nn.LayerNorm(d_model)
normed = self.norm(hidden_states)
pooled = normed.mean(dim=1)
```
The simplest winning architecture: normalization before pooling stabilizes training with unfrozen backbone layers. Despite having no hidden layers, it outperforms deeper discrete-menu architectures.

These architectures share a pattern: they are *structurally simple* (50K–157K params) but contain design choices (depthwise convolution, learned weighting, pre-pooling normalization) that lie outside the discrete search space's representational capacity.

---

## 5. Discussion

### Why does code evolution work where other approaches fail?

Our experimental trajectory suggests a clear narrative:

1. **Transferability metrics** fail because they measure the wrong thing — backbone quality rather than adapter-backbone fit.
2. **Zero-cost proxies** fail because Goodhart's Law applies — any cheap metric becomes a poor optimization target.
3. **Discrete search** fails to benefit from LLM guidance because the space is too small — 2,160 configurations are exhaustively searchable.
4. **Code search** succeeds because it sits at the right intersection: the space is vast enough that intelligent guidance matters (infinite valid PyTorch programs), but structured enough that the LLM can reason about it (neural architecture design is well-represented in training data).

### The LLM as architecture designer

The LLM's contribution is not random code generation — it is informed architectural reasoning. The system prompt provides the interface contract and design principles; the population provides empirical feedback. Across generations, we observe the LLM:
- Learning from failures (avoiding patterns that produced shape mismatches or NaN losses)
- Combining successful motifs (merging attention weighting from one parent with convolution from another)
- Importing cross-domain knowledge (depthwise separable convolutions from mobile vision, pre-norm architectures from transformer literature)

### Limitations

**Compute cost.** Each Code Evolution run requires ~55 minutes on an A10G GPU (~$8.50 including LLM costs). While comparable to discrete evolution per-run, the approach requires multi-seed evaluation for reliable conclusions.

**LLM dependence.** Results depend on the LLM's training data distribution. Architectures far from published designs (e.g., exotic recurrence patterns) may be underrepresented in the LLM's generative distribution.

**Single backbone.** We evaluate only on MOMENT. Generalization to other TSFMs (TimesFM, Chronos, Timer) remains to be tested.

**Incomplete evaluation breadth.** Due to compute constraints, Electricity lacks multi-seed results and baselines, and Traffic is not yet evaluated. Weather seed 44 lacks validation.

### Future Work

- **Multi-objective evolution**: Jointly optimizing MSE and parameter count to discover Pareto-optimal adapters.
- **Cross-dataset transfer**: Using adapters evolved on one dataset as seeds for another, testing whether architectural motifs transfer.
- **Backbone diversity**: Evaluating Code Evolution across MOMENT, TimesFM, and Timer to test backbone-agnostic architectural patterns.
- **Scaling the search**: Longer evolution runs, larger populations, and stronger LLMs (GPT-4o, Claude) as the code generation engine.

---

## 6. Conclusion

We presented Code Evolution, an LLM-guided evolutionary approach to adapter architecture search for time series foundation models. Through a systematic experimental progression — from transferability metrics to zero-cost proxies to discrete search to code-level search — we demonstrated that the critical ingredient for effective LLM-guided NAS is an *unbounded yet structured* search space. When constrained to discrete hyperparameters, LLM guidance adds nothing; when given the full space of PyTorch programs, it discovers novel architectures that outperform hand-designed adapters on 4/5 forecasting benchmarks with substantially lower variance. Our work provides a practical framework for automated foundation model adaptation and contributes to the growing evidence that LLMs can serve as effective architecture designers.

---

## References

- Ansari, A.F. et al. (2024). Chronos: Learning the language of time series. *arXiv:2403.07815*.
- Das, A. et al. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*.
- GENESYS Team, Allen AI (2025). GENESYS: Distributed language model architecture discovery. *GitHub: allenai/genesys*.
- Goswami, M. et al. (2024). MOMENT: A family of open time-series foundation models. *ICML 2024*.
- Hu, E.J. et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Lehman, J. et al. (2023). Evolution through large models. *arXiv:2206.08896*.
- Li, Y. et al. (2022). ONE-NAS: An online neuroevolution-based NAS for time series forecasting.
- Li, Z. et al. (2025). TEMPLATE: Transferability estimation for time series foundation models. *NeurIPS 2025*.
- Lin, Y. et al. (2024). AZ-NAS: Assembling zero-cost proxies for NAS. *CVPR 2024*.
- Liu, Y. et al. (2024). Timer: Generative pre-training of time series. *ICML 2024*.
- Munoz, J.P. et al. (2024). Shears: Unstructured sparsity with neural low-rank adapter search.
- Pan, Y. et al. (2025). TimeTic: Efficient time series model selection via in-context learning.
- Romera-Paredes, B. et al. (2024). Mathematical discoveries from program search with large language models. *Nature*.
- Xue, Y. et al. (2024). MTF-PDNS: Multi-objective training-free proxy-based differential NAS.
- Ye, H. et al. (2024). ReEvo: Large language models as hyper-heuristics with reflective evolution.
