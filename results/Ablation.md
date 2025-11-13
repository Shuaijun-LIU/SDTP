# Phase E: SDTP Ablation Study

**Note:** This document records the design of ablation experiments for future work, but is not in the current task list.

## E1. Ablation Purpose

SDTP's core innovations include:

1. Learnable token pruning module (可学习剪枝器)
2. Saliency-based supervision + ranking loss
3. Layer-wise dynamic pruning scheme
4. RoPE-compatible shrinking strategy
5. Head/tail token preserving strategy (前 4 + 后 16)
6. Multi-layer pruning schedule (10 layers)
7. Keep ratio 0.7, dynamic token length updates

The purpose of ablation studies is to prove that each component contributes to performance or speed, and removing any component will degrade the results.

---

## E2. Ablation Points by Category

**Legend:**
- **Performance**: Ablation points with significant impact on performance
- **Speed**: Ablation points with noticeable impact on speed
- **Analysis**: Ablation points for analyzing internal mechanisms (e.g., ranking, supervision methods)

---

### ① Pruning Capability (Core)

| Module | Ablation Method |
|--------|----------------|
| **Performance** Learnable Pruner | Replace with saliency-only baseline (remove MLP) |
| **Performance** Learnable Pruner | Replace with random pruning (keep pruning ratio unchanged) |
| **Performance** Layer-wise pruning | Replace with single-layer pruning (e.g., only prune layer 4) |
| **Performance** Layer-wise pruning | Replace with fixed layer set (e.g., fixed 3 layers) |
| **Speed** Keep ratio | Change from 0.7 to 0.8 / 0.5 / 0.3 |
| **Speed** Head/Tail Preserve | Remove "first 4 / last 16" preservation strategy |
| **Speed** RoPE Fix | Remove RoPE index fix → compare errors/performance degradation |

---

### ② Loss Function Related

| Module | Ablation Method |
|--------|----------------|
| **Analysis** Ranking Loss | Remove, keep only MSE + LM loss |
| **Analysis** MSE Loss | Remove saliency regression, use only ranking |
| **Analysis** LM loss | Remove LM loss (supervise only importance) |
| **Performance** Full supervision | Use only LM loss (no saliency) |

These experiments can verify the paper's claim that "ranking loss → improves ranking quality".

---

### ③ Inference Strategy Related

| Module | Ablation Method |
|--------|----------------|
| **Speed** Token selection | Use Top-k vs Top-p vs threshold-based |
| **Speed** Token merge | Do not update attention mask → observe error accumulation |
| **Speed** Multi-layer mask propagation | Use hard mask vs soft mask |

---

### ④ Multi-GPU / Communication Related (Focus on Speed)

| Module | Ablation Method |
|--------|----------------|
| **Speed** Multi-GPU pruning | Remove cross-GPU pre-pruning (speed decreases) |
| **Speed** Communication cost | Prune only in first 2 layers or last 2 layers (compare communication load changes) |

---

## E3. Ablation Experiment Matrix

**Table: SDTP Ablation Design Matrix**

| Ablation ID | Description | Variable Changed | Hypothesis | Expected Effect |
|-------------|-------------|------------------|------------|-----------------|
| A1 | Remove Pruner → Use saliency baseline | model=saliency-only | Learnable pruner performance > saliency | Performance ↓, Speed ≈ |
| A2 | Random pruning | importance=random | SDTP must learn ranking | Performance ↓↓ |
| A3 | No ranking loss | loss=LM+MSE | Ranking loss is important | Performance ↓ (moderate) |
| A4 | No MSE loss | loss=LM+Ranking | Regression alignment is critical | Performance ↓ (moderate) |
| A5 | Only LM loss (no saliency) | loss=LM | No supervision is weaker | Performance ↓ (significant) |
| A6 | Remove head/tail Preserve | no head/tail | Preservation strategy ensures semantic stability | Performance ↓ |
| A7 | Keep ratio 0.5 | ratio=0.5 | Over-pruning | Performance ↓, Speed ↑ |
| A8 | Keep ratio 0.3 | ratio=0.3 | Extreme pruning | Performance ↓↓, Speed ↑↑ |
| A9 | Keep ratio 0.8 | ratio=0.8 | Under-pruning | Performance ≈, Speed ↓ |
| B1 | Single-layer pruning | prune only L4 | Multi-layer is more effective | Performance ↓, Speed ↓ |
| B2 | No RoPE index fix | rope broken | Must fix | Model crashes |
| C1 | Soft mask only | no hard mask | Soft pruning accuracy higher or lower | Small change |
| C2 | No mask propagation | per-layer local | Upper layers will fail | Model instability |

You can combine 6–8 key ablation experiments for the final paper experiments.

---

## E4. Paper-Level Ablation Control Groups (Clear, Academic)

Each ablation experiment needs to compare:

### Baseline (Standard SDTP)

- Learnable pruning module (MLP)
- Loss: LM + MSE + Ranking
- Keep ratio = 0.7
- Head 4 + tail 16 preservation
- Layer-wise pruning (10 layers)
- RoPE index correction
- Attention mask and hidden state cascade updates

### Control Group (e.g., No-Ranking Loss)

**Ablation Settings:**
- Remove ranking loss
- Keep all other components unchanged

**Final Report Writing:**

"We isolate the contribution of the ranking loss by disabling it while keeping all other components fixed. This setting evaluates the effect of supervising the ordering of saliency scores."

---

## E5. Ablation Section (Paper Structure Template)

Below is a standard template based on top-tier conference writing style. You only need to fill in the results:

---

### 6. Ablation Studies

To better understand the contribution of each SDTP component, we conduct a comprehensive ablation study covering the pruning mechanism, supervision strategy, and inference-time pruning policy.

#### 6.1 Effect of the Learnable Pruner

(A1, A2 results)

#### 6.2 Contribution of Each Loss Component

(A3–A5 results)

#### 6.3 Influence of Pruning Ratio

(A7–A9 results)

#### 6.4 Role of Head/Tail Preservation

(A6 results)

#### 6.5 RoPE Index Correction Is Essential

(B2 results)

#### 6.6 Multi-Layer vs Single-Layer Pruning

(B1 results)

#### 6.7 Communication Reduction in Multi-GPU Inference

(C1–C2 + multi-GPU results)
