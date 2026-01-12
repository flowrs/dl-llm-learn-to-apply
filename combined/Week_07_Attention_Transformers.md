# Week 7: Attention and Transformers

> Combined guide merging CSE 493G1/CS231n course materials with Stanford CS231n notes

## Overview

This document covers:
- Attention mechanism fundamentals
- Self-attention and cross-attention
- Transformer architecture
- Vision Transformers (ViT)
- Encoder-only vs decoder-only architectures

---

## Part 1: The Attention Mechanism

### Why Attention?

```
    RNN Bottleneck Problem
    ══════════════════════

    Encoder-Decoder with RNN:

    "The cat sat on the mat" → [Encoder] → context → [Decoder] → "Le chat..."

    Problem: Everything compressed into ONE context vector!

    ┌─────────────────────────────────────────────────────┐
    │  "The" "cat" "sat" "on" "the" "mat"                │
    │    │     │     │     │     │     │                  │
    │    ▼     ▼     ▼     ▼     ▼     ▼                  │
    │  [RNN]→[RNN]→[RNN]→[RNN]→[RNN]→[RNN]               │
    │                                   │                 │
    │                              [context]  ← bottleneck│
    │                                   │                 │
    │                                   ▼                 │
    │  [RNN]→[RNN]→[RNN]→[RNN]→[RNN]→[RNN]               │
    │    │     │     │     │     │     │                  │
    │    ▼     ▼     ▼     ▼     ▼     ▼                  │
    │   "Le" "chat" "était" "assis" "sur" "le"           │
    └─────────────────────────────────────────────────────┘
```

### Attention Solution

Allow decoder to look at ALL encoder states:

```
    Attention: Dynamic Context
    ══════════════════════════

    Instead of one context vector, compute weighted sum
    of all encoder states at each decoder step:

    Encoder states: h₁, h₂, h₃, h₄, h₅, h₆

    For decoder at position t:
    1. Compute alignment scores: eₜᵢ = f(sₜ₋₁, hᵢ)
    2. Normalize: αₜᵢ = softmax(eₜᵢ)
    3. Context: cₜ = Σᵢ αₜᵢ × hᵢ

    "Le" attends mostly to "The"
    "chat" attends mostly to "cat"
    etc.
```

### Query, Key, Value

The modern formulation of attention:

```
    Attention(Q, K, V)
    ══════════════════

    Query (Q): What am I looking for?
    Key (K):   What do I contain?
    Value (V): What do I output?

    Intuition: Database lookup

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Query: "What is the capital of France?"            │
    │                                                     │
    │  Database:                                          │
    │  ┌─────────────────────────────────────────────┐   │
    │  │ Key: "capital of France"  → Value: "Paris"  │   │ ← Match!
    │  │ Key: "capital of Germany" → Value: "Berlin" │   │
    │  │ Key: "population France"  → Value: "67M"    │   │
    │  └─────────────────────────────────────────────┘   │
    │                                                     │
    │  Output: "Paris" (high attention to matching key)   │
    └─────────────────────────────────────────────────────┘
```

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```
    Attention Computation
    ═════════════════════

    Q: [n × d]  (n queries, d dimensions)
    K: [m × d]  (m keys)
    V: [m × d]  (m values)

    Step 1: Compute similarities
    ┌─────┐   ┌─────┐ᵀ   ┌─────────┐
    │  Q  │ × │  K  │ = │  QKᵀ    │   [n × m]
    │     │   │     │   │ scores  │
    └─────┘   └─────┘   └─────────┘

    Step 2: Scale by √d (for stable gradients)
    scores = scores / √d

    Step 3: Softmax (per query)
    α = softmax(scores)    [n × m], rows sum to 1

    Step 4: Weighted sum of values
    ┌─────────┐   ┌─────┐   ┌─────────┐
    │    α    │ × │  V  │ = │ output  │   [n × d]
    └─────────┘   └─────┘   └─────────┘
```

---

## Part 2: Self-Attention

### Self-Attention: Attend to Yourself

```
    Self-Attention
    ══════════════

    Q, K, V all come from the SAME sequence:

    Input: x₁, x₂, x₃, x₄

    ┌───────────────────────────────────────────────────┐
    │                                                   │
    │    x₁      x₂      x₃      x₄                    │
    │    │       │       │       │                      │
    │    ▼       ▼       ▼       ▼                      │
    │  ┌───┐   ┌───┐   ┌───┐   ┌───┐                   │
    │  │W_q│   │W_q│   │W_q│   │W_q│  → q₁,q₂,q₃,q₄   │
    │  │W_k│   │W_k│   │W_k│   │W_k│  → k₁,k₂,k₃,k₄   │
    │  │W_v│   │W_v│   │W_v│   │W_v│  → v₁,v₂,v₃,v₄   │
    │  └───┘   └───┘   └───┘   └───┘                   │
    │                                                   │
    │    Each position can attend to ALL positions      │
    │                                                   │
    └───────────────────────────────────────────────────┘

    Output for position i = Σⱼ αᵢⱼ × vⱼ
```

### Multi-Head Attention

Multiple attention patterns in parallel:

```
    Multi-Head Attention
    ════════════════════

    Why? Different heads can attend to different things:
    - Head 1: syntactic relationships
    - Head 2: semantic similarity
    - Head 3: positional patterns

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Input X                                            │
    │    │                                                │
    │    ├──────────┬──────────┬──────────┐              │
    │    ▼          ▼          ▼          ▼              │
    │  ┌────┐    ┌────┐    ┌────┐    ┌────┐             │
    │  │Head│    │Head│    │Head│    │Head│             │
    │  │ 1  │    │ 2  │    │ 3  │    │... │             │
    │  └──┬─┘    └──┬─┘    └──┬─┘    └──┬─┘             │
    │     │         │         │         │                │
    │     └────┬────┴────┬────┴────┬────┘                │
    │          │         │         │                      │
    │          ▼         ▼         ▼                      │
    │       [Concatenate all heads]                       │
    │                    │                                │
    │                    ▼                                │
    │              [Linear W_o]                           │
    │                    │                                │
    │                    ▼                                │
    │               Output                                │
    └─────────────────────────────────────────────────────┘

    MultiHead(Q,K,V) = Concat(head₁,...,headₕ) × W_o
    where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)
```

---

## Part 3: The Transformer

### Architecture Overview

```
    Transformer Architecture
    ════════════════════════

    ┌─────────────────────────────────────────────────────┐
    │                    ENCODER                          │
    │  ┌───────────────────────────────────────────────┐ │
    │  │  Input Embedding + Positional Encoding        │ │
    │  └───────────────────────────────────────────────┘ │
    │                        │                            │
    │                        ▼                            │
    │  ┌───────────────────────────────────────────────┐ │
    │  │         Multi-Head Self-Attention             │ │
    │  │              + Add & Norm                     │ │
    │  └───────────────────────────────────────────────┘ │
    │                        │                            │
    │                        ▼                            │
    │  ┌───────────────────────────────────────────────┐ │
    │  │            Feed Forward Network               │ │
    │  │              + Add & Norm                     │ │
    │  └───────────────────────────────────────────────┘ │
    │                        │                            │
    │                   × N layers                        │
    │                        │                            │
    └────────────────────────┼────────────────────────────┘
                             │
    ┌────────────────────────┼────────────────────────────┐
    │                    DECODER                          │
    │  ┌───────────────────────────────────────────────┐ │
    │  │  Output Embedding + Positional Encoding       │ │
    │  └───────────────────────────────────────────────┘ │
    │                        │                            │
    │                        ▼                            │
    │  ┌───────────────────────────────────────────────┐ │
    │  │     Masked Multi-Head Self-Attention          │ │
    │  │              + Add & Norm                     │ │
    │  └───────────────────────────────────────────────┘ │
    │                        │                            │
    │                        ▼                            │
    │  ┌───────────────────────────────────────────────┐ │
    │  │     Multi-Head Cross-Attention                │ │
    │  │     (attends to encoder output)               │ │
    │  │              + Add & Norm                     │ │
    │  └───────────────────────────────────────────────┘ │
    │                        │                            │
    │                        ▼                            │
    │  ┌───────────────────────────────────────────────┐ │
    │  │            Feed Forward Network               │ │
    │  │              + Add & Norm                     │ │
    │  └───────────────────────────────────────────────┘ │
    │                        │                            │
    │                   × N layers                        │
    │                        │                            │
    │                        ▼                            │
    │  ┌───────────────────────────────────────────────┐ │
    │  │              Linear + Softmax                 │ │
    │  └───────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────┘
```

### Positional Encoding

Transformers have no built-in position awareness:

```
    Positional Encoding
    ═══════════════════

    Without position info: "cat sat mat" = "mat sat cat"

    Solution: Add position information to embeddings

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Properties:
    - Unique encoding for each position
    - Can extrapolate to longer sequences
    - Relative positions can be computed

    Visualization:
    ┌─────────────────────────────────────────┐
    │  Pos 0: [sin(0), cos(0), sin(0), ...]  │
    │  Pos 1: [sin(1/f), cos(1/f), ...]      │
    │  Pos 2: [sin(2/f), cos(2/f), ...]      │
    │  ...                                    │
    └─────────────────────────────────────────┘
```

### Masked Self-Attention

For autoregressive generation, prevent looking at future:

```
    Masked Attention (Decoder)
    ══════════════════════════

    Generating: "The cat sat"

    At position 3 ("sat"):
    - CAN attend to: "The", "cat", "sat"
    - CANNOT attend to: future tokens

    Attention mask:
    ┌────────────────────┐
    │     The cat sat    │
    │ The  ✓   ✗   ✗     │
    │ cat  ✓   ✓   ✗     │
    │ sat  ✓   ✓   ✓     │
    └────────────────────┘

    Implementation: Add -∞ to masked positions before softmax
```

### Transformer Advantages

| Property | RNN | Transformer |
|----------|-----|-------------|
| Parallelization | Sequential | Fully parallel |
| Long-range deps | Difficult | Direct attention |
| Training speed | Slow | Fast |
| Memory | O(1) per step | O(n²) attention |

---

## Part 4: Vision Transformer (ViT)

### Applying Transformers to Images

```
    Vision Transformer Architecture
    ═══════════════════════════════

    1. Split image into patches
    ┌───┬───┬───┬───┐
    │ 1 │ 2 │ 3 │ 4 │
    ├───┼───┼───┼───┤     16×16 patches
    │ 5 │ 6 │ 7 │ 8 │     from 224×224 image
    ├───┼───┼───┼───┤     = 196 patches
    │ 9 │10 │11 │12 │
    ├───┼───┼───┼───┤
    │13 │14 │15 │16 │
    └───┴───┴───┴───┘

    2. Flatten patches into sequence
    [P₁, P₂, P₃, ..., P₁₉₆]

    3. Linear projection + position embeddings
    [E₁, E₂, E₃, ..., E₁₉₆]

    4. Add [CLS] token for classification
    [[CLS], E₁, E₂, ..., E₁₉₆]

    5. Standard Transformer encoder
    ┌─────────────────────────────────┐
    │  Multi-Head Self-Attention × L  │
    │  Feed Forward Network           │
    └─────────────────────────────────┘

    6. Classify using [CLS] output
    [CLS] output → MLP → class prediction
```

### ViT vs CNN

```
    ViT vs CNN Properties
    ═════════════════════

    CNN:
    ✓ Built-in spatial inductive bias
    ✓ Works well with less data
    ✓ Translation equivariant
    ✗ Local receptive fields (limited range)

    ViT:
    ✓ Global receptive field from layer 1
    ✓ Scales better with data/compute
    ✗ Needs more data (or pre-training)
    ✗ No built-in spatial structure

    When to use:
    - Small data: CNN
    - Large data/pre-training: ViT
    - Hybrid: Use CNN for patches, then Transformer
```

---

## Part 5: Architecture Variants

### Encoder-Only (BERT-style)

```
    Encoder-Only Architecture
    ═════════════════════════

    Use case: Understanding, classification, embeddings

    ┌─────────────────────────────────────────┐
    │  Input: [CLS] The cat sat [MASK] mat    │
    │                    │                    │
    │                    ▼                    │
    │  ┌─────────────────────────────────┐   │
    │  │    Bidirectional Attention      │   │
    │  │    (see all tokens)             │   │
    │  └─────────────────────────────────┘   │
    │                    │                    │
    │                    ▼                    │
    │  Output: embeddings for all positions   │
    │                                         │
    │  Examples: BERT, RoBERTa, ViT           │
    └─────────────────────────────────────────┘
```

### Decoder-Only (GPT-style)

```
    Decoder-Only Architecture
    ═════════════════════════

    Use case: Text generation, language modeling

    ┌─────────────────────────────────────────┐
    │  Input: The cat sat on                  │
    │                    │                    │
    │                    ▼                    │
    │  ┌─────────────────────────────────┐   │
    │  │    Causal (Masked) Attention    │   │
    │  │    (only see past tokens)       │   │
    │  └─────────────────────────────────┘   │
    │                    │                    │
    │                    ▼                    │
    │  Output: predict next token "the"       │
    │                                         │
    │  Examples: GPT, LLaMA, Claude           │
    └─────────────────────────────────────────┘

    Why decoder-only dominates for LLMs:
    - Simple: one architecture for everything
    - Efficient: KV cache for generation
    - Scalable: proven at 100B+ parameters
```

### KV Cache for Efficient Inference

```
    KV Cache Optimization
    ═════════════════════

    Without cache: recompute K,V for all tokens each step

    With cache: store K,V, only compute for new token

    Step 1: "The"
    K = [k₁], V = [v₁]                    cache: K,V

    Step 2: "The cat"
    K = [k₁, k₂], V = [v₁, v₂]            append k₂,v₂

    Step 3: "The cat sat"
    K = [k₁, k₂, k₃], V = [v₁, v₂, v₃]    append k₃,v₃

    Speedup: O(n²) → O(n) per token!
```

---

## Summary

| Concept | Key Points |
|---------|------------|
| **Attention** | Weighted sum based on query-key similarity |
| **Self-Attention** | Q, K, V from same sequence |
| **Multi-Head** | Multiple attention patterns in parallel |
| **Transformer** | Self-attention + FFN + residuals + LayerNorm |
| **Positional Encoding** | Inject position information |
| **Masked Attention** | Causal: only attend to past |
| **ViT** | Image patches as tokens |
| **Encoder-only** | Bidirectional, for understanding (BERT) |
| **Decoder-only** | Causal, for generation (GPT) |

---

## References

**Foundational Papers:**
- Vaswani et al., "Attention Is All You Need", 2017
- Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020 (ViT)
- Devlin et al., "BERT", 2018
- Radford et al., "GPT", 2018

**Course Materials:**
- [CS231n: Attention](https://cs231n.github.io/attention/)
- [CS231n: Transformers](https://cs231n.github.io/transformers/)
