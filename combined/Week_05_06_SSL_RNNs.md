# Week 5-6: Self-Supervised Learning & Recurrent Neural Networks

> Combined guide merging CSE 493G1/CS231n course materials with Stanford CS231n notes

## Overview

This document covers:
- Self-supervised learning paradigm
- Contrastive learning (SimCLR, MoCo)
- Recurrent Neural Networks for sequences
- Vanishing/exploding gradients
- LSTM and GRU architectures

---

## Part 1: Self-Supervised Learning

### The Label Bottleneck

```
    Supervised vs Self-Supervised
    ═════════════════════════════

    Supervised Learning:
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Image 1 │     │ Image 2 │     │ Image 3 │
    │   🐱    │     │   🐕    │     │   🚗    │
    └────┬────┘     └────┬────┘     └────┬────┘
         │               │               │
         ▼               ▼               ▼
       "cat"           "dog"           "car"
         │               │               │
         └───────────────┴───────────────┘
                         │
            Need humans to label everything!
            (expensive, doesn't scale)

    Self-Supervised Learning:
    ┌─────────────────────────────────────────┐
    │                                         │
    │   Unlabeled images (billions!)          │
    │   🖼️ 🖼️ 🖼️ 🖼️ 🖼️ 🖼️ 🖼️ 🖼️ 🖼️ 🖼️ 🖼️     │
    │                                         │
    └─────────────────────┬───────────────────┘
                          │
                Learn from the data itself!
                (create supervision from structure)
```

### Pretext Tasks

Design tasks where labels come from data structure:

```
    Common Pretext Tasks
    ════════════════════

    1. Rotation Prediction:
    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
    │  🐱 │   │ 🐱↺ │   │ 🐱↺↺│   │🐱↺↺↺│
    │ 0°  │   │ 90° │   │180° │   │270° │
    └─────┘   └─────┘   └─────┘   └─────┘
    Predict rotation → understand object orientation

    2. Jigsaw Puzzle:
    ┌───┬───┐      ┌───┬───┐
    │ 1 │ 2 │      │ 3 │ 1 │
    ├───┼───┤  →   ├───┼───┤   Predict: permutation
    │ 3 │ 4 │      │ 4 │ 2 │
    └───┴───┘      └───┴───┘

    3. Colorization:
    ┌─────────┐    ┌─────────┐
    │ Gray    │ →  │ Color   │
    │ image   │    │ image   │
    └─────────┘    └─────────┘
```

### Contrastive Learning

Learn by comparing similar and dissimilar examples:

```
    Contrastive Learning Framework
    ══════════════════════════════

    Same image, different augmentations → POSITIVE pair
    Different images → NEGATIVE pairs

                    ┌─────────────────────────────────┐
                    │         Original Image          │
                    └─────────────┬───────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
         ┌─────────┐        ┌─────────┐        ┌─────────┐
         │ Crop +  │        │ Flip +  │        │ Color   │
         │ Resize  │        │ Blur    │        │ Jitter  │
         └────┬────┘        └────┬────┘        └────┬────┘
              │                  │                   │
              ▼                  ▼                   ▼
         ┌─────────┐        ┌─────────┐        ┌─────────┐
         │Encoder f│        │Encoder f│        │Encoder f│
         └────┬────┘        └────┬────┘        └────┬────┘
              │                  │                   │
              ▼                  ▼                   ▼
             z₁                 z₂                  z₃

    Loss: Pull z₁, z₂ together (same image)
          Push z₁, z₃ apart (different images)
```

### SimCLR

**Simple Framework for Contrastive Learning** (Chen et al., 2020):

```
    SimCLR Architecture
    ═══════════════════

    1. Random augmentations (strong!)
       - Random crop + resize
       - Color distortion
       - Gaussian blur

    2. Base encoder (ResNet)
       Image → [ResNet] → h (2048-d)

    3. Projection head (MLP)
       h → [MLP] → z (128-d)

    4. Contrastive loss (NT-Xent)
       Pull augmented views of same image together
       Push different images apart

    ┌─────────┐                    ┌─────────┐
    │  Aug 1  │                    │  Aug 2  │
    └────┬────┘                    └────┬────┘
         │                              │
         ▼                              ▼
    ┌─────────┐                    ┌─────────┐
    │ ResNet  │                    │ ResNet  │
    └────┬────┘                    └────┬────┘
         │ h                            │ h
         ▼                              ▼
    ┌─────────┐                    ┌─────────┐
    │   MLP   │                    │   MLP   │
    └────┬────┘                    └────┬────┘
         │ z                            │ z
         └──────────────┬───────────────┘
                        │
                  NT-Xent Loss
```

### MoCo (Momentum Contrast)

```
    MoCo: Memory Bank for More Negatives
    ════════════════════════════════════

    Problem with SimCLR: Need large batch for many negatives

    MoCo Solution: Maintain a queue of past embeddings

    ┌─────────┐      ┌─────────────────────────────────┐
    │  Query  │      │     Memory Queue (K negatives)  │
    │ Encoder │      │  ┌───┬───┬───┬───┬───┬───┬───┐ │
    │   f_q   │      │  │ k₁│ k₂│ k₃│...│...│...│kₖ │ │
    └────┬────┘      │  └───┴───┴───┴───┴───┴───┴───┘ │
         │           └─────────────────────────────────┘
         ▼                          ▲
        q ─────── compare ──────────┘

    Key Encoder f_k updated via momentum:
    θ_k = m × θ_k + (1-m) × θ_q    (m = 0.999)

    Benefits:
    - Large effective batch size (65536 negatives)
    - Memory efficient
    - Consistent keys via momentum
```

### Transfer Learning with SSL

```
    SSL Training → Downstream Tasks
    ═══════════════════════════════

    Phase 1: Pre-training (unlabeled data)
    ┌─────────────────────────────────────┐
    │  Millions of unlabeled images       │
    │  → Contrastive learning             │
    │  → Learn general representations    │
    └─────────────────────────────────────┘
                    │
                    ▼
    Phase 2: Fine-tuning (labeled data)
    ┌─────────────────────────────────────┐
    │  Small labeled dataset              │
    │  → Fine-tune encoder + new head     │
    │  → Excellent performance!           │
    └─────────────────────────────────────┘

    SSL representations often match or beat
    ImageNet supervised pre-training!
```

---

## Part 2: Recurrent Neural Networks

### Why Sequences?

```
    Sequence Problems
    ═════════════════

    One-to-One:    Fixed input → Fixed output
    (Standard NN)  [Image] → [Label]

    One-to-Many:   Fixed input → Sequence output
    (Captioning)   [Image] → [The] [cat] [sat] [down]

    Many-to-One:   Sequence input → Fixed output
    (Sentiment)    [This] [movie] [was] [great] → [Positive]

    Many-to-Many:  Sequence → Sequence
    (Translation)  [Bonjour] [monde] → [Hello] [world]

    ┌─────┐    ┌───┬───┬───┐    ┌───┬───┬───┐    ┌───┬───┬───┐
    │     │    │   │   │   │    │   │   │   │    │   │   │   │
    │  ●  │    │ ● │ ● │ ● │    │ ● │ ● │ ● │    │ ● │ ● │ ● │
    │     │    │   │   │   │    │   │   │   │    │   │   │   │
    └──┬──┘    └───┴───┴───┘    └───┴───┴───┘    └───┴───┴───┘
       │            │                │                │
       ▼            ▼                ▼                ▼
    ┌─────┐    ┌─────────┐      ┌─────────┐      ┌───┬───┬───┐
    │  ●  │    │    ●    │      │ ● │ ● │ ● │    │ ● │ ● │ ● │
    └─────┘    └─────────┘      └───┴───┴───┘    └───┴───┴───┘
    1-to-1      1-to-many        many-to-1       many-to-many
```

### RNN Core Idea

Process sequences by maintaining a hidden state:

```
    RNN: Recurrent Computation
    ══════════════════════════

    hₜ = f_W(hₜ₋₁, xₜ)

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │    x₁        x₂        x₃        x₄                │
    │    │         │         │         │                  │
    │    ▼         ▼         ▼         ▼                  │
    │  ┌───┐     ┌───┐     ┌───┐     ┌───┐               │
    │  │RNN│────▶│RNN│────▶│RNN│────▶│RNN│               │
    │  └─┬─┘     └─┬─┘     └─┬─┘     └─┬─┘               │
    │    │         │         │         │                  │
    │    ▼         ▼         ▼         ▼                  │
    │    y₁        y₂        y₃        y₄                │
    │                                                     │
    │    h₀ ──▶ h₁ ──▶ h₂ ──▶ h₃ ──▶ h₄                 │
    │                                                     │
    │    Same weights W used at every timestep!           │
    └─────────────────────────────────────────────────────┘
```

### Vanilla RNN Equations

```
    Vanilla RNN Computation
    ═══════════════════════

    Hidden state update:
    hₜ = tanh(W_hh × hₜ₋₁ + W_xh × xₜ + b_h)

    Output:
    yₜ = W_hy × hₜ + b_y

    Matrix dimensions:
    - xₜ: [input_dim]
    - hₜ: [hidden_dim]
    - W_xh: [hidden_dim × input_dim]
    - W_hh: [hidden_dim × hidden_dim]
    - W_hy: [output_dim × hidden_dim]
```

```python
class VanillaRNN:
    def step(self, x, h_prev):
        # Single timestep forward pass
        h = np.tanh(np.dot(self.W_hh, h_prev) +
                    np.dot(self.W_xh, x) + self.b_h)
        y = np.dot(self.W_hy, h) + self.b_y
        return h, y
```

### Backpropagation Through Time (BPTT)

```
    BPTT: Unroll and Backprop
    ═════════════════════════

    Forward (unrolled):
    x₁ ──▶ [h₁] ──▶ [h₂] ──▶ [h₃] ──▶ [h₄]
             │        │        │        │
             ▼        ▼        ▼        ▼
            L₁       L₂       L₃       L₄

    Backward (chain rule through time):
    ∂L/∂W = Σₜ ∂Lₜ/∂W

    ∂L₄/∂h₁ = ∂L₄/∂h₄ × ∂h₄/∂h₃ × ∂h₃/∂h₂ × ∂h₂/∂h₁
                        └──────────────────────────┘
                        Product of many Jacobians!
```

### Vanishing/Exploding Gradients

```
    The Gradient Problem
    ════════════════════

    Gradient at time t depends on product:
    ∂hₜ/∂h₁ = ∏ᵢ ∂hᵢ₊₁/∂hᵢ

    Each term involves W_hh and tanh derivative:
    ∂hₜ₊₁/∂hₜ = W_hh × diag(1 - tanh²(hₜ))

    Problem:
    - If max eigenvalue of W_hh > 1: EXPLODING gradients
    - If max eigenvalue of W_hh < 1: VANISHING gradients

    ┌─────────────────────────────────────────┐
    │  Gradient magnitude over time           │
    │                                         │
    │  │                   ╱ Exploding        │
    │  │                 ╱                    │
    │  │               ╱                      │
    │  │─────────────────────── Ideal         │
    │  │               ╲                      │
    │  │                 ╲                    │
    │  │                   ╲ Vanishing        │
    │  └───────────────────────────────▶      │
    │              timesteps                  │
    └─────────────────────────────────────────┘

    Result: Can't learn long-range dependencies!
```

---

## Part 3: LSTM (Long Short-Term Memory)

### The LSTM Solution

Introduce a **cell state** as a highway for information:

```
    LSTM vs Vanilla RNN
    ═══════════════════

    Vanilla RNN:
    hₜ = tanh(W × [hₜ₋₁, xₜ])
    (information must pass through tanh every step)

    LSTM:
    - Cell state Cₜ: long-term memory highway
    - Hidden state hₜ: short-term memory
    - Gates control information flow

    ┌─────────────────────────────────────────────────────┐
    │                    Cell State (Highway)             │
    │    ═══════════════════════════════════════════▶     │
    │         ↑           ↑           ↑                   │
    │         │           │           │                   │
    │      Forget       Input      Output                 │
    │       Gate         Gate        Gate                 │
    │         ↑           ↑           ↑                   │
    │    ─────┴───────────┴───────────┴─────▶            │
    │                Hidden State                         │
    └─────────────────────────────────────────────────────┘
```

### LSTM Gates

```
    LSTM Gate Operations
    ════════════════════

    Input: hₜ₋₁, xₜ
    Combined: [hₜ₋₁, xₜ]

    1. FORGET GATE: What to remove from cell state
       fₜ = σ(W_f × [hₜ₋₁, xₜ] + b_f)

    2. INPUT GATE: What new info to store
       iₜ = σ(W_i × [hₜ₋₁, xₜ] + b_i)
       C̃ₜ = tanh(W_C × [hₜ₋₁, xₜ] + b_C)

    3. CELL UPDATE: Update cell state
       Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ

    4. OUTPUT GATE: What to output
       oₜ = σ(W_o × [hₜ₋₁, xₜ] + b_o)
       hₜ = oₜ ⊙ tanh(Cₜ)

    Key: Cell state can flow unchanged (gradient highway!)
```

### LSTM Diagram

```
    LSTM Cell Architecture
    ══════════════════════

                    Cₜ₋₁                               Cₜ
    ═══════════════════════╗         ╔═══════════════════════▶
                           ║    +    ║
                           ▼    ↑    ▼
                        ┌──────┴──────┐
                 fₜ ───▶│      ×      │◀─── iₜ × C̃ₜ
                        └─────────────┘
                              ▲
                              │
    ┌─────────────────────────┴─────────────────────────┐
    │                                                   │
    │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐          │
    │  │  σ  │   │  σ  │   │tanh │   │  σ  │          │
    │  │ fₜ  │   │ iₜ  │   │ C̃ₜ  │   │ oₜ  │          │
    │  └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘          │
    │     │         │         │         │              │
    │     └─────────┴─────────┴─────────┴──────────┐   │
    │                                              │   │
    │  ┌───────────────────────────────────────────┴─┐ │
    │  │           [hₜ₋₁, xₜ]                        │ │
    │  └─────────────────────────────────────────────┘ │
    │         ▲                          ▲             │
    └─────────│──────────────────────────│─────────────┘
              │                          │
             hₜ₋₁                        xₜ
```

### GRU (Gated Recurrent Unit)

Simpler alternative to LSTM:

```
    GRU: Simplified Gating
    ══════════════════════

    Only 2 gates (vs 3 in LSTM):
    - Reset gate rₜ
    - Update gate zₜ

    Equations:
    rₜ = σ(W_r × [hₜ₋₁, xₜ])        Reset gate
    zₜ = σ(W_z × [hₜ₋₁, xₜ])        Update gate
    h̃ₜ = tanh(W × [rₜ ⊙ hₜ₋₁, xₜ])  Candidate
    hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ    Output

    No separate cell state!
    Often performs similarly to LSTM with fewer params
```

---

## Part 4: Applications

### Image Captioning

```
    Image Captioning Architecture
    ═════════════════════════════

    ┌─────────────┐
    │             │
    │   Image     │
    │             │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    CNN      │
    │  Encoder    │
    └──────┬──────┘
           │ features
           ▼
    ┌─────────────┐
    │   LSTM      │──▶ "A"
    │  Decoder    │──▶ "cat"
    │             │──▶ "sitting"
    └─────────────┘──▶ "on"
                   ──▶ "mat"
                   ──▶ <END>
```

### Sequence-to-Sequence

```
    Encoder-Decoder Architecture
    ════════════════════════════

    Input:  "Hello world"
    Output: "Bonjour monde"

    ┌─────────────────────────────────────────────────┐
    │ ENCODER                                         │
    │                                                 │
    │ "Hello" ──▶ [LSTM] ──▶ [LSTM] ◀── "world"     │
    │                          │                      │
    │                    context vector               │
    └──────────────────────────┼──────────────────────┘
                               │
                               ▼
    ┌──────────────────────────┼──────────────────────┐
    │ DECODER                  │                      │
    │                          ▼                      │
    │             [LSTM] ──▶ [LSTM] ──▶ [LSTM]       │
    │               │          │          │           │
    │               ▼          ▼          ▼           │
    │          "Bonjour"    "monde"    <END>         │
    └─────────────────────────────────────────────────┘
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **Self-Supervised Learning** | Learn from unlabeled data via pretext tasks |
| **Contrastive Learning** | Pull similar, push different embeddings |
| **SimCLR** | Strong augmentations + NT-Xent loss |
| **MoCo** | Momentum encoder + memory queue |
| **RNN** | Hidden state carries sequential information |
| **Vanishing Gradients** | Long-range dependencies hard to learn |
| **LSTM** | Cell state highway + gates for gradient flow |
| **GRU** | Simpler 2-gate alternative to LSTM |

---

## References

**Self-Supervised Learning:**
- SimCLR: Chen et al., "A Simple Framework for Contrastive Learning", 2020
- MoCo: He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", 2020
- BYOL: Grill et al., "Bootstrap Your Own Latent", 2020

**RNN/LSTM:**
- [CS231n RNN Notes](https://cs231n.github.io/rnn/)
- Hochreiter & Schmidhuber, "Long Short-Term Memory", 1997
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder", 2014 (GRU)
