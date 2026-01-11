# Week 5-6: Self-Supervised Learning & Recurrent Neural Networks
## From Novice to Practitioner: Learning Without Labels and Sequential Data

---

## Table of Contents
1. [Self-Supervised Learning](#self-supervised-learning)
2. [Contrastive Learning](#contrastive-learning)
3. [Recurrent Neural Networks](#recurrent-neural-networks)
4. [Vanishing Gradients](#vanishing-gradients)
5. [LSTM Networks](#lstm-networks)
6. [GRU Networks](#gru-networks)
7. [Sequence-to-Sequence](#sequence-to-sequence)
8. [Image Captioning](#image-captioning)
9. [Coding Exercises](#coding-exercises)
10. [Business Applications](#business-applications)

---

## Self-Supervised Learning

### The Labeling Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THE LABELING BOTTLENECK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SUPERVISED LEARNING:                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Image ──► Human Labels ──► Model Training ──► Good Features      │  │
│   │                  │                                                  │  │
│   │                  ▼                                                  │  │
│   │   PROBLEMS:                                                        │  │
│   │   • ImageNet: 14M images, $25K+ to label                          │  │
│   │   • Medical: Requires expert doctors ($$$)                        │  │
│   │   • Time: Months to create large datasets                          │  │
│   │   • Errors: Human labeling is inconsistent                         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   SELF-SUPERVISED LEARNING:                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Image ──► CREATE labels FROM data ──► Pretrain ──► Good Features │  │
│   │                      │                                              │  │
│   │                      ▼                                              │  │
│   │   The data ITSELF provides supervision!                            │  │
│   │   • Predict missing patches                                        │  │
│   │   • Predict image rotations                                        │  │
│   │   • Match augmented views                                          │  │
│   │                                                                     │  │
│   │   Benefits:                                                        │  │
│   │   • Unlimited "free" training data                                 │  │
│   │   • Better features than supervised (often!)                       │  │
│   │   • Transfer to many downstream tasks                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   DATA AVAILABILITY:                                                       │
│                                                                             │
│   Unlabeled Data ████████████████████████████████████████ (billions)      │
│   Labeled Data   ██ (millions)                                             │
│                                                                             │
│   Why waste 99.9% of available data?                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pretext Tasks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRETEXT TASKS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IDEA: Create supervised learning tasks from unlabeled data               │
│   The model learns useful features to solve these "proxy" tasks            │
│                                                                             │
│                                                                             │
│   1. IMAGE ROTATION PREDICTION:                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                               │  │
│   │   │  🐱 │  │ 🐱  │  │  🐱 │  │ 🐱  │                               │  │
│   │   │     │  │    ◄│  │  ▼  │  │►    │                               │  │
│   │   └─────┘  └─────┘  └─────┘  └─────┘                               │  │
│   │     0°       90°      180°     270°                                │  │
│   │                                                                     │  │
│   │   Task: Predict which rotation was applied                         │  │
│   │   Model must understand "up" vs "down" → learns object structure  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   2. JIGSAW PUZZLE:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Original:         Shuffled:           Task:                      │  │
│   │   ┌───┬───┬───┐     ┌───┬───┬───┐      Predict                    │  │
│   │   │ 1 │ 2 │ 3 │     │ 5 │ 3 │ 1 │      correct                    │  │
│   │   ├───┼───┼───┤     ├───┼───┼───┤      arrangement               │  │
│   │   │ 4 │ 5 │ 6 │  →  │ 2 │ 9 │ 4 │      (1 of 9!                   │  │
│   │   ├───┼───┼───┤     ├───┼───┼───┤      permutations)              │  │
│   │   │ 7 │ 8 │ 9 │     │ 8 │ 6 │ 7 │                                 │  │
│   │   └───┴───┴───┘     └───┴───┴───┘                                 │  │
│   │                                                                     │  │
│   │   Model must understand spatial relationships                      │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   3. CONTEXT PREDICTION (Inpainting):                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input:              Target:           Model predicts             │  │
│   │   ┌─────────────┐     ┌─────────────┐   missing patch              │  │
│   │   │ 🌲  ▓▓  🏠  │     │ 🌲  🚗  🏠  │                              │  │
│   │   │     ▓▓      │  →  │     🚗      │   Must understand           │  │
│   │   │ 🌲  ▓▓  🏠  │     │ 🌲  🚗  🏠  │   context!                  │  │
│   │   └─────────────┘     └─────────────┘                              │  │
│   │                                                                     │  │
│   │   ▓▓ = masked region                                               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   4. COLORIZATION:                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Grayscale ──► Model ──► Predict Colors                           │  │
│   │                                                                     │  │
│   │   Must learn: sky=blue, grass=green, skin tones, etc.              │  │
│   │   Learns semantic understanding!                                    │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Contrastive Learning

### Core Idea

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONTRASTIVE LEARNING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CORE IDEA:                                                               │
│   - Similar things should be CLOSE in embedding space                      │
│   - Different things should be FAR APART                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Embedding Space:                                                  │  │
│   │                                                                     │  │
│   │          ▲                                                          │  │
│   │          │      🐕 🐕                                               │  │
│   │          │       🐕        ← Dogs cluster together                  │  │
│   │          │                                                          │  │
│   │          │                                                          │  │
│   │   🐱 🐱   │                                                          │  │
│   │    🐱    │                 ← Cats cluster together                  │  │
│   │          │                                                          │  │
│   │          │          🚗🚗                                            │  │
│   │          │           🚗   ← Cars cluster together                   │  │
│   │          └─────────────────────────────────►                        │  │
│   │                                                                     │  │
│   │   Different classes are FAR from each other                        │  │
│   │   Same class instances are CLOSE to each other                     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   BUT WAIT - we don't have labels! How do we know what's "same class"?    │
│                                                                             │
│   SOLUTION: Use DATA AUGMENTATION                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Original Image                                                    │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   ┌─────────┐                                                       │  │
│   │   │         │                                                       │  │
│   │   │   🐱    │                                                       │  │
│   │   │         │                                                       │  │
│   │   └─────────┘                                                       │  │
│   │        │                                                            │  │
│   │   ┌────┴────┐                                                       │  │
│   │   │         │                                                       │  │
│   │   ▼         ▼                                                       │  │
│   │  Aug 1     Aug 2                                                    │  │
│   │ (crop)   (color)                                                    │  │
│   │ ┌─────┐  ┌─────┐                                                    │  │
│   │ │  🐱 │  │  🐱 │  ← POSITIVE PAIR (same image, different views)    │  │
│   │ └─────┘  └─────┘                                                    │  │
│   │                                                                     │  │
│   │   Different images = NEGATIVE PAIRS                                 │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SimCLR Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SimCLR FRAMEWORK                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ARCHITECTURE:                                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │       Image x                                                       │  │
│   │          │                                                          │  │
│   │    ┌─────┴─────┐                                                    │  │
│   │    │           │                                                    │  │
│   │    ▼           ▼                                                    │  │
│   │  t(x)         t'(x)      ← Random augmentations                    │  │
│   │    │           │                                                    │  │
│   │    ▼           ▼                                                    │  │
│   │ ┌──────┐   ┌──────┐                                                │  │
│   │ │  f   │   │  f   │      ← Encoder (ResNet) - SHARED weights       │  │
│   │ │      │   │      │                                                │  │
│   │ └──┬───┘   └──┬───┘                                                │  │
│   │    │  h_i     │  h_j     ← Representations                         │  │
│   │    ▼          ▼                                                    │  │
│   │ ┌──────┐   ┌──────┐                                                │  │
│   │ │  g   │   │  g   │      ← Projection head (MLP) - SHARED weights  │  │
│   │ └──┬───┘   └──┬───┘                                                │  │
│   │    │  z_i     │  z_j     ← Projected embeddings                    │  │
│   │    │          │                                                    │  │
│   │    └────┬─────┘                                                    │  │
│   │         │                                                          │  │
│   │         ▼                                                          │  │
│   │   Contrastive Loss                                                 │  │
│   │   (attract z_i, z_j)                                               │  │
│   │   (repel from others)                                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   AUGMENTATIONS (critical for good performance!):                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   • Random crop and resize (most important!)                       │  │
│   │   • Color jitter (brightness, contrast, saturation, hue)           │  │
│   │   • Random grayscale                                               │  │
│   │   • Gaussian blur                                                  │  │
│   │   • Random horizontal flip                                         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   NT-Xent LOSS (Normalized Temperature-scaled Cross Entropy):              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │               exp(sim(z_i, z_j) / τ)                               │  │
│   │   L_ij = -log ─────────────────────────────                        │  │
│   │               Σ_k exp(sim(z_i, z_k) / τ)                           │  │
│   │                                                                     │  │
│   │   sim(u, v) = u · v / (||u|| ||v||)  (cosine similarity)          │  │
│   │   τ = temperature (typically 0.1 - 0.5)                            │  │
│   │                                                                     │  │
│   │   Numerator: Positive pair similarity                              │  │
│   │   Denominator: Sum over all pairs (positive + negatives)           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recurrent Neural Networks

### Why RNNs?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY RECURRENT NEURAL NETWORKS?                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FEEDFORWARD NETWORKS:                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input ──► Hidden ──► Output                                      │  │
│   │                                                                     │  │
│   │   • Fixed input size                                               │  │
│   │   • No memory of previous inputs                                   │  │
│   │   • Can't handle sequences                                         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   BUT MANY PROBLEMS ARE SEQUENTIAL:                                        │
│                                                                             │
│   "The movie was great but the ending was terrible"                        │
│                                                                             │
│   To understand "terrible", you need context from earlier words!           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Text:     "The"  "movie"  "was"  "great"  "but"  "ending"  ...    │  │
│   │  Meaning:   ───────────────────────────────────────────►           │  │
│   │             Context builds up over time                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   SEQUENTIAL DATA EXAMPLES:                                                │
│   • Text / Language                                                        │
│   • Time series (stock prices, weather)                                    │
│   • Audio / Speech                                                         │
│   • Video (sequence of frames)                                             │
│   • DNA sequences                                                          │
│   • User behavior over time                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RNN Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RNN ARCHITECTURE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CORE IDEA: Hidden state carries information through time                 │
│                                                                             │
│   COMPACT VIEW:                   UNROLLED VIEW:                           │
│                                                                             │
│        ┌────┐                     h₀    h₁    h₂    h₃                     │
│        │    │◄─┐                  │     │     │     │                      │
│   x ──►│ h  │──┤                  ▼     ▼     ▼     ▼                      │
│        │    │──┼──► y           ┌───┐ ┌───┐ ┌───┐ ┌───┐                   │
│        └────┘  │           x₀ ─►│ h │─►│ h │─►│ h │─►│ h │─► ...           │
│           ▲    │                └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘                   │
│           └────┘                  │     │     │     │                      │
│        (loop back)                ▼     ▼     ▼     ▼                      │
│                                   y₀    y₁    y₂    y₃                     │
│                                                                             │
│   SAME weights at every time step!                                         │
│                                                                             │
│                                                                             │
│   THE EQUATIONS:                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)                    │  │
│   │         └──────┬──────┘   └─────┬─────┘                            │  │
│   │         Previous state    Current input                             │  │
│   │                                                                     │  │
│   │   y_t = W_hy · h_t + b_y                                           │  │
│   │                                                                     │  │
│   │   Parameters:                                                      │  │
│   │   • W_hh: hidden-to-hidden weights (H × H)                        │  │
│   │   • W_xh: input-to-hidden weights (H × D)                         │  │
│   │   • W_hy: hidden-to-output weights (C × H)                        │  │
│   │   • b_h, b_y: biases                                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   COMPUTATIONAL GRAPH:                                                     │
│                                                                             │
│   x₀     x₁     x₂     x₃                                                 │
│    │      │      │      │                                                  │
│    ▼      ▼      ▼      ▼                                                  │
│   ┌──┐   ┌──┐   ┌──┐   ┌──┐                                               │
│   │W │   │W │   │W │   │W │    W_xh (shared)                              │
│   └┬─┘   └┬─┘   └┬─┘   └┬─┘                                               │
│    │      │      │      │                                                  │
│    ▼      ▼      ▼      ▼                                                  │
│   ┌──┐   ┌──┐   ┌──┐   ┌──┐                                               │
│ ─►│+ │──►│+ │──►│+ │──►│+ │──►   (+ with W_hh × h_{t-1})                  │
│   └┬─┘   └┬─┘   └┬─┘   └┬─┘                                               │
│    │      │      │      │                                                  │
│    ▼      ▼      ▼      ▼                                                  │
│  tanh   tanh   tanh   tanh                                                │
│    │      │      │      │                                                  │
│    ▼      ▼      ▼      ▼                                                  │
│   h₁     h₂     h₃     h₄                                                 │
│    │      │      │      │                                                  │
│    ▼      ▼      ▼      ▼                                                  │
│   y₁     y₂     y₃     y₄                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RNN Architectures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RNN ARCHITECTURES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. ONE-TO-ONE (Standard Neural Net):                                     │
│   ┌─────┐                                                                  │
│   │  ○  │    Fixed input → Fixed output                                   │
│   │  │  │    Example: Image classification                                 │
│   │  ○  │                                                                  │
│   └─────┘                                                                  │
│                                                                             │
│   2. ONE-TO-MANY (Image Captioning):                                       │
│   ┌─────────────────────────────────────┐                                  │
│   │  ○───○───○───○───○                  │  Single input → Sequence output │
│   │  ↑   ↓   ↓   ↓   ↓                  │  Example: Generate description  │
│   │ IMG  A   cat  is  sitting           │  from image                     │
│   └─────────────────────────────────────┘                                  │
│                                                                             │
│   3. MANY-TO-ONE (Sentiment Analysis):                                     │
│   ┌─────────────────────────────────────┐                                  │
│   │  ○───○───○───○───○                  │  Sequence input → Single output │
│   │  ↑   ↑   ↑   ↑   ↓                  │  Example: Classify review as    │
│   │  I  love this movie! POSITIVE       │  positive/negative              │
│   └─────────────────────────────────────┘                                  │
│                                                                             │
│   4. MANY-TO-MANY (Machine Translation):                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ○───○───○───○        ○───○───○───○                               │  │
│   │   ↑   ↑   ↑   ↑        ↓   ↓   ↓   ↓                               │  │
│   │   I  love cats  → ENC → DEC → J'aime les chats                     │  │
│   │                                                                     │  │
│   │   Encoder reads input, Decoder generates output                    │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   5. MANY-TO-MANY (Video Classification):                                  │
│   ┌─────────────────────────────────────┐                                  │
│   │  ○───○───○───○───○                  │  Aligned sequences              │
│   │  ↑   ↑   ↑   ↑   ↑                  │  Same length I/O               │
│   │  ↓   ↓   ↓   ↓   ↓                  │  Example: Per-frame labels     │
│   │  f1  f2  f3  f4  f5                 │                                  │
│   └─────────────────────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Vanishing Gradients

### The Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VANISHING GRADIENT PROBLEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BACKPROPAGATION THROUGH TIME:                                            │
│                                                                             │
│   Forward:                                                                  │
│   h₀ ─W─► h₁ ─W─► h₂ ─W─► h₃ ─W─► h₄ ─W─► h₅ ─W─► ... ─W─► h₁₀₀          │
│                                                                             │
│   Backward (chain rule):                                                   │
│   ∂L/∂h₀ = ∂L/∂h₁₀₀ × ∂h₁₀₀/∂h₉₉ × ∂h₉₉/∂h₉₈ × ... × ∂h₁/∂h₀            │
│                                                                             │
│   Each ∂h_t/∂h_{t-1} involves multiplying by W and tanh derivative!       │
│                                                                             │
│                                                                             │
│   THE PROBLEM:                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   tanh derivative: max = 1 at x=0, <1 everywhere else              │  │
│   │                                                                     │  │
│   │        ▲                                                           │  │
│   │      1 │    ╭─────╮                                                │  │
│   │        │   ╱       ╲                                               │  │
│   │        │  ╱         ╲                                              │  │
│   │      0 ├─╯───────────╰─────► x                                     │  │
│   │        │                                                           │  │
│   │                                                                     │  │
│   │   Multiplying many values < 1:                                     │  │
│   │   0.9 × 0.9 × 0.9 × ... × 0.9 (100 times) = 0.9¹⁰⁰ ≈ 0.000027     │  │
│   │                                                                     │  │
│   │   Gradient basically DISAPPEARS!                                   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   GRADIENT MAGNITUDE OVER TIME:                                            │
│                                                                             │
│   Gradient                                                                 │
│      │                                                                     │
│    1 │██                                                                   │
│      │███                                                                  │
│      │████                                                                 │
│      │█████                                                                │
│      │██████▁                                                              │
│    0 └──────▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁► Time steps                              │
│      h₁₀₀ h₉₀ h₈₀ h₇₀ h₆₀ h₅₀ h₄₀ h₃₀ h₂₀ h₁₀ h₀                        │
│                                                                             │
│   Early time steps get almost NO gradient → No learning!                  │
│                                                                             │
│                                                                             │
│   EXPLODING GRADIENTS (opposite problem):                                  │
│   If eigenvalues of W > 1, gradients EXPLODE                              │
│   1.1¹⁰⁰ ≈ 13,780 → NaN in training                                       │
│                                                                             │
│   Solution: Gradient clipping                                              │
│   if ||gradient|| > threshold:                                            │
│       gradient = gradient * threshold / ||gradient||                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LSTM Networks

### LSTM Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LSTM (Long Short-Term Memory)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   KEY IDEA: Add a separate CELL STATE that flows through time              │
│             with minimal modification (like a "highway")                   │
│                                                                             │
│   VANILLA RNN:                  LSTM:                                      │
│   ┌─────────┐                   ┌─────────────────────────┐               │
│   │         │                   │  Cell state highway     │               │
│   │    h    │                   │  ─────────────────────► │               │
│   │         │                   │                         │               │
│   └─────────┘                   │    ┌───┐ ┌───┐ ┌───┐   │               │
│                                 │    │ f │ │ i │ │ o │   │               │
│   Only hidden state             │    └───┘ └───┘ └───┘   │               │
│   (gets squashed)               │    Gates control flow  │               │
│                                 └─────────────────────────┘               │
│                                                                             │
│                                                                             │
│   LSTM CELL DIAGRAM:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │         c_{t-1} ─────────────────►×────►+────────────► c_t          │  │
│   │                                   ▲     ▲                           │  │
│   │                                   │     │                           │  │
│   │                              ┌────┴─┐ ┌─┴────┐                      │  │
│   │                              │  f   │ │  i   │                      │  │
│   │                              │      │ │      │                      │  │
│   │                              │forget│ │input │                      │  │
│   │                              │ gate │ │ gate │                      │  │
│   │                              └──────┘ └──┬───┘                      │  │
│   │                                          │                          │  │
│   │                                       ┌──┴──┐                       │  │
│   │                                       │tanh │                       │  │
│   │                                       │  g  │                       │  │
│   │                                       └──┬──┘                       │  │
│   │                                          │                          │  │
│   │         h_{t-1} ─────────────────────────┼────────────►×───► h_t    │  │
│   │                                          │             ▲            │  │
│   │                                          │             │            │  │
│   │                                    ┌─────┴─────┐   ┌───┴───┐        │  │
│   │           x_t ────────────────────►│  Concat   │   │tanh   │        │  │
│   │                                    │ [h,x] → W │   │  ↑    │        │  │
│   │                                    └───────────┘   │  c_t  │        │  │
│   │                                          │         └───────┘        │  │
│   │                                          ▼             ▲            │  │
│   │                                    ┌──────────┐    ┌───┴───┐        │  │
│   │                                    │  Output  │    │output │        │  │
│   │                                    │   gate   │───►│ gate  │        │  │
│   │                                    │    o     │    │   o   │        │  │
│   │                                    └──────────┘    └───────┘        │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   THE GATES (all use sigmoid → values between 0 and 1):                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   FORGET GATE: What to forget from cell state                      │  │
│   │   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)                              │  │
│   │   0 = forget everything, 1 = keep everything                        │  │
│   │                                                                     │  │
│   │   INPUT GATE: What new information to add                          │  │
│   │   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)                              │  │
│   │   g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   (candidate values)      │  │
│   │                                                                     │  │
│   │   CELL UPDATE:                                                     │  │
│   │   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                                  │  │
│   │         └────┬────┘   └────┬────┘                                  │  │
│   │         keep old      add new                                       │  │
│   │                                                                     │  │
│   │   OUTPUT GATE: What to output to hidden state                      │  │
│   │   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)                              │  │
│   │   h_t = o_t ⊙ tanh(c_t)                                            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WHY THIS HELPS:                                                          │
│   • Cell state gradient flows through addition (not multiplication)        │
│   • Gates can learn to keep gradients = 1 when needed                     │
│   • Long-range dependencies can be learned                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## GRU Networks

### GRU Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRU (Gated Recurrent Unit)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SIMPLIFIED VERSION OF LSTM:                                              │
│   • Fewer parameters (faster training)                                     │
│   • Similar performance in many tasks                                      │
│   • Combines forget and input gates into one "update" gate                 │
│   • No separate cell state                                                 │
│                                                                             │
│   COMPARISON:                                                              │
│   ┌───────────────────┬───────────────────────────────────────────────────┐│
│   │      LSTM         │              GRU                                  ││
│   ├───────────────────┼───────────────────────────────────────────────────┤│
│   │  3 gates          │  2 gates                                          ││
│   │  (forget, input,  │  (reset, update)                                  ││
│   │   output)         │                                                   ││
│   ├───────────────────┼───────────────────────────────────────────────────┤│
│   │  Separate cell    │  Only hidden state                                ││
│   │  state c_t        │                                                   ││
│   ├───────────────────┼───────────────────────────────────────────────────┤│
│   │  8 weight         │  6 weight matrices                                ││
│   │  matrices         │                                                   ││
│   └───────────────────┴───────────────────────────────────────────────────┘│
│                                                                             │
│   GRU EQUATIONS:                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   RESET GATE: How much of past to forget                           │  │
│   │   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)                              │  │
│   │                                                                     │  │
│   │   UPDATE GATE: How much to update hidden state                     │  │
│   │   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)                              │  │
│   │                                                                     │  │
│   │   CANDIDATE HIDDEN STATE:                                          │  │
│   │   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)                     │  │
│   │              └─────────┬─────────┘                                 │  │
│   │              Reset gate controls how much                           │  │
│   │              of previous hidden to use                              │  │
│   │                                                                     │  │
│   │   FINAL HIDDEN STATE:                                              │  │
│   │   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t                           │  │
│   │         └─────────┬─────────┘   └────┬───┘                         │  │
│   │         Keep from old            Add new                            │  │
│   │                                                                     │  │
│   │   Note: z_t = 0 → keep old, z_t = 1 → use new                      │  │
│   │   (Opposite intuition from LSTM forget gate!)                       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   GRU CELL DIAGRAM:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   h_{t-1} ─────────────────────────────────────► × ───┐             │  │
│   │       │                                          ▲    │             │  │
│   │       │                                       (1-z_t) │             │  │
│   │       │                                               ▼             │  │
│   │       │                                           ┌───┐             │  │
│   │       ├───────────────► × ──► tanh ──► × ─────────►│ + │───► h_t    │  │
│   │       │                 ▲              ▲           └───┘             │  │
│   │       │                 │            z_t                             │  │
│   │       │               r_t                                            │  │
│   │       │                 ▲              ▲                             │  │
│   │       ▼                 │              │                             │  │
│   │   ┌───────┐         ┌───┴───┐      ┌───┴───┐                        │  │
│   │   │Concat │────────►│ Reset │      │Update │                        │  │
│   │   │ [h,x] │         │  Gate │      │ Gate  │                        │  │
│   │   └───────┘         └───────┘      └───────┘                        │  │
│   │       ▲                                                              │  │
│   │       │                                                              │  │
│   │     x_t                                                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WHEN TO USE WHICH:                                                       │
│   • LSTM: Longer sequences, more complex dependencies                     │
│   • GRU: Smaller datasets, faster training needed, similar performance    │
│   • Both: Try both and compare on validation set!                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sequence-to-Sequence

### Encoder-Decoder Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE-TO-SEQUENCE MODELS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ENCODER-DECODER ARCHITECTURE:                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │           ENCODER                      DECODER                      │  │
│   │                                                                     │  │
│   │   ┌───┐ ┌───┐ ┌───┐ ┌───┐         ┌───┐ ┌───┐ ┌───┐ ┌───┐         │  │
│   │   │ h │─│ h │─│ h │─│ h │───────►│ h │─│ h │─│ h │─│ h │          │  │
│   │   └─▲─┘ └─▲─┘ └─▲─┘ └─▲─┘  context└─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘          │  │
│   │     │     │     │     │    vector   │     │     │     │            │  │
│   │     │     │     │     │     (c)     ▼     ▼     ▼     ▼            │  │
│   │    "I"  "love" "cats" <EOS>       <SOS> "J'" "aime" "les" "chats"  │  │
│   │                                                                     │  │
│   │   English input                    French output                    │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TRAINING:                                                                │
│   • Input: source sequence                                                 │
│   • Target: shifted target sequence (teacher forcing)                     │
│   • Loss: cross-entropy on each output token                              │
│                                                                             │
│   INFERENCE:                                                               │
│   • Start with <SOS> token                                                │
│   • Generate one token at a time                                          │
│   • Feed generated token as next input                                    │
│   • Stop when <EOS> generated                                             │
│                                                                             │
│                                                                             │
│   CONTEXT VECTOR BOTTLENECK:                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Problem: Entire input sequence compressed into one vector!       │  │
│   │                                                                     │  │
│   │   "The quick brown fox jumps over the lazy dog" ──► [0.1, 0.3, ...]│  │
│   │                                                     Single vector! │  │
│   │                                                                     │  │
│   │   Long sentences: Information lost!                                │  │
│   │   Solution: ATTENTION (coming in Week 7!)                          │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   APPLICATIONS:                                                            │
│   • Machine translation (English → French)                                │
│   • Text summarization (article → summary)                                │
│   • Chatbots (question → answer)                                          │
│   • Code generation (description → code)                                  │
│   • Image captioning (image features → description)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Image Captioning

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMAGE CAPTIONING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TASK: Generate natural language description of an image                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ┌─────────────┐        ┌─────────────────────────────────────┐   │  │
│   │   │             │        │                                     │   │  │
│   │   │    IMAGE    │   ───► │  "A cat sitting on a couch"         │   │  │
│   │   │     🐱      │        │                                     │   │  │
│   │   │             │        │                                     │   │  │
│   │   └─────────────┘        └─────────────────────────────────────┘   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ARCHITECTURE:                                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ┌─────────────┐     ┌─────────────┐                              │  │
│   │   │             │     │             │                              │  │
│   │   │    IMAGE    │────►│    CNN      │────► Image Features (v)      │  │
│   │   │             │     │  (ResNet)   │      (e.g., 2048-dim)        │  │
│   │   │             │     │             │                              │  │
│   │   └─────────────┘     └─────────────┘                              │  │
│   │                              │                                      │  │
│   │                              ▼                                      │  │
│   │                        Project to                                   │  │
│   │                        hidden size                                  │  │
│   │                              │                                      │  │
│   │                              ▼                                      │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │                         LSTM                                 │  │  │
│   │   │                                                              │  │  │
│   │   │    v ─► h₀ ─► h₁ ─► h₂ ─► h₃ ─► h₄ ─► h₅                   │  │  │
│   │   │              ▲     ▲     ▲     ▲     ▲                      │  │  │
│   │   │              │     │     │     │     │                      │  │  │
│   │   │           <START>  A    cat sitting  on                     │  │  │
│   │   │              │     │     │     │     │                      │  │  │
│   │   │              ▼     ▼     ▼     ▼     ▼                      │  │  │
│   │   │              A    cat sitting  on   couch                   │  │  │
│   │   │                                                              │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TRAINING:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input: Image + Ground truth caption                              │  │
│   │                                                                     │  │
│   │   1. Extract image features with pretrained CNN                    │  │
│   │   2. Feed image features as initial hidden state (or first input) │  │
│   │   3. Teacher forcing: feed ground truth words at each step        │  │
│   │   4. Cross-entropy loss on predicted words                         │  │
│   │                                                                     │  │
│   │   Loss = -Σ log P(word_t | image, word_1, ..., word_{t-1})        │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   INFERENCE (Sampling):                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. Extract image features                                        │  │
│   │   2. Start with <START> token                                      │  │
│   │   3. Generate next word (sample or argmax)                         │  │
│   │   4. Feed generated word as next input                             │  │
│   │   5. Repeat until <END> or max length                              │  │
│   │                                                                     │  │
│   │   Beam Search: Keep top-k candidates at each step                  │  │
│   │   (Better quality than greedy decoding)                            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Coding Exercises

### Exercise 1: Vanilla RNN

```python
#==============================================================================
# EXERCISE 1: VANILLA RNN IMPLEMENTATION
#==============================================================================

import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Forward pass for a single timestep of a vanilla RNN.

    Args:
        x: Input data of shape (N, D)
        prev_h: Previous hidden state of shape (N, H)
        Wx: Input-to-hidden weights of shape (D, H)
        Wh: Hidden-to-hidden weights of shape (H, H)
        b: Biases of shape (H,)

    Returns:
        next_h: Next hidden state of shape (N, H)
        cache: Values needed for backward pass
    """
    #===========================================================================
    # TODO: Implement one step of RNN forward pass
    # h_t = tanh(x @ Wx + h_{t-1} @ Wh + b)
    #===========================================================================

    next_h = np.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (x, prev_h, Wx, Wh, next_h)

    #===========================================================================

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state (N, H)
        cache: Cache from forward pass

    Returns:
        dx: Gradient of input (N, D)
        dprev_h: Gradient of previous hidden state (N, H)
        dWx: Gradient of input-to-hidden weights (D, H)
        dWh: Gradient of hidden-to-hidden weights (H, H)
        db: Gradient of biases (H,)
    """
    x, prev_h, Wx, Wh, next_h = cache

    #===========================================================================
    # TODO: Implement backward pass
    # Hint: derivative of tanh(x) is (1 - tanh(x)^2)
    #===========================================================================

    # Gradient through tanh
    dtanh = dnext_h * (1 - next_h ** 2)  # (N, H)

    # Gradients of parameters
    dx = dtanh @ Wx.T                    # (N, D)
    dprev_h = dtanh @ Wh.T               # (N, H)
    dWx = x.T @ dtanh                    # (D, H)
    dWh = prev_h.T @ dtanh               # (H, H)
    db = np.sum(dtanh, axis=0)           # (H,)

    #===========================================================================

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for a vanilla RNN over an entire sequence.

    Args:
        x: Input data of shape (N, T, D)
        h0: Initial hidden state of shape (N, H)
        Wx: Input-to-hidden weights of shape (D, H)
        Wh: Hidden-to-hidden weights of shape (H, H)
        b: Biases of shape (H,)

    Returns:
        h: Hidden states for all timesteps of shape (N, T, H)
        cache: Values needed for backward pass
    """
    N, T, D = x.shape
    H = h0.shape[1]

    h = np.zeros((N, T, H))
    cache = []

    prev_h = h0

    #===========================================================================
    # TODO: Implement forward pass through entire sequence
    #===========================================================================

    for t in range(T):
        next_h, step_cache = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(step_cache)
        prev_h = next_h

    #===========================================================================

    return h, cache


def rnn_backward(dh, cache):
    """
    Backward pass for a vanilla RNN over an entire sequence.

    Args:
        dh: Upstream gradients of hidden states (N, T, H)
        cache: Cache from forward pass

    Returns:
        dx: Gradient of inputs (N, T, D)
        dh0: Gradient of initial hidden state (N, H)
        dWx: Gradient of input-to-hidden weights (D, H)
        dWh: Gradient of hidden-to-hidden weights (H, H)
        db: Gradient of biases (H,)
    """
    N, T, H = dh.shape
    x, _, Wx, _, _ = cache[0]
    D = x.shape[1]

    dx = np.zeros((N, T, D))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros((H, H))
    db = np.zeros(H)

    dprev_h = np.zeros((N, H))

    #===========================================================================
    # TODO: Implement backward pass through entire sequence (reverse order!)
    #===========================================================================

    for t in reversed(range(T)):
        # Add upstream gradient from current timestep
        dnext_h = dh[:, t, :] + dprev_h

        # Backward through one step
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])

        # Accumulate gradients
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dh0 = dprev_h

    #===========================================================================

    return dx, dh0, dWx, dWh, db


#==============================================================================
# EXERCISE 2: LSTM IMPLEMENTATION
#==============================================================================

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    Args:
        x: Input data of shape (N, D)
        prev_h: Previous hidden state of shape (N, H)
        prev_c: Previous cell state of shape (N, H)
        Wx: Input-to-hidden weights of shape (D, 4H)
        Wh: Hidden-to-hidden weights of shape (H, 4H)
        b: Biases of shape (4H,)

    Returns:
        next_h: Next hidden state of shape (N, H)
        next_c: Next cell state of shape (N, H)
        cache: Values needed for backward pass
    """
    N, H = prev_h.shape

    #===========================================================================
    # TODO: Implement LSTM forward pass
    # The weight matrices pack all 4 gates: [i, f, o, g]
    #===========================================================================

    # Compute all gate values at once
    a = x @ Wx + prev_h @ Wh + b  # (N, 4H)

    # Split into individual gates
    ai = a[:, 0*H:1*H]    # Input gate
    af = a[:, 1*H:2*H]    # Forget gate
    ao = a[:, 2*H:3*H]    # Output gate
    ag = a[:, 3*H:4*H]    # Cell gate (g/candidate)

    # Apply activations
    i = 1 / (1 + np.exp(-ai))     # Sigmoid for input gate
    f = 1 / (1 + np.exp(-af))     # Sigmoid for forget gate
    o = 1 / (1 + np.exp(-ao))     # Sigmoid for output gate
    g = np.tanh(ag)               # Tanh for cell gate

    # Update cell state
    next_c = f * prev_c + i * g

    # Compute hidden state
    next_h = o * np.tanh(next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)

    #===========================================================================

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Args:
        dnext_h: Gradient of next hidden state (N, H)
        dnext_c: Gradient of next cell state (N, H)
        cache: Cache from forward pass

    Returns:
        dx: Gradient of input (N, D)
        dprev_h: Gradient of previous hidden state (N, H)
        dprev_c: Gradient of previous cell state (N, H)
        dWx: Gradient of input-to-hidden weights (D, 4H)
        dWh: Gradient of hidden-to-hidden weights (H, 4H)
        db: Gradient of biases (4H,)
    """
    x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache
    N, H = dnext_h.shape

    #===========================================================================
    # TODO: Implement LSTM backward pass
    #===========================================================================

    # Gradient of output gate contribution
    tanh_c = np.tanh(next_c)
    do = dnext_h * tanh_c
    dc = dnext_h * o * (1 - tanh_c ** 2)

    # Add upstream gradient from cell
    dc += dnext_c

    # Gradient through cell update
    df = dc * prev_c
    dprev_c = dc * f
    di = dc * g
    dg = dc * i

    # Gradient through activations
    dai = di * i * (1 - i)    # Sigmoid derivative
    daf = df * f * (1 - f)
    dao = do * o * (1 - o)
    dag = dg * (1 - g ** 2)   # Tanh derivative

    # Stack gate gradients
    da = np.hstack([dai, daf, dao, dag])  # (N, 4H)

    # Gradient of parameters
    dx = da @ Wx.T
    dprev_h = da @ Wh.T
    dWx = x.T @ da
    dWh = prev_h.T @ da
    db = np.sum(da, axis=0)

    #===========================================================================

    return dx, dprev_h, dprev_c, dWx, dWh, db


#==============================================================================
# EXERCISE 3: WORD EMBEDDING
#==============================================================================

def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings.

    Args:
        x: Integer array of word indices of shape (N, T)
        W: Embedding matrix of shape (V, D)
           V = vocabulary size, D = embedding dimension

    Returns:
        out: Word vectors of shape (N, T, D)
        cache: Values needed for backward pass
    """
    #===========================================================================
    # TODO: Implement embedding lookup
    #===========================================================================

    out = W[x]  # Fancy indexing!
    cache = (x, W.shape[0])

    #===========================================================================

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings.

    Args:
        dout: Upstream gradients of shape (N, T, D)
        cache: Cache from forward pass

    Returns:
        dW: Gradient of embedding matrix (V, D)
    """
    x, V = cache
    D = dout.shape[2]

    #===========================================================================
    # TODO: Implement embedding backward pass
    # Hint: Use np.add.at for in-place accumulation
    #===========================================================================

    dW = np.zeros((V, D))
    np.add.at(dW, x, dout)

    #===========================================================================

    return dW


#==============================================================================
# EXERCISE 4: CAPTIONING RNN
#==============================================================================

class CaptioningRNN:
    """
    RNN for image captioning.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn'):
        """
        Initialize the captioning model.

        Args:
            word_to_idx: Dictionary mapping words to indices
            input_dim: Dimension of image feature vectors
            wordvec_dim: Dimension of word embeddings
            hidden_dim: Dimension of RNN hidden state
            cell_type: 'rnn' or 'lstm'
        """
        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx.get('<NULL>', 0)
        self._start = word_to_idx.get('<START>', 1)
        self._end = word_to_idx.get('<END>', 2)

        # Initialize weights
        self.params = {}

        # Image feature projection
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Word embedding
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim) / 100

        # RNN/LSTM weights
        dim_mul = 4 if cell_type == 'lstm' else 1
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim) / np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim) / np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Output projection
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size) / np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

    def loss(self, features, captions):
        """
        Compute training loss for the captioning model.

        Args:
            features: Image features of shape (N, D)
            captions: Ground truth captions of shape (N, T)

        Returns:
            loss: Scalar loss value
            grads: Dictionary of gradients
        """
        # Split captions into input and target
        captions_in = captions[:, :-1]   # Everything except last
        captions_out = captions[:, 1:]   # Everything except first

        # Get params
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        N, T = captions_in.shape

        #=======================================================================
        # FORWARD PASS
        #=======================================================================

        # Project image features to initial hidden state
        h0 = features @ W_proj + b_proj  # (N, H)

        # Get word embeddings
        x, embed_cache = word_embedding_forward(captions_in, W_embed)  # (N, T, D)

        # Run through RNN
        if self.cell_type == 'rnn':
            h, rnn_cache = rnn_forward(x, h0, Wx, Wh, b)  # (N, T, H)
        else:
            c0 = np.zeros_like(h0)
            h, rnn_cache = lstm_forward(x, h0, c0, Wx, Wh, b)

        # Compute scores
        scores = h.reshape(N * T, -1) @ W_vocab + b_vocab  # (N*T, V)
        scores = scores.reshape(N, T, -1)

        # Compute loss (cross-entropy)
        # Create mask for non-null targets
        mask = (captions_out != self._null)

        # Softmax and cross-entropy
        scores_flat = scores.reshape(-1, scores.shape[-1])
        targets_flat = captions_out.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Numerically stable softmax
        probs = np.exp(scores_flat - np.max(scores_flat, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)

        # Cross-entropy loss (only for non-null targets)
        loss = -np.sum(
            np.log(probs[np.arange(N*T), targets_flat] + 1e-10) * mask_flat
        ) / np.sum(mask_flat)

        #=======================================================================
        # BACKWARD PASS
        #=======================================================================

        grads = {}

        # Gradient of softmax
        dscores = probs.copy()
        dscores[np.arange(N*T), targets_flat] -= 1
        dscores *= mask_flat[:, None]
        dscores /= np.sum(mask_flat)
        dscores = dscores.reshape(N, T, -1)

        # Gradient of vocab projection
        dscores_flat = dscores.reshape(N*T, -1)
        h_flat = h.reshape(N*T, -1)
        grads['W_vocab'] = h_flat.T @ dscores_flat
        grads['b_vocab'] = np.sum(dscores_flat, axis=0)

        # Gradient of hidden states
        dh = (dscores_flat @ W_vocab.T).reshape(N, T, -1)

        # Gradient through RNN
        if self.cell_type == 'rnn':
            dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh, rnn_cache)
        else:
            dx, dh0, _, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dh, rnn_cache)

        # Gradient of word embeddings
        grads['W_embed'] = word_embedding_backward(dx, embed_cache)

        # Gradient of image projection
        grads['W_proj'] = features.T @ dh0
        grads['b_proj'] = np.sum(dh0, axis=0)

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Generate captions for images using greedy decoding.

        Args:
            features: Image features of shape (N, D)
            max_length: Maximum caption length

        Returns:
            captions: Array of shape (N, max_length) with sampled word indices
        """
        N = features.shape[0]
        captions = np.zeros((N, max_length), dtype=np.int32)

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        # Initialize hidden state from image features
        h = features @ W_proj + b_proj
        c = np.zeros_like(h) if self.cell_type == 'lstm' else None

        # Start with <START> token
        current_word = np.full(N, self._start)

        for t in range(max_length):
            # Get word embedding
            x = W_embed[current_word]  # (N, D)

            # RNN step
            if self.cell_type == 'rnn':
                h = np.tanh(x @ Wx + h @ Wh + b)
            else:
                h, c, _ = lstm_step_forward(x, h, c, Wx, Wh, b)

            # Compute scores and get next word
            scores = h @ W_vocab + b_vocab
            current_word = np.argmax(scores, axis=1)
            captions[:, t] = current_word

        return captions
```

---

## Business Applications

### Production Text Processing System

```python
#==============================================================================
# BUSINESS APPLICATION: Customer Review Analyzer
#==============================================================================

import numpy as np
from typing import List, Dict, Tuple
import re

class ReviewAnalyzer:
    """
    Production system for analyzing customer reviews.

    Use Cases:
    - E-commerce: Understand product sentiment
    - Hospitality: Monitor customer satisfaction
    - Support: Prioritize urgent issues
    - Marketing: Track brand perception
    """

    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        """Initialize the review analyzer."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Vocabulary
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

        # Initialize simple RNN model
        self.params = {
            'W_embed': np.random.randn(vocab_size, embedding_dim) * 0.01,
            'Wx': np.random.randn(embedding_dim, hidden_dim) * 0.01,
            'Wh': np.random.randn(hidden_dim, hidden_dim) * 0.01,
            'b': np.zeros(hidden_dim),
            'W_out': np.random.randn(hidden_dim, 3) * 0.01,  # 3 classes: neg, neu, pos
            'b_out': np.zeros(3)
        }

        # Sentiment labels
        self.sentiment_labels = ['negative', 'neutral', 'positive']

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for analysis.

        Args:
            text: Raw text string

        Returns:
            List of cleaned tokens
        """
        # Lowercase
        text = text.lower()

        # Remove special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Tokenize
        tokens = text.split()

        # Remove short tokens
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def build_vocab(self, reviews: List[str], min_freq: int = 2):
        """
        Build vocabulary from training reviews.

        Args:
            reviews: List of review texts
            min_freq: Minimum frequency for a word to be included
        """
        word_counts = {}

        for review in reviews:
            tokens = self.preprocess_text(review)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

        # Add to vocabulary
        idx = 2  # Start after <PAD> and <UNK>
        for word, count in sorted_words:
            if count >= min_freq and idx < self.vocab_size:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1

        print(f"Vocabulary size: {len(self.word_to_idx)}")

    def encode_text(self, text: str, max_length: int = 100) -> np.ndarray:
        """
        Encode text as sequence of indices.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Array of word indices
        """
        tokens = self.preprocess_text(text)

        # Convert to indices
        indices = []
        for token in tokens[:max_length]:
            idx = self.word_to_idx.get(token, self.word_to_idx['<UNK>'])
            indices.append(idx)

        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.word_to_idx['<PAD>'])

        return np.array(indices)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through RNN.

        Args:
            x: Input indices of shape (N, T)

        Returns:
            Tuple of (scores, hidden_states)
        """
        N, T = x.shape

        # Get embeddings
        embeddings = self.params['W_embed'][x]  # (N, T, D)

        # Initialize hidden state
        h = np.zeros((N, self.hidden_dim))

        # Process sequence
        for t in range(T):
            x_t = embeddings[:, t, :]
            h = np.tanh(
                x_t @ self.params['Wx'] +
                h @ self.params['Wh'] +
                self.params['b']
            )

        # Output layer
        scores = h @ self.params['W_out'] + self.params['b_out']

        return scores, h

    def analyze_sentiment(self, review: str) -> Dict:
        """
        Analyze sentiment of a single review.

        Args:
            review: Review text

        Returns:
            Dictionary with sentiment analysis results
        """
        # Encode
        x = self.encode_text(review).reshape(1, -1)

        # Forward pass
        scores, _ = self.forward(x)

        # Softmax
        probs = np.exp(scores - np.max(scores))
        probs = probs / np.sum(probs)
        probs = probs.flatten()

        # Get prediction
        pred_idx = np.argmax(probs)
        sentiment = self.sentiment_labels[pred_idx]

        return {
            'text': review[:100] + '...' if len(review) > 100 else review,
            'sentiment': sentiment,
            'confidence': float(probs[pred_idx]),
            'probabilities': {
                'negative': float(probs[0]),
                'neutral': float(probs[1]),
                'positive': float(probs[2])
            }
        }

    def analyze_batch(self, reviews: List[str]) -> List[Dict]:
        """
        Analyze multiple reviews efficiently.

        Args:
            reviews: List of review texts

        Returns:
            List of analysis results
        """
        # Encode all reviews
        max_len = 100
        x = np.array([self.encode_text(r, max_len) for r in reviews])

        # Forward pass
        scores, _ = self.forward(x)

        # Softmax
        probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # Create results
        results = []
        for i, review in enumerate(reviews):
            pred_idx = np.argmax(probs[i])
            results.append({
                'text': review[:100] + '...' if len(review) > 100 else review,
                'sentiment': self.sentiment_labels[pred_idx],
                'confidence': float(probs[i, pred_idx])
            })

        return results

    def extract_key_phrases(self, review: str, top_k: int = 5) -> List[str]:
        """
        Extract key phrases from a review.

        Args:
            review: Review text
            top_k: Number of phrases to extract

        Returns:
            List of key phrases
        """
        tokens = self.preprocess_text(review)

        # Simple approach: return most unique words
        # In production, use TF-IDF or attention weights
        word_counts = {}
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

        return [word for word, _ in sorted_words[:top_k]]

    def generate_summary(self, reviews: List[str]) -> Dict:
        """
        Generate summary statistics for a batch of reviews.

        Args:
            reviews: List of review texts

        Returns:
            Summary dictionary
        """
        results = self.analyze_batch(reviews)

        # Count sentiments
        sentiment_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
        for r in results:
            sentiment_counts[r['sentiment']] += 1

        total = len(reviews)

        # Calculate averages
        avg_confidence = np.mean([r['confidence'] for r in results])

        # Find most positive and negative
        sorted_by_confidence = sorted(results, key=lambda x: -x['confidence'])
        positive_reviews = [r for r in sorted_by_confidence if r['sentiment'] == 'positive']
        negative_reviews = [r for r in sorted_by_confidence if r['sentiment'] == 'negative']

        return {
            'total_reviews': total,
            'sentiment_distribution': {
                k: v/total for k, v in sentiment_counts.items()
            },
            'average_confidence': float(avg_confidence),
            'net_sentiment_score': (sentiment_counts['positive'] - sentiment_counts['negative']) / total,
            'top_positive': positive_reviews[:3] if positive_reviews else [],
            'top_negative': negative_reviews[:3] if negative_reviews else []
        }


# Demo
def demo_review_analyzer():
    """Demonstrate the review analyzer."""

    analyzer = ReviewAnalyzer()

    # Sample reviews
    reviews = [
        "This product is amazing! Best purchase I've ever made.",
        "Terrible quality, broke after one day. Want my money back.",
        "It's okay, nothing special but does the job.",
        "Love it! Exceeded my expectations in every way.",
        "Waste of money. Don't buy this garbage.",
        "Pretty good value for the price. Would recommend.",
        "The product arrived damaged. Very disappointed.",
        "Works as described. Happy with my purchase."
    ]

    # Build vocabulary
    analyzer.build_vocab(reviews, min_freq=1)

    print("=" * 60)
    print("CUSTOMER REVIEW ANALYZER DEMO")
    print("=" * 60)

    # Analyze individual reviews
    print("\nIndividual Review Analysis:")
    print("-" * 40)

    for review in reviews[:3]:
        result = analyzer.analyze_sentiment(review)
        print(f"\nReview: {result['text']}")
        print(f"Sentiment: {result['sentiment']} ({result['confidence']:.1%})")

    # Generate summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)

    summary = analyzer.generate_summary(reviews)
    print(f"\nTotal Reviews: {summary['total_reviews']}")
    print(f"Sentiment Distribution:")
    for sentiment, ratio in summary['sentiment_distribution'].items():
        bar = "█" * int(ratio * 20)
        print(f"  {sentiment:10s}: {bar} {ratio:.1%}")

    print(f"\nNet Sentiment Score: {summary['net_sentiment_score']:.2f}")
    print("  (-1 = all negative, +1 = all positive)")


if __name__ == '__main__':
    demo_review_analyzer()
```

---

## Summary: Week 5-6 Checklist

### Concepts You Should Understand
- [ ] Self-supervised learning motivation
- [ ] Pretext tasks (rotation, jigsaw, colorization)
- [ ] Contrastive learning (positive/negative pairs)
- [ ] SimCLR framework and NT-Xent loss
- [ ] RNN architecture and unrolling
- [ ] Vanishing/exploding gradient problem
- [ ] LSTM gates (forget, input, output)
- [ ] GRU gates (reset, update)
- [ ] Sequence-to-sequence models
- [ ] Teacher forcing
- [ ] Image captioning pipeline

### Skills You Should Have
- [ ] Implement vanilla RNN forward and backward
- [ ] Implement LSTM forward and backward
- [ ] Implement word embeddings
- [ ] Build a captioning RNN
- [ ] Implement contrastive loss
- [ ] Apply data augmentation for SSL

### Key Formulas
```
RNN: h_t = tanh(Wx*x_t + Wh*h_{t-1} + b)

LSTM Gates:
  f_t = σ(Wf*[h_{t-1}, x_t] + bf)  # Forget
  i_t = σ(Wi*[h_{t-1}, x_t] + bi)  # Input
  o_t = σ(Wo*[h_{t-1}, x_t] + bo)  # Output
  c_t = f_t⊙c_{t-1} + i_t⊙tanh(Wc*[h_{t-1}, x_t] + bc)
  h_t = o_t⊙tanh(c_t)

Contrastive Loss:
  L = -log(exp(sim(z_i,z_j)/τ) / Σ_k exp(sim(z_i,z_k)/τ))
```

### Next Steps
After completing Week 5-6, you're ready for:
- **Week 7**: Attention and Transformers
- Understanding how attention solves sequence bottlenecks
- Building transformer models from scratch
