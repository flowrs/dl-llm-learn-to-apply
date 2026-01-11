# Week 7: Attention and Transformers
## From Novice to Practitioner: The Architecture That Changed AI

---

## Table of Contents
1. [The Attention Revolution](#the-attention-revolution)
2. [Attention Mechanism](#attention-mechanism)
3. [Self-Attention](#self-attention)
4. [Multi-Head Attention](#multi-head-attention)
5. [The Transformer](#the-transformer)
6. [Positional Encoding](#positional-encoding)
7. [Vision Transformer (ViT)](#vision-transformer-vit)
8. [Coding Exercises](#coding-exercises)
9. [Business Applications](#business-applications)

---

## The Attention Revolution

### The Problem with Seq2Seq

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  THE BOTTLENECK PROBLEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ENCODER-DECODER WITHOUT ATTENTION:                                       │
│                                                                             │
│   "The cat sat on the mat" ──► ENCODER ──► [context vector] ──► DECODER   │
│                                                   │                        │
│                                                   ▼                        │
│                                           Single fixed-size                │
│                                           vector (e.g., 512-dim)           │
│                                                                             │
│   PROBLEM: ALL information must fit in one vector!                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Short sentence: "Hi" ──────────────► [512-dim] ✓ Easy            │  │
│   │                                                                     │  │
│   │   Long document:                                                    │  │
│   │   "The comprehensive report on climate change and its impacts     │  │
│   │    on global ecosystems, including detailed analysis of           │  │
│   │    temperature patterns, species migration, and policy            │  │
│   │    recommendations for the next century..."                        │  │
│   │                                         │                          │  │
│   │                                         ▼                          │  │
│   │                                    [512-dim] ✗ Lossy!              │  │
│   │                                                                     │  │
│   │   Information is COMPRESSED and LOST                               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   RESULT: Performance degrades with longer sequences                       │
│                                                                             │
│   Translation Quality                                                      │
│        │                                                                   │
│      1 │████                                                               │
│        │ ████                                                              │
│        │  ████                                                             │
│        │   ████                                                            │
│        │    ████_____                                                      │
│      0 └───────────────────► Sequence Length                               │
│              10   20   30   40   50                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Attention: The Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION: THE SOLUTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IDEA: Instead of one fixed vector, let decoder LOOK BACK at             │
│         all encoder states and focus on relevant parts                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Encoder hidden states:                                           │  │
│   │   h₁     h₂     h₃     h₄     h₅     h₆                           │  │
│   │   │      │      │      │      │      │                             │  │
│   │  "The" "cat"  "sat"  "on"  "the"  "mat"                            │  │
│   │                                                                     │  │
│   │   When generating "chat" (French for cat):                         │  │
│   │                                                                     │  │
│   │   Attention weights:                                               │  │
│   │   α₁=0.1  α₂=0.7  α₃=0.1  α₄=0.05  α₅=0.03  α₆=0.02               │  │
│   │   ▼      ▼▼▼▼▼   ▼       ▼        ▼        ▼                       │  │
│   │   │      │││││   │       │        │        │                       │  │
│   │   └──────┴┴┴┴┴───┴───────┴────────┴────────┘                       │  │
│   │                  │                                                  │  │
│   │                  ▼                                                  │  │
│   │          Context = Σ αᵢ × hᵢ                                       │  │
│   │                  │                                                  │  │
│   │                  ▼                                                  │  │
│   │             "chat" 🐱                                               │  │
│   │                                                                     │  │
│   │   The model ATTENDS to "cat" when generating "chat"!               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   KEY INSIGHT:                                                             │
│   • Different output words focus on different input words                  │
│   • No information bottleneck - access all encoder states                  │
│   • Attention weights are learned end-to-end                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Attention Mechanism

### Computing Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION COMPUTATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   THREE KEY CONCEPTS:                                                      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   QUERY (Q):  "What am I looking for?"                             │  │
│   │               (decoder hidden state)                                │  │
│   │                                                                     │  │
│   │   KEY (K):    "What do I contain?"                                 │  │
│   │               (encoder hidden states)                               │  │
│   │                                                                     │  │
│   │   VALUE (V):  "What information do I provide?"                     │  │
│   │               (encoder hidden states, often same as K)             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ANALOGY: Library Search                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   You have a QUERY: "books about cats"                             │  │
│   │                                                                     │  │
│   │   Library has KEYS: [animals] [cooking] [history] [pets]           │  │
│   │                                                                     │  │
│   │   Match query to keys → highest match: [animals], [pets]           │  │
│   │                                                                     │  │
│   │   Retrieve VALUES: books from those sections                       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                                                                             │
│   ATTENTION FORMULA:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │                         Q × Kᵀ                                     │  │
│   │   Attention(Q,K,V) = softmax(────────) × V                         │  │
│   │                         √d_k                                       │  │
│   │                                                                     │  │
│   │   Step 1: Compute similarity scores (Q × Kᵀ)                       │  │
│   │   Step 2: Scale by √d_k (prevent large values)                     │  │
│   │   Step 3: Softmax (normalize to get weights)                       │  │
│   │   Step 4: Weighted sum of values                                   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   VISUAL COMPUTATION:                                                      │
│                                                                             │
│   Q (1×d)         K (n×d)           Scores (1×n)                           │
│   ┌───────┐       ┌───────┐         ┌───────────────────┐                  │
│   │ query │   ×   │  k₁   │ᵀ   =    │ s₁  s₂  s₃  ...  │                  │
│   └───────┘       │  k₂   │         └───────────────────┘                  │
│                   │  k₃   │                   │                            │
│                   │  ...  │                   ▼ softmax                    │
│                   └───────┘         ┌───────────────────┐                  │
│                                     │ α₁  α₂  α₃  ...  │ (weights sum to 1)│
│                                     └───────────────────┘                  │
│                                              │                             │
│                   V (n×d)                    │                             │
│                   ┌───────┐                  │                             │
│                   │  v₁   │                  │                             │
│                   │  v₂   │    ◄─────────────┘                             │
│                   │  v₃   │    weighted sum                                │
│                   │  ...  │         │                                      │
│                   └───────┘         ▼                                      │
│                               ┌───────────┐                                │
│                               │  context  │ (1×d)                          │
│                               └───────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Scale by √d_k?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY SCALE BY √d_k?                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PROBLEM: Without scaling, large d_k causes extreme softmax values        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Dot product of d-dimensional vectors:                            │  │
│   │                                                                     │  │
│   │   q · k = Σᵢ qᵢkᵢ                                                  │  │
│   │                                                                     │  │
│   │   If qᵢ, kᵢ ~ N(0,1):                                              │  │
│   │   • Each term qᵢkᵢ has variance 1                                  │  │
│   │   • Sum of d terms has variance d                                  │  │
│   │   • Standard deviation = √d                                        │  │
│   │                                                                     │  │
│   │   For d=512:                                                       │  │
│   │   Dot products range roughly: [-50, +50]                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   SOFTMAX WITH LARGE VALUES:                                               │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   softmax([1, 2, 3]) = [0.09, 0.24, 0.67]  ← Nice gradient         │  │
│   │                                                                     │  │
│   │   softmax([10, 20, 30]) = [0.00, 0.00, 1.00]  ← Gradient ≈ 0!      │  │
│   │                                                                     │  │
│   │   The larger values completely dominate                            │  │
│   │   Gradients vanish, training fails                                 │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   SOLUTION: Divide by √d_k                                                 │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   (q · k) / √d has variance ≈ 1                                    │  │
│   │                                                                     │  │
│   │   Scores stay in reasonable range [-3, +3]                         │  │
│   │   Softmax produces nice probability distribution                   │  │
│   │   Gradients flow properly                                          │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Self-Attention

### Concept

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SELF-ATTENTION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   KEY DIFFERENCE FROM ENCODER-DECODER ATTENTION:                           │
│                                                                             │
│   Encoder-Decoder: Query from decoder, Keys/Values from encoder            │
│   Self-Attention:  Query, Keys, Values ALL from same sequence              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input: "The cat sat on the mat"                                  │  │
│   │                                                                     │  │
│   │   For each word, we ask: "Which OTHER words are relevant to me?"   │  │
│   │                                                                     │  │
│   │   "sat" attends to:                                                │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │  "The"  "cat"  "sat"  "on"  "the"  "mat"                   │  │  │
│   │   │  0.05   0.60   0.10   0.10  0.05   0.10                    │  │  │
│   │   │   ▲      ▲▲▲    ▲      ▲     ▲      ▲                      │  │  │
│   │   │   │      │││    │      │     │      │                      │  │  │
│   │   │   └──────┴┴┴────┴──────┴─────┴──────┘                      │  │  │
│   │   │          │                                                  │  │  │
│   │   │     "sat" learns it's related to "cat" (who sat)           │  │  │
│   │   │                                                             │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   │   Every position gets a CONTEXT-AWARE representation               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   HOW IT WORKS:                                                            │
│                                                                             │
│   Input embeddings: X (n × d)                                              │
│                                                                             │
│   Project to Q, K, V using learned weight matrices:                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Q = X × W_Q    (n × d_k)                                         │  │
│   │   K = X × W_K    (n × d_k)                                         │  │
│   │   V = X × W_V    (n × d_v)                                         │  │
│   │                                                                     │  │
│   │   Same input X, different projections!                             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Then apply attention:                                                    │
│   Output = softmax(Q × Kᵀ / √d_k) × V                                      │
│                                                                             │
│                                                                             │
│   ATTENTION MATRIX VISUALIZATION:                                          │
│                                                                             │
│              Keys                                                           │
│           The  cat  sat  on  the  mat                                      │
│         ┌────────────────────────────┐                                     │
│   The   │ .9  .02  .02 .02  .02  .02 │                                     │
│   cat   │ .1  .5   .2  .05  .05  .1  │                                     │
│   sat   │ .05 .6   .1  .1   .05  .1  │  ← Attention weights               │
│   on    │ .02 .1   .3  .3   .08  .2  │                                     │
│   the   │ .02 .02  .1  .1   .7   .06 │                                     │
│   mat   │ .02 .1   .1  .2   .08  .5  │                                     │
│         └────────────────────────────┘                                     │
│   Queries                                                                  │
│                                                                             │
│   Each row = how one word attends to all others                           │
│   Rows sum to 1 (softmax)                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Head Attention

### Why Multiple Heads?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MULTI-HEAD ATTENTION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PROBLEM: One attention can only focus on one type of relationship        │
│                                                                             │
│   "The animal didn't cross the street because it was too tired"            │
│                                                                             │
│   "it" could refer to:                                                     │
│   • "animal" (correct here - syntactic)                                    │
│   • "street" (would need different context)                                │
│                                                                             │
│   We need MULTIPLE attention patterns simultaneously!                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Head 1: Syntactic relationships (subject-verb)                   │  │
│   │   Head 2: Coreference (pronouns → nouns)                          │  │
│   │   Head 3: Semantic similarity                                      │  │
│   │   Head 4: Position-based attention                                 │  │
│   │   ...                                                              │  │
│   │                                                                     │  │
│   │   Each head can learn a DIFFERENT attention pattern!               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                                                                             │
│   ARCHITECTURE:                                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input X (n × d_model)                                            │  │
│   │        │                                                            │  │
│   │   ┌────┴────┬────────┬────────┬────────┐                           │  │
│   │   │         │        │        │        │                           │  │
│   │   ▼         ▼        ▼        ▼        ▼                           │  │
│   │ ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                        │  │
│   │ │Head │  │Head │  │Head │  │Head │  │Head │   h heads              │  │
│   │ │  1  │  │  2  │  │  3  │  │ ... │  │  h  │   (typically 8)        │  │
│   │ └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                        │  │
│   │    │        │        │        │        │                           │  │
│   │    │  (n×d_k) each   │        │        │                           │  │
│   │    │        │        │        │        │                           │  │
│   │    └────────┴────────┴────────┴────────┘                           │  │
│   │                      │                                              │  │
│   │                      ▼                                              │  │
│   │              ┌───────────────┐                                     │  │
│   │              │   CONCAT      │  (n × h*d_k)                        │  │
│   │              └───────┬───────┘                                     │  │
│   │                      │                                              │  │
│   │                      ▼                                              │  │
│   │              ┌───────────────┐                                     │  │
│   │              │   Linear W_O  │  Project back to d_model            │  │
│   │              └───────┬───────┘                                     │  │
│   │                      │                                              │  │
│   │                      ▼                                              │  │
│   │               Output (n × d_model)                                 │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   FORMULAS:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   head_i = Attention(X × W_Q^i, X × W_K^i, X × W_V^i)              │  │
│   │                                                                     │  │
│   │   MultiHead(X) = Concat(head_1, ..., head_h) × W_O                 │  │
│   │                                                                     │  │
│   │   Dimensions:                                                      │  │
│   │   • d_model = 512 (model dimension)                                │  │
│   │   • h = 8 (number of heads)                                        │  │
│   │   • d_k = d_v = d_model / h = 64 (per-head dimension)             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Transformer

### Full Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐       │
│   │         ENCODER             │    │         DECODER             │       │
│   │                             │    │                             │       │
│   │   ┌───────────────────┐     │    │   ┌───────────────────┐    │       │
│   │   │  Input Embedding  │     │    │   │ Output Embedding  │    │       │
│   │   │        +          │     │    │   │        +          │    │       │
│   │   │ Positional Enc    │     │    │   │ Positional Enc    │    │       │
│   │   └─────────┬─────────┘     │    │   └─────────┬─────────┘    │       │
│   │             │               │    │             │              │       │
│   │   ┌─────────▼─────────┐     │    │   ┌─────────▼─────────┐    │       │
│   │   │                   │     │    │   │  Masked Multi-    │    │       │
│   │   │   Multi-Head      │     │    │   │  Head Attention   │    │       │
│   │   │   Attention       │     │    │   │  (self)           │    │       │
│   │   │                   │     │    │   └─────────┬─────────┘    │       │
│   │   └─────────┬─────────┘     │    │             │              │       │
│   │             │               │    │        Add & Norm         │       │
│   │        Add & Norm          │    │             │              │       │
│   │             │               │    │   ┌─────────▼─────────┐    │       │
│   │   ┌─────────▼─────────┐     │    │   │  Multi-Head       │◄───┼───┐   │
│   │   │                   │     │    │   │  Attention        │    │   │   │
│   │   │   Feed Forward    │     │    │   │  (cross)          │    │   │   │
│   │   │   Network         │     │    │   └─────────┬─────────┘    │   │   │
│   │   │                   │     │    │             │              │   │   │
│   │   └─────────┬─────────┘     │    │        Add & Norm         │   │   │
│   │             │               │    │             │              │   │   │
│   │        Add & Norm          │    │   ┌─────────▼─────────┐    │   │   │
│   │             │               │    │   │                   │    │   │   │
│   │             │    ×N         │    │   │   Feed Forward    │    │   │   │
│   │             │               │    │   │   Network         │    │   │   │
│   │             │               │    │   │                   │    │   │   │
│   │             ▼               │    │   └─────────┬─────────┘    │   │   │
│   │     Encoder Output ─────────┼────┼─────────────┼──────────────┘   │   │
│   │                             │    │        Add & Norm  ×N         │   │
│   └─────────────────────────────┘    │             │                  │   │
│                                      │             ▼                  │   │
│                                      │   ┌───────────────────┐        │   │
│                                      │   │      Linear       │        │   │
│                                      │   │     Softmax       │        │   │
│                                      │   └───────────────────┘        │   │
│                                      │                                │   │
│                                      └────────────────────────────────┘   │
│                                                                             │
│   KEY COMPONENTS:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. SELF-ATTENTION: Each position attends to all positions       │  │
│   │                                                                     │  │
│   │   2. ADD & NORM (Residual + Layer Norm):                           │  │
│   │      output = LayerNorm(x + Sublayer(x))                           │  │
│   │      • Residual: helps gradient flow                               │  │
│   │      • LayerNorm: stabilizes training                              │  │
│   │                                                                     │  │
│   │   3. FEED FORWARD NETWORK:                                         │  │
│   │      FFN(x) = ReLU(x × W₁ + b₁) × W₂ + b₂                         │  │
│   │      • Two linear layers with ReLU                                 │  │
│   │      • Applied to each position independently                      │  │
│   │      • Inner dimension typically 4× model dimension               │  │
│   │                                                                     │  │
│   │   4. MASKED ATTENTION (decoder only):                              │  │
│   │      • Prevents attending to future positions                      │  │
│   │      • Ensures autoregressive generation                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Masked Self-Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MASKED SELF-ATTENTION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHY MASK? During training, decoder sees all target tokens at once.       │
│             But at inference, we generate one at a time.                   │
│             Mask prevents "cheating" by looking at future tokens.          │
│                                                                             │
│   UNMASKED ATTENTION:              MASKED ATTENTION:                       │
│                                                                             │
│        I   love  cats              I   love  cats                          │
│   I   [✓   ✓     ✓  ]         I   [✓   ✗     ✗  ]   ← can only see "I"    │
│   love[✓   ✓     ✓  ]         love[✓   ✓     ✗  ]   ← see "I", "love"     │
│   cats[✓   ✓     ✓  ]         cats[✓   ✓     ✓  ]   ← see all previous    │
│                                                                             │
│   ✓ = can attend                   ✗ = masked (set to -∞)                  │
│                                                                             │
│   IMPLEMENTATION:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   # Create causal mask                                             │  │
│   │   mask = np.triu(np.ones((n, n)), k=1)  # Upper triangular        │  │
│   │   mask = mask * -1e9  # Large negative number                      │  │
│   │                                                                     │  │
│   │   # Apply before softmax                                           │  │
│   │   scores = (Q @ K.T) / sqrt(d_k)                                   │  │
│   │   scores = scores + mask  # Add -∞ to future positions            │  │
│   │   weights = softmax(scores)  # -∞ → 0 after softmax               │  │
│   │                                                                     │  │
│   │   Mask:           After softmax:                                   │  │
│   │   [0   -∞  -∞]    [1.0  0    0  ]                                 │  │
│   │   [0    0  -∞] →  [0.4  0.6  0  ]                                 │  │
│   │   [0    0   0]    [0.2  0.3  0.5]                                 │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Positional Encoding

### Why Position Matters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POSITIONAL ENCODING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PROBLEM: Self-attention is PERMUTATION INVARIANT                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   "The cat sat on the mat"                                         │  │
│   │   "mat the on sat cat The"                                         │  │
│   │                                                                     │  │
│   │   Without position info, attention treats these identically!       │  │
│   │   (same words, same attention patterns)                            │  │
│   │                                                                     │  │
│   │   We need to tell the model WHERE each word is.                    │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   SOLUTION: Add position information to embeddings                         │
│                                                                             │
│   SINUSOIDAL POSITIONAL ENCODING (original Transformer):                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   PE(pos, 2i)   = sin(pos / 10000^(2i/d))                          │  │
│   │   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))                          │  │
│   │                                                                     │  │
│   │   pos = position in sequence (0, 1, 2, ...)                        │  │
│   │   i = dimension index (0, 1, 2, ..., d/2)                          │  │
│   │   d = model dimension                                               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   VISUALIZATION:                                                           │
│                                                                             │
│   Position    Dimension 0   Dim 2    Dim 4    ...   Dim d-1                │
│      0        sin(0)=0      sin(0)   sin(0)         cos(...)              │
│      1        sin(1)        sin(0.1) sin(0.01)      cos(...)              │
│      2        sin(2)        sin(0.2) sin(0.02)      cos(...)              │
│      3        sin(3)        sin(0.3) sin(0.03)      cos(...)              │
│      ...                                                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Dim 0 (fast)    Dim 10 (medium)    Dim 100 (slow)                 │  │
│   │                                                                     │  │
│   │  ∿∿∿∿∿∿∿∿∿∿      ∿   ∿   ∿   ∿      ∿           ∿                 │  │
│   │                                                                     │  │
│   │  Low dims: high frequency (nearby positions differ)                │  │
│   │  High dims: low frequency (distant positions differ)              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WHY SINE/COSINE?                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   PE(pos+k) can be expressed as linear function of PE(pos)         │  │
│   │   → Model can easily learn relative positions                      │  │
│   │   → Works for sequences longer than training                       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   LEARNED POSITIONAL EMBEDDINGS (alternative):                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   pos_embed = nn.Embedding(max_seq_len, d_model)                   │  │
│   │                                                                     │  │
│   │   Learn a separate embedding for each position                     │  │
│   │   • More flexible                                                  │  │
│   │   • But limited to max_seq_len seen during training               │  │
│   │   • Used in BERT, GPT                                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   FINAL EMBEDDING:                                                         │
│                                                                             │
│   input_embedding = word_embedding + positional_encoding                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Vision Transformer (ViT)

### Applying Transformers to Images

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VISION TRANSFORMER (ViT)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CORE IDEA: Treat image as sequence of patches (like words in sentence)   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Image (224×224×3)         Patches (14×14 = 196 patches)          │  │
│   │                                                                     │  │
│   │   ┌──────────────────┐      ┌──┬──┬──┬──┬──┬──┬──┐                 │  │
│   │   │                  │      │1 │2 │3 │4 │5 │6 │7 │                 │  │
│   │   │                  │      ├──┼──┼──┼──┼──┼──┼──┤                 │  │
│   │   │      🐱          │  →   │8 │9 │10│11│12│13│14│  ...            │  │
│   │   │                  │      ├──┼──┼──┼──┼──┼──┼──┤                 │  │
│   │   │                  │      │..│..│..│..│..│..│..│                 │  │
│   │   └──────────────────┘      └──┴──┴──┴──┴──┴──┴──┘                 │  │
│   │                                                                     │  │
│   │   Each 16×16 patch = 16×16×3 = 768 pixels                          │  │
│   │   Flatten + Linear projection → 768-dim embedding                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ARCHITECTURE:                                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input Image                                                       │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   ┌───────────────────────────────────────────────────────────┐    │  │
│   │   │              Split into Patches (16×16)                   │    │  │
│   │   └───────────────────────────────────────────────────────────┘    │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   ┌───────────────────────────────────────────────────────────┐    │  │
│   │   │         Linear Projection (Patch Embedding)               │    │  │
│   │   │         Each patch: 768 → d_model                         │    │  │
│   │   └───────────────────────────────────────────────────────────┘    │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   ┌──────────────────────────────────────────────────────────┐     │  │
│   │   │ [CLS] │ Patch 1 │ Patch 2 │ ... │ Patch 196 │            │     │  │
│   │   │       +         +         +     +           │            │     │  │
│   │   │ Pos 0 │ Pos 1   │ Pos 2   │ ... │ Pos 196   │            │     │  │
│   │   └──────────────────────────────────────────────────────────┘     │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   ┌───────────────────────────────────────────────────────────┐    │  │
│   │   │              Transformer Encoder (×12)                    │    │  │
│   │   │                                                           │    │  │
│   │   │   Multi-Head Self-Attention + FFN + Add&Norm             │    │  │
│   │   │                                                           │    │  │
│   │   └───────────────────────────────────────────────────────────┘    │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   ┌───────────────────────────────────────────────────────────┐    │  │
│   │   │    Take [CLS] token output → MLP Head → Class prediction  │    │  │
│   │   └───────────────────────────────────────────────────────────┘    │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   [CLS] TOKEN:                                                             │
│   • Special learnable embedding prepended to sequence                     │
│   • Aggregates information from all patches via attention                 │
│   • Used for final classification (like BERT)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Coding Exercises

### Exercise 1: Scaled Dot-Product Attention

```python
#==============================================================================
# EXERCISE 1: SCALED DOT-PRODUCT ATTENTION
#==============================================================================

import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries of shape (..., seq_len_q, d_k)
        K: Keys of shape (..., seq_len_k, d_k)
        V: Values of shape (..., seq_len_k, d_v)
        mask: Optional mask of shape (..., seq_len_q, seq_len_k)

    Returns:
        output: Attention output of shape (..., seq_len_q, d_v)
        attention_weights: Weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    #===========================================================================
    # TODO: Implement scaled dot-product attention
    # 1. Compute attention scores: Q @ K^T
    # 2. Scale by sqrt(d_k)
    # 3. Apply mask (if provided)
    # 4. Apply softmax
    # 5. Multiply by V
    #===========================================================================

    # Step 1 & 2: Compute scaled scores
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)

    # Step 3: Apply mask
    if mask is not None:
        scores = scores + mask * -1e9

    # Step 4: Apply softmax
    attention_weights = softmax(scores, axis=-1)

    # Step 5: Multiply by values
    output = np.matmul(attention_weights, V)

    #===========================================================================

    return output, attention_weights


def create_causal_mask(seq_len):
    """
    Create causal mask for decoder self-attention.

    Args:
        seq_len: Length of sequence

    Returns:
        mask: Upper triangular mask of shape (seq_len, seq_len)
    """
    #===========================================================================
    # TODO: Create mask where future positions have 1, past have 0
    #===========================================================================

    mask = np.triu(np.ones((seq_len, seq_len)), k=1)

    #===========================================================================

    return mask


#==============================================================================
# EXERCISE 2: MULTI-HEAD ATTENTION
#==============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weight matrices
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape

        #=======================================================================
        # TODO: Reshape and transpose to split heads
        #=======================================================================

        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, d_k)

        #=======================================================================

        return x

    def combine_heads(self, x):
        """
        Combine heads back to (batch_size, seq_len, d_model).

        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape

        #=======================================================================
        # TODO: Transpose and reshape to combine heads
        #=======================================================================

        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, d_k)
        x = x.reshape(batch_size, seq_len, self.d_model)

        #=======================================================================

        return x

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            Q: Query input of shape (batch_size, seq_len_q, d_model)
            K: Key input of shape (batch_size, seq_len_k, d_model)
            V: Value input of shape (batch_size, seq_len_k, d_model)
            mask: Optional mask

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Weights for visualization
        """
        batch_size = Q.shape[0]

        #=======================================================================
        # TODO: Implement multi-head attention forward pass
        # 1. Linear projections
        # 2. Split heads
        # 3. Apply attention
        # 4. Combine heads
        # 5. Final linear projection
        #=======================================================================

        # Linear projections
        Q = Q @ self.W_Q  # (batch, seq_len, d_model)
        K = K @ self.W_K
        V = V @ self.W_V

        # Split heads
        Q = self.split_heads(Q)  # (batch, heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply attention
        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads
        attn_output = self.combine_heads(attn_output)  # (batch, seq_len, d_model)

        # Final projection
        output = attn_output @ self.W_O

        #=======================================================================

        return output, attention_weights


#==============================================================================
# EXERCISE 3: POSITIONAL ENCODING
#==============================================================================

def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encoding.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        pe: Positional encoding of shape (max_len, d_model)
    """
    #===========================================================================
    # TODO: Implement sinusoidal positional encoding
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    #===========================================================================

    pe = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # Even indices
    pe[:, 1::2] = np.cos(position * div_term)  # Odd indices

    #===========================================================================

    return pe


#==============================================================================
# EXERCISE 4: TRANSFORMER ENCODER LAYER
#==============================================================================

def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer normalization.

    Args:
        x: Input of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)

    Returns:
        Normalized output of same shape
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def feed_forward(x, W1, b1, W2, b2):
    """
    Position-wise feed-forward network.

    Args:
        x: Input of shape (batch, seq_len, d_model)
        W1: First layer weights (d_model, d_ff)
        b1: First layer bias (d_ff,)
        W2: Second layer weights (d_ff, d_model)
        b2: Second layer bias (d_model,)

    Returns:
        Output of shape (batch, seq_len, d_model)
    """
    #===========================================================================
    # TODO: Implement FFN: ReLU(x @ W1 + b1) @ W2 + b2
    #===========================================================================

    hidden = np.maximum(0, x @ W1 + b1)  # ReLU
    output = hidden @ W2 + b2

    #===========================================================================

    return output


class TransformerEncoderLayer:
    """
    Single Transformer encoder layer.
    """

    def __init__(self, d_model, num_heads, d_ff):
        """
        Initialize encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        self.d_model = d_model

        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

        # Layer norm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass for encoder layer.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Encoded output of shape (batch, seq_len, d_model)
        """
        #=======================================================================
        # TODO: Implement encoder layer with residual connections and layer norm
        # 1. Self-attention + Add & Norm
        # 2. Feed-forward + Add & Norm
        #=======================================================================

        # Self-attention sublayer
        attn_output, _ = self.mha.forward(x, x, x, mask)
        x = layer_norm(x + attn_output, self.gamma1, self.beta1)

        # Feed-forward sublayer
        ff_output = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        output = layer_norm(x + ff_output, self.gamma2, self.beta2)

        #=======================================================================

        return output


#==============================================================================
# EXERCISE 5: VISION TRANSFORMER (ViT)
#==============================================================================

class VisionTransformer:
    """
    Vision Transformer for image classification.
    """

    def __init__(self, image_size=224, patch_size=16, num_classes=10,
                 d_model=768, num_heads=12, num_layers=12, d_ff=3072):
        """
        Initialize ViT.

        Args:
            image_size: Input image size (assumes square)
            patch_size: Size of each patch
            num_classes: Number of output classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
        """
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model

        # Patch embedding: Linear projection of flattened patches
        patch_dim = 3 * patch_size * patch_size  # RGB * patch_size^2
        self.patch_embed = np.random.randn(patch_dim, d_model) * 0.02

        # Class token
        self.cls_token = np.random.randn(1, 1, d_model) * 0.02

        # Position embeddings (learned)
        self.pos_embed = np.random.randn(1, self.num_patches + 1, d_model) * 0.02

        # Transformer encoder layers
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Classification head
        self.head = np.random.randn(d_model, num_classes) * 0.02

    def patchify(self, images):
        """
        Split images into patches.

        Args:
            images: Batch of images (batch, channels, height, width)

        Returns:
            patches: Flattened patches (batch, num_patches, patch_dim)
        """
        batch_size, C, H, W = images.shape
        P = self.patch_size

        #=======================================================================
        # TODO: Extract patches from images
        # Reshape to (batch, num_patches, patch_dim)
        #=======================================================================

        # Reshape to (batch, H/P, P, W/P, P, C)
        images = images.reshape(batch_size, C, H // P, P, W // P, P)

        # Transpose to (batch, H/P, W/P, P, P, C)
        images = images.transpose(0, 2, 4, 3, 5, 1)

        # Flatten patches
        patches = images.reshape(batch_size, -1, C * P * P)

        #=======================================================================

        return patches

    def forward(self, images):
        """
        Forward pass for ViT.

        Args:
            images: Batch of images (batch, channels, height, width)

        Returns:
            logits: Class logits (batch, num_classes)
        """
        batch_size = images.shape[0]

        #=======================================================================
        # TODO: Implement ViT forward pass
        # 1. Patchify and embed
        # 2. Add class token
        # 3. Add positional embeddings
        # 4. Apply transformer layers
        # 5. Take CLS token output
        # 6. Classification head
        #=======================================================================

        # Step 1: Patchify and embed
        patches = self.patchify(images)  # (batch, num_patches, patch_dim)
        embeddings = patches @ self.patch_embed  # (batch, num_patches, d_model)

        # Step 2: Prepend class token
        cls_tokens = np.repeat(self.cls_token, batch_size, axis=0)
        embeddings = np.concatenate([cls_tokens, embeddings], axis=1)

        # Step 3: Add positional embeddings
        embeddings = embeddings + self.pos_embed

        # Step 4: Apply transformer layers
        x = embeddings
        for layer in self.layers:
            x = layer.forward(x)

        # Step 5: Take CLS token output
        cls_output = x[:, 0]  # (batch, d_model)

        # Step 6: Classification head
        logits = cls_output @ self.head  # (batch, num_classes)

        #=======================================================================

        return logits
```

---

## Business Applications

### Production Transformer System

```python
#==============================================================================
# BUSINESS APPLICATION: Document Understanding System
#==============================================================================

import numpy as np
from typing import List, Dict, Tuple
import re

class DocumentAnalyzer:
    """
    Transformer-based document understanding system.

    Use Cases:
    - Legal: Contract analysis and clause extraction
    - Finance: Financial report summarization
    - Healthcare: Medical record understanding
    - HR: Resume screening and matching
    """

    def __init__(self, vocab_size=30000, d_model=256, num_heads=8,
                 num_layers=4, max_seq_len=512):
        """Initialize the document analyzer."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Vocabulary
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        # Initialize transformer components
        self.word_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed = positional_encoding(max_seq_len, d_model)

        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ]

        # Task-specific heads
        self.classifier_head = np.random.randn(d_model, 5) * 0.02  # 5 doc types
        self.ner_head = np.random.randn(d_model, 10) * 0.02  # 10 entity types

        # Document type labels
        self.doc_types = ['contract', 'report', 'email', 'memo', 'other']
        self.entity_types = ['PERSON', 'ORG', 'DATE', 'MONEY', 'LOCATION',
                            'PRODUCT', 'EVENT', 'LEGAL', 'PERCENTAGE', 'O']

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', ' ', text)
        tokens = text.split()
        return tokens[:self.max_seq_len - 2]  # Leave room for CLS/SEP

    def encode(self, text: str) -> np.ndarray:
        """Encode text to token indices."""
        tokens = ['<CLS>'] + self.tokenize(text) + ['<SEP>']
        indices = [self.word_to_idx.get(t, self.word_to_idx['<UNK>'])
                   for t in tokens]

        # Pad to max length
        while len(indices) < self.max_seq_len:
            indices.append(self.word_to_idx['<PAD>'])

        return np.array(indices)

    def get_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        """Get word + positional embeddings."""
        word_emb = self.word_embed[token_ids]
        seq_len = token_ids.shape[-1]
        pos_emb = self.pos_embed[:seq_len]

        return word_emb + pos_emb

    def encode_document(self, text: str) -> np.ndarray:
        """
        Encode document using transformer.

        Args:
            text: Document text

        Returns:
            encodings: Contextualized embeddings (seq_len, d_model)
        """
        # Tokenize and embed
        token_ids = self.encode(text)
        embeddings = self.get_embeddings(token_ids)

        # Add batch dimension
        x = embeddings[np.newaxis, :]  # (1, seq_len, d_model)

        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer.forward(x)

        return x[0]  # Remove batch dimension

    def classify_document(self, text: str) -> Dict:
        """
        Classify document type.

        Args:
            text: Document text

        Returns:
            Classification results with probabilities
        """
        # Encode document
        encodings = self.encode_document(text)

        # Use CLS token for classification
        cls_encoding = encodings[0]  # First token is CLS

        # Classify
        logits = cls_encoding @ self.classifier_head
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        # Get top prediction
        pred_idx = np.argmax(probs)

        return {
            'document_type': self.doc_types[pred_idx],
            'confidence': float(probs[pred_idx]),
            'all_probabilities': {
                doc_type: float(probs[i])
                for i, doc_type in enumerate(self.doc_types)
            }
        }

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from document.

        Args:
            text: Document text

        Returns:
            List of extracted entities with positions
        """
        # Encode document
        encodings = self.encode_document(text)
        tokens = ['<CLS>'] + self.tokenize(text) + ['<SEP>']

        # Predict entity type for each token
        logits = encodings @ self.ner_head  # (seq_len, num_entity_types)
        predictions = np.argmax(logits, axis=-1)

        # Extract entities
        entities = []
        current_entity = None
        current_tokens = []

        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            entity_type = self.entity_types[pred]

            if entity_type != 'O':
                if current_entity == entity_type:
                    current_tokens.append(token)
                else:
                    if current_entity is not None:
                        entities.append({
                            'text': ' '.join(current_tokens),
                            'type': current_entity,
                            'start_token': i - len(current_tokens),
                            'end_token': i
                        })
                    current_entity = entity_type
                    current_tokens = [token]
            else:
                if current_entity is not None:
                    entities.append({
                        'text': ' '.join(current_tokens),
                        'type': current_entity,
                        'start_token': i - len(current_tokens),
                        'end_token': i
                    })
                    current_entity = None
                    current_tokens = []

        return entities

    def analyze_document(self, text: str) -> Dict:
        """
        Complete document analysis.

        Args:
            text: Document text

        Returns:
            Comprehensive analysis results
        """
        # Classify document
        classification = self.classify_document(text)

        # Extract entities
        entities = self.extract_entities(text)

        # Get summary statistics
        tokens = self.tokenize(text)
        word_count = len(tokens)

        # Entity statistics
        entity_counts = {}
        for entity in entities:
            entity_type = entity['type']
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        return {
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'word_count': word_count,
            'classification': classification,
            'entities': entities[:10],  # Top 10 entities
            'entity_summary': entity_counts,
            'key_phrases': self._extract_key_phrases(text, entities)
        }

    def _extract_key_phrases(self, text: str, entities: List[Dict]) -> List[str]:
        """Extract key phrases based on entities and attention."""
        # Simple approach: return unique entity texts
        phrases = list(set(e['text'] for e in entities if len(e['text']) > 2))
        return phrases[:5]

    def compare_documents(self, text1: str, text2: str) -> Dict:
        """
        Compare two documents using their embeddings.

        Args:
            text1, text2: Documents to compare

        Returns:
            Similarity analysis
        """
        # Get CLS embeddings for both documents
        enc1 = self.encode_document(text1)[0]  # CLS token
        enc2 = self.encode_document(text2)[0]

        # Cosine similarity
        similarity = np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2))

        # Get classifications
        class1 = self.classify_document(text1)
        class2 = self.classify_document(text2)

        return {
            'cosine_similarity': float(similarity),
            'same_type': class1['document_type'] == class2['document_type'],
            'document1_type': class1['document_type'],
            'document2_type': class2['document_type'],
            'similarity_interpretation': self._interpret_similarity(similarity)
        }

    def _interpret_similarity(self, score: float) -> str:
        """Interpret similarity score."""
        if score > 0.9:
            return "Nearly identical documents"
        elif score > 0.7:
            return "Very similar documents"
        elif score > 0.5:
            return "Moderately similar documents"
        elif score > 0.3:
            return "Somewhat related documents"
        else:
            return "Different documents"


# Demo
def demo_document_analyzer():
    """Demonstrate the document analyzer."""

    analyzer = DocumentAnalyzer(vocab_size=1000, d_model=64, num_heads=4, num_layers=2)

    # Build simple vocabulary
    sample_docs = [
        "This contract agreement is between Party A and Party B dated January 2024.",
        "Q3 Financial Report shows revenue of 5 million dollars with 20 percent growth.",
        "Dear John, please review the attached proposal by Friday. Best regards, Sarah.",
    ]

    idx = 4  # Start after special tokens
    for doc in sample_docs:
        for token in analyzer.tokenize(doc):
            if token not in analyzer.word_to_idx:
                analyzer.word_to_idx[token] = idx
                analyzer.idx_to_word[idx] = token
                idx += 1
                if idx >= analyzer.vocab_size:
                    break

    print("=" * 60)
    print("DOCUMENT ANALYZER DEMO")
    print("=" * 60)

    # Analyze a contract
    contract = """
    This Service Agreement is entered into as of March 15, 2024, between
    Acme Corporation, a Delaware company, and TechStart Inc., a California
    startup. The total contract value is 2.5 million dollars over 24 months.
    """

    print("\nDocument Analysis:")
    print("-" * 40)
    result = analyzer.analyze_document(contract)
    print(f"Document Type: {result['classification']['document_type']}")
    print(f"Confidence: {result['classification']['confidence']:.1%}")
    print(f"Word Count: {result['word_count']}")
    print(f"Entities Found: {len(result['entities'])}")

    # Compare documents
    report = """
    Quarterly revenue reached 10 million with strong growth in the enterprise
    segment. Customer acquisition increased by 35 percent year over year.
    """

    print("\n" + "=" * 60)
    print("DOCUMENT COMPARISON")
    print("=" * 60)

    comparison = analyzer.compare_documents(contract, report)
    print(f"\nSimilarity: {comparison['cosine_similarity']:.3f}")
    print(f"Interpretation: {comparison['similarity_interpretation']}")
    print(f"Same Type: {comparison['same_type']}")


if __name__ == '__main__':
    demo_document_analyzer()
```

---

## Summary: Week 7 Checklist

### Concepts You Should Understand
- [ ] Attention as Query-Key-Value lookup
- [ ] Scaled dot-product attention
- [ ] Why we scale by √d_k
- [ ] Self-attention vs. cross-attention
- [ ] Multi-head attention
- [ ] Transformer architecture
- [ ] Residual connections and layer normalization
- [ ] Positional encoding (sinusoidal vs. learned)
- [ ] Masked attention for autoregressive generation
- [ ] Vision Transformer (ViT) patch-based approach

### Skills You Should Have
- [ ] Implement scaled dot-product attention
- [ ] Implement multi-head attention
- [ ] Implement positional encoding
- [ ] Build a transformer encoder layer
- [ ] Build a Vision Transformer

### Key Formulas
```
Attention(Q,K,V) = softmax(QK^T / √d_k) × V

MultiHead(Q,K,V) = Concat(head_1,...,head_h) × W_O
  where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Next Steps
After completing Week 7, you're ready for:
- **Week 8-10**: Detection, Segmentation, Generative Models, and Advanced Topics
- Applying transformers to complex vision tasks
- Understanding generative AI (GANs, Diffusion, LLMs)
