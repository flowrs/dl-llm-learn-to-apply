# Module 5: Attention & Transformers

## Learning Objectives

By the end of this module, you will understand:
- The attention mechanism and why it revolutionized deep learning
- Self-attention and its computational properties
- Multi-head attention for richer representations
- The complete Transformer architecture
- Positional encoding for sequence order
- Vision Transformers (ViT)
- KV Cache for efficient generation

---

## 5.1 Motivation: The Bottleneck Problem

### The Seq2Seq Limitation

In Module 4, we learned about Sequence-to-Sequence models that use an encoder-decoder architecture. The fundamental problem with basic Seq2Seq is the **information bottleneck**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     THE SEQ2SEQ BOTTLENECK PROBLEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Source: "The quick brown fox jumps over the lazy dog"                      │
│                                                                             │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐       │
│  │ The │──▶│quick│──▶│brown│──▶│ fox │──▶│jumps│──▶│ ... │──▶│ dog │       │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘       │
│     │         │         │         │         │         │         │          │
│     ▼         ▼         ▼         ▼         ▼         ▼         ▼          │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐       │
│  │ h₁  │──▶│ h₂  │──▶│ h₃  │──▶│ h₄  │──▶│ h₅  │──▶│ ... │──▶│ h₉  │       │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └──┬──┘       │
│                                                                  │          │
│                                                                  ▼          │
│                                                           ┌───────────┐     │
│                                                           │  CONTEXT  │     │
│                                                           │  VECTOR   │     │
│                                                           │    c      │     │
│                                                           └─────┬─────┘     │
│                                                                 │           │
│           ALL information must pass through this single vector! │           │
│                                                                 │           │
│                                                                 ▼           │
│                                                           ┌───────────┐     │
│                                                           │  DECODER  │     │
│                                                           └───────────┘     │
│                                                                 │           │
│                                                                 ▼           │
│  Target: "Le rapide renard brun saute par-dessus le chien paresseux"       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

PROBLEM: The context vector c (typically 256-1024 dimensions) must encode:
- All 9 words' meanings
- Word relationships
- Syntactic structure
- Order information

For long sequences, information is inevitably lost!
```

### Evidence of the Bottleneck

Research showed that Seq2Seq performance **degrades with sequence length**:

```
┌─────────────────────────────────────────────────────────────────┐
│               TRANSLATION QUALITY vs SENTENCE LENGTH            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BLEU Score                                                     │
│     40 │ ████████████                                           │
│        │ ████████████                                           │
│     35 │ ████████████ ████████                                  │
│        │ ████████████ ████████                                  │
│     30 │ ████████████ ████████ ██████                           │
│        │ ████████████ ████████ ██████                           │
│     25 │ ████████████ ████████ ██████ █████                     │
│        │ ████████████ ████████ ██████ █████ ████                │
│     20 │ ████████████ ████████ ██████ █████ ████ ███            │
│        │ ████████████ ████████ ██████ █████ ████ ███ ██         │
│     15 │ ████████████ ████████ ██████ █████ ████ ███ ██ █       │
│        └────────────────────────────────────────────────────    │
│            10       20       30       40   50  60  70  80       │
│                        Sentence Length (words)                  │
│                                                                 │
│  As sentences get longer, quality drops significantly           │
│  because the fixed-size context vector can't hold everything    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Key Insight: Dynamic Context

Instead of compressing everything into one vector, what if the decoder could **look back at all encoder states** and dynamically decide which parts are relevant?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE ATTENTION SOLUTION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Encoder States (all preserved):                                            │
│                                                                             │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                            │
│  │ h₁  │   │ h₂  │   │ h₃  │   │ h₄  │   │ h₅  │                            │
│  │"The"│   │"cat"│   │"sat"│   │ "on"│   │"mat"│                            │
│  └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘                            │
│     │         │         │         │         │                               │
│     │    ┌────┴─────────┴─────────┴─────────┴────┐                          │
│     │    │                                       │                          │
│     └────┤         ATTENTION MECHANISM           │                          │
│          │  "Which states matter right now?"     │                          │
│          │                                       │                          │
│          └───────────────────┬───────────────────┘                          │
│                              │                                              │
│                              ▼                                              │
│                    ┌───────────────────┐                                    │
│                    │  Context vector   │                                    │
│                    │  (different for   │                                    │
│                    │  each decoder     │                                    │
│                    │  step!)           │                                    │
│                    └─────────┬─────────┘                                    │
│                              │                                              │
│                              ▼                                              │
│                    ┌───────────────────┐                                    │
│                    │     Decoder       │                                    │
│                    │   Current Step    │                                    │
│                    └───────────────────┘                                    │
│                                                                             │
│  When generating "chat" (French for "cat"):                                 │
│    - High attention to h₂ ("cat") ──────▶ α₂ = 0.75                        │
│    - Low attention to others ───────────▶ α₁ = 0.10, α₃ = 0.08, ...        │
│                                                                             │
│  When generating "tapis" (French for "mat"):                                │
│    - High attention to h₅ ("mat") ──────▶ α₅ = 0.80                        │
│    - Low attention to others                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 The Attention Mechanism

### The Intuition: Soft Dictionary Lookup

Think of attention as a **differentiable database query**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARD vs SOFT LOOKUP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HARD LOOKUP (Traditional Dictionary):                                      │
│  ─────────────────────────────────────                                      │
│                                                                             │
│    Query: "cat"                                                             │
│                                                                             │
│    Dictionary:                                                              │
│    ┌─────────────────────────────────────┐                                  │
│    │  Key        │  Value                │                                  │
│    ├─────────────┼───────────────────────┤                                  │
│    │  "dog"      │  "canine animal"      │   ✗ No match                    │
│    │  "cat"      │  "feline animal"      │   ✓ Exact match → Return this   │
│    │  "bird"     │  "flying animal"      │   ✗ No match                    │
│    └─────────────┴───────────────────────┘                                  │
│                                                                             │
│    Result: "feline animal"                                                  │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  SOFT LOOKUP (Attention):                                                   │
│  ─────────────────────────                                                  │
│                                                                             │
│    Query: [0.9, 0.1]  (vector representing "cat-like")                      │
│                                                                             │
│    Database:                                                                │
│    ┌─────────────────────────────────────────────────────────────┐          │
│    │  Key              │  Similarity  │  Weight   │  Value       │          │
│    ├───────────────────┼──────────────┼───────────┼──────────────┤          │
│    │  [0.1, 0.9] dog   │     0.18     │   0.05    │  v_dog       │          │
│    │  [0.8, 0.2] cat   │     0.74     │   0.70    │  v_cat       │          │
│    │  [0.5, 0.5] pet   │     0.50     │   0.25    │  v_pet       │          │
│    └───────────────────┴──────────────┴───────────┴──────────────┘          │
│                                         (softmax)                           │
│                                                                             │
│    Result: 0.05 × v_dog + 0.70 × v_cat + 0.25 × v_pet                       │
│            (weighted combination of ALL values!)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Key insight: Soft lookup is DIFFERENTIABLE!
  - We can backpropagate through it
  - The model learns WHAT to look for
```

### Query, Key, Value Framework

The attention mechanism uses three components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUERY, KEY, VALUE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  QUERY (Q): "What am I looking for?"                                        │
│  ──────────────────────────────────                                         │
│    - Comes from the current position/state                                  │
│    - Represents the "question" being asked                                  │
│    - Example: Decoder hidden state asking "what word should I translate?"   │
│                                                                             │
│  KEY (K): "What does each item contain?"                                    │
│  ───────────────────────────────────────                                    │
│    - One key per item in the "database"                                     │
│    - Used to compute similarity with query                                  │
│    - Example: Encoder hidden states as keys                                 │
│                                                                             │
│  VALUE (V): "What information to retrieve?"                                 │
│  ─────────────────────────────────────────                                  │
│    - The actual content to be retrieved                                     │
│    - One value per item (same number as keys)                               │
│    - Example: Encoder hidden states as values (often K = V)                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                                                                 │        │
│  │    Q ──────┐                                                    │        │
│  │            │                                                    │        │
│  │            ▼                                                    │        │
│  │    ┌──────────────┐      ┌──────────────┐                       │        │
│  │    │  Similarity  │      │    Values    │                       │        │
│  │    │  Computation │      │              │                       │        │
│  │    │              │      │   V₁ ────────┼──┐                    │        │
│  │    │   Q · K₁ ────┼──┐   │   V₂ ────────┼──┼──┐                 │        │
│  │    │   Q · K₂ ────┼──┼── │   V₃ ────────┼──┼──┼──┐              │        │
│  │    │   Q · K₃ ────┼──┼── │              │  │  │  │              │        │
│  │    │              │  │   └──────────────┘  │  │  │              │        │
│  │    └──────────────┘  │                     │  │  │              │        │
│  │                      │                     │  │  │              │        │
│  │                      ▼                     │  │  │              │        │
│  │               ┌──────────────┐             │  │  │              │        │
│  │               │   Softmax    │             │  │  │              │        │
│  │               │              │             │  │  │              │        │
│  │               │  α₁ = 0.1 ───┼─────────────┘  │  │              │        │
│  │               │  α₂ = 0.7 ───┼────────────────┘  │              │        │
│  │               │  α₃ = 0.2 ───┼───────────────────┘              │        │
│  │               │              │                                  │        │
│  │               └──────────────┘                                  │        │
│  │                      │                                          │        │
│  │                      ▼                                          │        │
│  │     Output = α₁·V₁ + α₂·V₂ + α₃·V₃ = Σᵢ αᵢ·Vᵢ                   │        │
│  │                                                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Scaled Dot-Product Attention

The most common attention mechanism used in Transformers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCALED DOT-PRODUCT ATTENTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V               │
│                                                                             │
│  Step-by-step breakdown:                                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 1: Compute Attention Scores (QKᵀ)                             │    │
│  │  ═══════════════════════════════════════                            │    │
│  │                                                                     │    │
│  │   Q (queries)          K (keys)              Scores                 │    │
│  │   [seq_q × d_k]        [seq_k × d_k]         [seq_q × seq_k]        │    │
│  │                                                                     │    │
│  │   ┌─────────┐          ┌─────────┐ᵀ          ┌───────────────┐      │    │
│  │   │ q₁ ───▶ │    ×     │ k₁      │     =     │ q₁·k₁  q₁·k₂ │      │    │
│  │   │ q₂ ───▶ │          │ k₂      │           │ q₂·k₁  q₂·k₂ │      │    │
│  │   └─────────┘          └─────────┘           └───────────────┘      │    │
│  │                             │                                       │    │
│  │                             ▼                                       │    │
│  │                        ┌─────────┐                                  │    │
│  │                        │ k₁  k₂  │                                  │    │
│  │                        │ ↓   ↓   │                                  │    │
│  │                        └─────────┘                                  │    │
│  │                                                                     │    │
│  │   Each score[i,j] = how much query i should attend to key j        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 2: Scale by √d_k                                              │    │
│  │  ════════════════════════                                           │    │
│  │                                                                     │    │
│  │   scaled_scores = scores / √d_k                                     │    │
│  │                                                                     │    │
│  │   WHY SCALE?                                                        │    │
│  │   ──────────                                                        │    │
│  │   If q and k have components ~ N(0, 1):                             │    │
│  │     • q·k = Σᵢ qᵢkᵢ has variance d_k (sum of d_k unit variances)   │    │
│  │     • Large d_k → large variance → extreme values                   │    │
│  │     • Extreme values → softmax becomes very peaked                  │    │
│  │     • Peaked softmax → tiny gradients (saturation)                  │    │
│  │                                                                     │    │
│  │   Scaling by √d_k normalizes variance back to 1:                    │    │
│  │     • Var(q·k / √d_k) = Var(q·k) / d_k = d_k / d_k = 1             │    │
│  │                                                                     │    │
│  │   Example with d_k = 64:                                            │    │
│  │     • Without scaling: q·k might be ~8 (√64) on average            │    │
│  │     • With scaling: (q·k)/8 keeps values ~1                         │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 3: Apply Softmax                                              │    │
│  │  ═══════════════════════                                            │    │
│  │                                                                     │    │
│  │   attention_weights = softmax(scaled_scores, dim=-1)                │    │
│  │                                                                     │    │
│  │   ┌─────────────────┐         ┌─────────────────┐                   │    │
│  │   │  2.1   0.5  0.8 │         │ 0.65  0.13  0.22│  (row sums to 1)  │    │
│  │   │  0.3   1.9  0.4 │  ───▶   │ 0.15  0.72  0.13│  (row sums to 1)  │    │
│  │   │  1.1   0.2  2.3 │         │ 0.23  0.10  0.67│  (row sums to 1)  │    │
│  │   └─────────────────┘         └─────────────────┘                   │    │
│  │      scaled scores             attention weights                    │    │
│  │                                                                     │    │
│  │   Each row: probability distribution over keys                      │    │
│  │   Weights are non-negative and sum to 1                             │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 4: Weighted Sum of Values                                     │    │
│  │  ══════════════════════════════                                     │    │
│  │                                                                     │    │
│  │   output = attention_weights × V                                    │    │
│  │                                                                     │    │
│  │   Attention Weights        Values              Output               │    │
│  │   [seq_q × seq_k]          [seq_k × d_v]       [seq_q × d_v]        │    │
│  │                                                                     │    │
│  │   ┌───────────────┐        ┌─────────┐         ┌─────────┐          │    │
│  │   │ 0.65 0.13 0.22│   ×    │  v₁ ──▶ │    =    │  o₁ ──▶ │          │    │
│  │   │ 0.15 0.72 0.13│        │  v₂ ──▶ │         │  o₂ ──▶ │          │    │
│  │   │ 0.23 0.10 0.67│        │  v₃ ──▶ │         │  o₃ ──▶ │          │    │
│  │   └───────────────┘        └─────────┘         └─────────┘          │    │
│  │                                                                     │    │
│  │   o₁ = 0.65·v₁ + 0.13·v₂ + 0.22·v₃                                  │    │
│  │   o₂ = 0.15·v₁ + 0.72·v₂ + 0.13·v₃                                  │    │
│  │   o₃ = 0.23·v₁ + 0.10·v₂ + 0.67·v₃                                  │    │
│  │                                                                     │    │
│  │   Each output is a weighted combination of all values               │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Numerical Example

Let's work through a concrete example:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION NUMERICAL EXAMPLE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Given:                                                                     │
│    Q = [1, 0]              (single query, d_k = 2)                          │
│    K = [[1, 0],            (3 keys)                                         │
│         [0, 1],                                                             │
│         [1, 1]]                                                             │
│    V = [[1, 2],            (3 values, d_v = 2)                              │
│         [3, 4],                                                             │
│         [5, 6]]                                                             │
│                                                                             │
│  Step 1: Compute scores (Q · Kᵀ)                                            │
│  ─────────────────────────────────                                          │
│    score(Q, K₁) = [1,0] · [1,0] = 1×1 + 0×0 = 1                            │
│    score(Q, K₂) = [1,0] · [0,1] = 1×0 + 0×1 = 0                            │
│    score(Q, K₃) = [1,0] · [1,1] = 1×1 + 0×1 = 1                            │
│                                                                             │
│    scores = [1, 0, 1]                                                       │
│                                                                             │
│  Step 2: Scale by √d_k = √2 ≈ 1.414                                         │
│  ──────────────────────────────────                                         │
│    scaled = [1/1.414, 0/1.414, 1/1.414]                                     │
│           = [0.707, 0, 0.707]                                               │
│                                                                             │
│  Step 3: Softmax                                                            │
│  ─────────────────                                                          │
│    exp(0.707) = 2.028                                                       │
│    exp(0)     = 1.000                                                       │
│    exp(0.707) = 2.028                                                       │
│                                                                             │
│    sum = 2.028 + 1.000 + 2.028 = 5.056                                      │
│                                                                             │
│    weights = [2.028/5.056, 1.000/5.056, 2.028/5.056]                        │
│            = [0.401, 0.198, 0.401]                                          │
│                                                                             │
│  Step 4: Weighted sum of values                                             │
│  ─────────────────────────────────                                          │
│    output = 0.401 × [1,2] + 0.198 × [3,4] + 0.401 × [5,6]                   │
│           = [0.401, 0.802] + [0.594, 0.792] + [2.005, 2.406]                │
│           = [3.000, 4.000]                                                  │
│                                                                             │
│  Interpretation:                                                            │
│  ───────────────                                                            │
│    • Query [1,0] matches K₁=[1,0] and K₃=[1,1] equally well (both have 1   │
│      in the first dimension)                                                │
│    • K₂=[0,1] has 0 in first dimension, so it's less relevant              │
│    • Output is a weighted blend, dominated by V₁ and V₃                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Python Implementation

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.

    Args:
        Q: Queries of shape [batch, num_heads, seq_len_q, d_k]
           or [batch, seq_len_q, d_k]
        K: Keys of shape [batch, num_heads, seq_len_k, d_k]
           or [batch, seq_len_k, d_k]
        V: Values of shape [batch, num_heads, seq_len_k, d_v]
           or [batch, seq_len_k, d_v]
        mask: Optional mask of shape broadcastable to [batch, seq_q, seq_k]
              Use 0 for positions to mask (will become -inf before softmax)

    Returns:
        output: Attended values [batch, ..., seq_len_q, d_v]
        attention_weights: Attention distribution [batch, ..., seq_len_q, seq_len_k]
    """
    # Get the dimension of keys for scaling
    d_k = K.shape[-1]

    # Step 1: Compute attention scores
    # Q @ K^T: [..., seq_q, d_k] @ [..., d_k, seq_k] -> [..., seq_q, seq_k]
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask if provided (for causal attention or padding)
    if mask is not None:
        # Mask positions with -inf so softmax gives 0
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax to get attention weights (probabilities)
    attention_weights = F.softmax(scores, dim=-1)

    # Handle case where entire row is masked (all -inf -> NaN after softmax)
    attention_weights = attention_weights.masked_fill(
        torch.isnan(attention_weights), 0.0
    )

    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# Example usage
if __name__ == "__main__":
    # Simple example from the numerical walkthrough
    Q = torch.tensor([[[1.0, 0.0]]])  # [1, 1, 2]
    K = torch.tensor([[[1.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0]]])  # [1, 3, 2]
    V = torch.tensor([[[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]]])  # [1, 3, 2]

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Attention weights: {weights}")
    print(f"Output: {output}")
    # Expected: weights ≈ [0.401, 0.198, 0.401], output ≈ [3.0, 4.0]
```

---

## 5.3 Self-Attention

### What is Self-Attention?

In **self-attention**, the queries, keys, and values all come from the **same sequence**. Each position can attend to every other position (including itself).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SELF-ATTENTION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input sequence: "The cat sat"                                              │
│                                                                             │
│         x₁           x₂           x₃                                        │
│       "The"        "cat"        "sat"                                       │
│         │            │            │                                         │
│         ▼            ▼            ▼                                         │
│     ┌───────┐    ┌───────┐    ┌───────┐                                     │
│     │ Embed │    │ Embed │    │ Embed │                                     │
│     └───┬───┘    └───┬───┘    └───┬───┘                                     │
│         │            │            │                                         │
│    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐                                    │
│    │    │    │  │    │    │  │    │    │                                    │
│    ▼    ▼    ▼  ▼    ▼    ▼  ▼    ▼    ▼                                    │
│   Wq   Wk   Wv Wq   Wk   Wv Wq   Wk   Wv    (learned projections)           │
│    │    │    │  │    │    │  │    │    │                                    │
│    ▼    ▼    ▼  ▼    ▼    ▼  ▼    ▼    ▼                                    │
│   Q₁   K₁   V₁ Q₂   K₂   V₂ Q₃   K₃   V₃                                    │
│    │    │    │  │    │    │  │    │    │                                    │
│    └────┼────┼──┴────┼────┼──┴────┼────┼──────────────┐                     │
│         │    │       │    │       │    │              │                     │
│         └────┴───────┴────┴───────┴────┘              │                     │
│                        │                              │                     │
│                        ▼                              ▼                     │
│              ┌─────────────────────┐        ┌─────────────────┐             │
│              │   Q = [Q₁, Q₂, Q₃]  │        │  V = [V₁,V₂,V₃] │             │
│              │   K = [K₁, K₂, K₃]  │        │                 │             │
│              └──────────┬──────────┘        └────────┬────────┘             │
│                         │                            │                      │
│                         ▼                            │                      │
│              ┌─────────────────────┐                 │                      │
│              │   Attention Scores  │                 │                      │
│              │      Q × Kᵀ         │                 │                      │
│              │                     │                 │                      │
│              │  ┌───────────────┐  │                 │                      │
│              │  │ to: 1   2   3 │  │                 │                      │
│              │  │from:          │  │                 │                      │
│              │  │  1  .8  .1  .1│  │                 │                      │
│              │  │  2  .2  .6  .2│  │                 │                      │
│              │  │  3  .1  .3  .6│  │                 │                      │
│              │  └───────────────┘  │                 │                      │
│              └──────────┬──────────┘                 │                      │
│                         │                            │                      │
│                         ▼                            ▼                      │
│              ┌──────────────────────────────────────────┐                   │
│              │         Attention × V                    │                   │
│              │                                          │                   │
│              │  y₁ = 0.8·V₁ + 0.1·V₂ + 0.1·V₃          │                   │
│              │  y₂ = 0.2·V₁ + 0.6·V₂ + 0.2·V₃          │                   │
│              │  y₃ = 0.1·V₁ + 0.3·V₂ + 0.6·V₃          │                   │
│              │                                          │                   │
│              └──────────────────┬───────────────────────┘                   │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌────────────────────────┐                               │
│                    │  Output: [y₁, y₂, y₃]  │                               │
│                    └────────────────────────┘                               │
│                                                                             │
│  Each output yᵢ is a CONTEXTUALIZED representation:                         │
│    • y₂ ("cat") contains information from "The", "cat", and "sat"          │
│    • The model learns WHAT information to gather from each position        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Self-Attention is Revolutionary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              SELF-ATTENTION vs PREVIOUS APPROACHES                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: How to model dependencies between distant positions?              │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  RNN: Sequential Processing                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  Position:  1 ──▶ 2 ──▶ 3 ──▶ 4 ──▶ 5 ──▶ 6 ──▶ 7 ──▶ 8                    │
│                                                                             │
│  For position 8 to "see" position 1:                                        │
│    • Information must flow through 7 sequential steps                       │
│    • Path length = O(n)                                                     │
│    • Information degrades (vanishing gradient)                              │
│    • Cannot parallelize (must wait for previous step)                       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  CNN: Fixed Receptive Field                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  Layer 1:  [1 2 3] [2 3 4] [3 4 5] [4 5 6] [5 6 7] [6 7 8]                  │
│               │       │       │       │       │       │                     │
│  Layer 2:      [   ●   ●   ●   ]   [   ●   ●   ●   ]                        │
│                       │                   │                                 │
│  Layer 3:              [       ●       ●       ]                            │
│                                   │                                         │
│                                   ▼                                         │
│                            Global context                                   │
│                                                                             │
│  For position 8 to "see" position 1:                                        │
│    • Need multiple layers (log_k(n) for kernel size k)                      │
│    • Path length = O(log n)                                                 │
│    • Some parallelization possible within each layer                        │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  SELF-ATTENTION: Direct Connections                                         │
│  ──────────────────────────────────                                         │
│                                                                             │
│           1       2       3       4       5       6       7       8         │
│           ●───────●───────●───────●───────●───────●───────●───────●         │
│           │╲     ╱│╲     ╱│╲     ╱│╲     ╱│╲     ╱│╲     ╱│╲     ╱│         │
│           │ ╲   ╱ │ ╲   ╱ │ ╲   ╱ │ ╲   ╱ │ ╲   ╱ │ ╲   ╱ │ ╲   ╱ │         │
│           │  ╲ ╱  │  ╲ ╱  │  ╲ ╱  │  ╲ ╱  │  ╲ ╱  │  ╲ ╱  │  ╲ ╱  │         │
│           │   ╳   │   ╳   │   ╳   │   ╳   │   ╳   │   ╳   │   ╳   │         │
│           │  ╱ ╲  │  ╱ ╲  │  ╱ ╲  │  ╱ ╲  │  ╱ ╲  │  ╱ ╲  │  ╱ ╲  │         │
│           │ ╱   ╲ │ ╱   ╲ │ ╱   ╲ │ ╱   ╲ │ ╱   ╲ │ ╱   ╲ │ ╱   ╲ │         │
│           │╱     ╲│╱     ╲│╱     ╲│╱     ╲│╱     ╲│╱     ╲│╱     ╲│         │
│           ●───────●───────●───────●───────●───────●───────●───────●         │
│           1       2       3       4       5       6       7       8         │
│                                                                             │
│  EVERY position directly attends to EVERY other position!                   │
│                                                                             │
│  For position 8 to "see" position 1:                                        │
│    • Direct connection in single operation                                  │
│    • Path length = O(1)                                                     │
│    • No information degradation from long paths                             │
│    • Fully parallelizable (all positions computed simultaneously)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complexity Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPLEXITY COMPARISON                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  n = sequence length                                                        │
│  d = model dimension (hidden size)                                          │
│  k = kernel size (for CNN)                                                  │
│                                                                             │
│  ┌─────────────────┬──────────────┬─────────────┬─────────────────────────┐ │
│  │   Operation     │ Computation  │ Sequential  │ Maximum Path Length     │ │
│  │                 │ per Layer    │ Operations  │ (to connect any 2 pos)  │ │
│  ├─────────────────┼──────────────┼─────────────┼─────────────────────────┤ │
│  │ Self-Attention  │   O(n²·d)    │    O(1)     │       O(1)              │ │
│  │ Recurrent       │   O(n·d²)    │    O(n)     │       O(n)              │ │
│  │ Convolutional   │   O(k·n·d²)  │    O(1)     │       O(log_k(n))       │ │
│  └─────────────────┴──────────────┴─────────────┴─────────────────────────┘ │
│                                                                             │
│  KEY TRADEOFF:                                                              │
│  ─────────────                                                              │
│                                                                             │
│  Self-Attention:                                                            │
│    ✓ O(1) path length (great for long-range dependencies)                  │
│    ✓ O(1) sequential ops (fully parallelizable)                            │
│    ✗ O(n²) computation (quadratic in sequence length!)                     │
│                                                                             │
│  For a sequence of 1000 tokens with d=512:                                  │
│    Self-Attention: 1000² × 512 = 512 million operations                    │
│    RNN:            1000 × 512² = 262 million operations                    │
│                                                                             │
│  Self-attention is FASTER despite more ops because of parallelism,         │
│  but memory grows quadratically, limiting very long sequences.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Permutation Equivariance

A crucial property of self-attention:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERMUTATION EQUIVARIANCE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Self-attention treats input as a SET, not a SEQUENCE!                      │
│                                                                             │
│  DEFINITION: A function f is permutation equivariant if:                    │
│              f(π(x)) = π(f(x))                                              │
│              where π is any permutation                                     │
│                                                                             │
│  Example:                                                                   │
│  ─────────                                                                  │
│                                                                             │
│  Original input:     ["The", "cat", "sat"]                                  │
│  Self-attention:     [y_The, y_cat, y_sat]                                  │
│                                                                             │
│  Permuted input:     ["cat", "The", "sat"]                                  │
│  Self-attention:     [y_cat, y_The, y_sat]   ← outputs are permuted same!   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  Input order:  [A, B, C]                                            │    │
│  │       ↓                                                             │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │             Self-Attention Layer                            │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │       ↓                                                             │    │
│  │  Output order: [A', B', C']                                         │    │
│  │                                                                     │    │
│  │  ═══════════════════════════════════════════════════════════════   │    │
│  │                                                                     │    │
│  │  Input order:  [B, A, C]  (swapped A and B)                         │    │
│  │       ↓                                                             │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │             Self-Attention Layer                            │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │       ↓                                                             │    │
│  │  Output order: [B', A', C']  (outputs also swapped!)                │    │
│  │                                                                     │    │
│  │  The VALUES of A' and B' are UNCHANGED regardless of position       │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  IMPLICATION: Self-attention has NO inherent notion of position!           │
│                                                                             │
│  "The cat sat on the mat" and "mat the on sat cat The"                      │
│  would produce the SAME representations (just reordered)!                   │
│                                                                             │
│  SOLUTION: Add positional encoding (see Section 5.5)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.4 Multi-Head Attention

### Why Multiple Heads?

Single-head attention captures one type of relationship. But language has many relationship types:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY MULTI-HEAD ATTENTION?                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sentence: "The animal didn't cross the street because it was too tired"   │
│                                                                             │
│  What relationships might the word "it" need to attend to?                  │
│                                                                             │
│  HEAD 1 - COREFERENCE:                                                      │
│  ──────────────────────                                                     │
│    "it" → "animal" (what does "it" refer to?)                               │
│                                                                             │
│    The animal didn't cross the street because it was too tired              │
│        ───────                                   ──                          │
│           │                                       │                         │
│           └───────────────────────────────────────┘                         │
│                    High attention weight                                    │
│                                                                             │
│  HEAD 2 - SYNTACTIC STRUCTURE:                                              │
│  ────────────────────────────                                               │
│    "it" → "because" (what clause am I part of?)                             │
│                                                                             │
│    The animal didn't cross the street because it was too tired              │
│                                       ───────  ──                           │
│                                          │      │                           │
│                                          └──────┘                           │
│                                                                             │
│  HEAD 3 - SEMANTIC PROPERTY:                                                │
│  ───────────────────────────                                                │
│    "it" → "tired" (what property is being attributed?)                      │
│                                                                             │
│    The animal didn't cross the street because it was too tired              │
│                                                ──          ─────            │
│                                                 │             │             │
│                                                 └─────────────┘             │
│                                                                             │
│  SINGLE HEAD: Can only capture ONE of these relationships                   │
│  MULTI-HEAD:  Each head specializes in different relationship types         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Head Attention Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: X ∈ ℝ^{n × d_model}                                                 │
│                                                                             │
│         X (input)                                                           │
│            │                                                                │
│     ┌──────┴──────┐                                                         │
│     │             │                                                         │
│     ▼             ▼                                                         │
│  ┌─────┐       ┌─────┐                                                      │
│  │ W_Q │       │W_K,V│                                                      │
│  └──┬──┘       └──┬──┘                                                      │
│     │             │                                                         │
│     ▼             ▼                                                         │
│     Q          K, V                                                         │
│     │             │                                                         │
│     │    ┌────────┴────────┐                                                │
│     │    │                 │                                                │
│     ▼    ▼                 ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         SPLIT INTO h HEADS                           │   │
│  │                                                                      │   │
│  │    Q, K, V ∈ ℝ^{n × d_model}                                         │   │
│  │         ↓                                                            │   │
│  │    Reshape to [n × h × d_k] where d_k = d_model / h                  │   │
│  │         ↓                                                            │   │
│  │    ┌─────────┬─────────┬─────────┬─────────┐                         │   │
│  │    │ Head 1  │ Head 2  │ Head 3  │  ...h   │                         │   │
│  │    │ Q₁,K₁,V₁│ Q₂,K₂,V₂│ Q₃,K₃,V₃│         │                         │   │
│  │    │ [n×d_k] │ [n×d_k] │ [n×d_k] │         │                         │   │
│  │    └────┬────┴────┬────┴────┬────┴────┬────┘                         │   │
│  │         │         │         │         │                              │   │
│  │         ▼         ▼         ▼         ▼                              │   │
│  │    ┌─────────┬─────────┬─────────┬─────────┐                         │   │
│  │    │Attention│Attention│Attention│Attention│  (parallel!)            │   │
│  │    │ head 1  │ head 2  │ head 3  │ head h  │                         │   │
│  │    └────┬────┴────┬────┴────┬────┴────┬────┘                         │   │
│  │         │         │         │         │                              │   │
│  │         ▼         ▼         ▼         ▼                              │   │
│  │    ┌─────────────────────────────────────────────────────────────┐   │   │
│  │    │              CONCATENATE ALL HEADS                          │   │   │
│  │    │                                                             │   │   │
│  │    │  [head₁ | head₂ | head₃ | ... | head_h]                     │   │   │
│  │    │  [n×d_k]  [n×d_k]  [n×d_k]      [n×d_k]                      │   │   │
│  │    │                     ↓                                       │   │   │
│  │    │              [n × h×d_k] = [n × d_model]                     │   │   │
│  │    └───────────────────────┬─────────────────────────────────────┘   │   │
│  │                            │                                         │   │
│  │                            ▼                                         │   │
│  │                     ┌──────────────┐                                 │   │
│  │                     │     W_O      │  Output projection              │   │
│  │                     │ [d_model ×   │  Mixes information from         │   │
│  │                     │  d_model]    │  all heads                      │   │
│  │                     └──────┬───────┘                                 │   │
│  │                            │                                         │   │
│  └────────────────────────────┼─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│                        Output [n × d_model]                                 │
│                                                                             │
│  FORMULA:                                                                   │
│  ────────                                                                   │
│  MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O                        │
│                                                                             │
│  where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)                         │
│                                                                             │
│  Typical values (BERT-base):                                                │
│    d_model = 768                                                            │
│    h = 12 heads                                                             │
│    d_k = 768 / 12 = 64 per head                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Visualizing What Heads Learn

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION HEAD SPECIALIZATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  From analyzing trained Transformer models, researchers found heads         │
│  specialize in different linguistic phenomena:                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  HEAD TYPE           │  EXAMPLE ATTENTION PATTERN                   │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                      │                                              │    │
│  │  POSITIONAL          │  "The cat sat on the mat"                    │    │
│  │  (nearby words)      │        ↖  ↑  ↗                               │    │
│  │                      │   Each word attends to neighbors             │    │
│  │                      │                                              │    │
│  │  SYNTACTIC           │  "The cat that I saw sat"                    │    │
│  │  (grammatical)       │        ───────────────→                      │    │
│  │                      │   "cat" attends to "sat" (subject-verb)      │    │
│  │                      │                                              │    │
│  │  COREFERENCE         │  "John said he was tired"                    │    │
│  │  (pronouns)          │    ────────→                                 │    │
│  │                      │   "he" attends strongly to "John"            │    │
│  │                      │                                              │    │
│  │  SEMANTIC            │  "The bank by the river"                     │    │
│  │  (meaning)           │        ───────────→                          │    │
│  │                      │   "bank" attends to "river" (disambiguates)  │    │
│  │                      │                                              │    │
│  │  DELIMITER           │  "[CLS] sentence [SEP]"                      │    │
│  │  (special tokens)    │   All words attend to [CLS]/[SEP]            │    │
│  │                      │                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  KEY INSIGHT: Having multiple heads lets the model capture ALL of these    │
│  relationships simultaneously, then combine them in the output projection.  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Head Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    The key insight is that we can compute all heads in parallel by
    reshaping tensors, rather than using separate linear layers.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V (all combined for efficiency)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_k, d_model]
            mask: [batch_size, 1, 1, seq_len_k] or [batch_size, 1, seq_len_q, seq_len_k]

        Returns:
            output: [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Step 1: Linear projections
        Q = self.W_q(query)  # [batch, seq_q, d_model]
        K = self.W_k(key)    # [batch, seq_k, d_model]
        V = self.W_v(value)  # [batch, seq_k, d_model]

        # Step 2: Reshape for multi-head attention
        # [batch, seq, d_model] -> [batch, seq, num_heads, d_k] -> [batch, num_heads, seq, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Compute attention scores
        # [batch, heads, seq_q, d_k] @ [batch, heads, d_k, seq_k] -> [batch, heads, seq_q, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 4: Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 5: Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Store for visualization
        self.attention_weights = attention_weights.detach()

        # Step 6: Apply attention to values
        # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, d_k] -> [batch, heads, seq_q, d_k]
        context = torch.matmul(attention_weights, V)

        # Step 7: Reshape back: [batch, heads, seq_q, d_k] -> [batch, seq_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Step 8: Final output projection
        output = self.W_o(context)

        return output


# Example usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)

    # Self-attention: Q = K = V
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {mha.attention_weights.shape}")
    # Expected: [2, 10, 512], [2, 10, 512], [2, 8, 10, 10]
```

---

## 5.5 Positional Encoding

### The Position Problem

Self-attention is permutation equivariant—it doesn't know position! We need to explicitly inject position information.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE POSITION PROBLEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Without positional information:                                            │
│                                                                             │
│  "Dog bites man"        Self-Attention        "Man bites dog"               │
│        │                     │                       │                      │
│        │                     │                       │                      │
│   {Dog, bites, man}    ────────────────→     Same representations!          │
│        │                     │                       │                      │
│        ▼                     ▼                       ▼                      │
│   Embeddings only    Bag of embeddings         Identical output             │
│   encode MEANING,    (order lost!)             (but meanings are            │
│   not POSITION                                 completely different!)       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  SOLUTION: Add position-dependent signals to embeddings                     │
│                                                                             │
│   Input embeddings:     [e₁,    e₂,    e₃]                                  │
│                           +      +      +                                   │
│   Position encodings:   [p₁,    p₂,    p₃]                                  │
│                           =      =      =                                   │
│   Model input:          [e₁+p₁, e₂+p₂, e₃+p₃]                               │
│                                                                             │
│   Now the model can distinguish positions!                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sinusoidal Positional Encoding

The original Transformer uses sine and cosine functions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINUSOIDAL POSITIONAL ENCODING                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))                              │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))                              │
│                                                                             │
│  where:                                                                     │
│    pos = position in sequence (0, 1, 2, ...)                                │
│    i = dimension index (0, 1, 2, ..., d_model/2)                            │
│    d_model = model dimension                                                │
│                                                                             │
│  VISUALIZATION (d_model = 8, positions 0-7):                                │
│  ─────────────────────────────────────────────                              │
│                                                                             │
│  Position│ dim0   dim1   dim2   dim3   dim4   dim5   dim6   dim7           │
│  ────────┼────────────────────────────────────────────────────────          │
│     0    │  0.00  1.00   0.00   1.00   0.00   1.00   0.00   1.00            │
│     1    │  0.84  0.54   0.10   0.99   0.01   1.00   0.00   1.00            │
│     2    │  0.91 -0.42   0.20   0.98   0.02   1.00   0.00   1.00            │
│     3    │  0.14 -0.99   0.30   0.95   0.03   1.00   0.00   1.00            │
│     4    │ -0.76 -0.65   0.39   0.92   0.04   1.00   0.00   1.00            │
│     5    │ -0.96  0.28   0.48   0.88   0.05   1.00   0.00   1.00            │
│     6    │ -0.28  0.96   0.56   0.83   0.06   1.00   0.00   1.00            │
│     7    │  0.66  0.75   0.64   0.77   0.07   1.00   0.00   1.00            │
│          │                                                                  │
│          │  ← Fast oscillation →    ← Medium →      ← Slow →               │
│                                                                             │
│  ASCII WAVE VISUALIZATION:                                                  │
│                                                                             │
│  dim 0 (high freq):   /‾‾\__/‾‾\__/‾‾\__    (changes rapidly)              │
│  dim 4 (med freq):    /‾‾‾‾‾‾\______/‾‾‾     (changes moderately)          │
│  dim 7 (low freq):    _______________/‾‾     (changes slowly)              │
│                       0  2  4  6  8 10 12    (position)                    │
│                                                                             │
│  KEY PROPERTIES:                                                            │
│  ───────────────                                                            │
│                                                                             │
│  1. UNIQUE: Each position has a unique encoding                             │
│                                                                             │
│  2. BOUNDED: Values always in [-1, 1] (sin/cos range)                       │
│                                                                             │
│  3. RELATIVE: PE(pos+k) can be expressed as linear function of PE(pos)      │
│     This allows the model to learn relative positions!                      │
│                                                                             │
│     Proof: sin(pos+k) = sin(pos)cos(k) + cos(pos)sin(k)                     │
│            cos(pos+k) = cos(pos)cos(k) - sin(pos)sin(k)                     │
│                                                                             │
│  4. EXTRAPOLATION: Can extend to longer sequences than seen in training     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sinusoidal Encoding Implementation

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    Uses sine and cosine functions of different frequencies to create
    unique position encodings that allow the model to attend to relative positions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Position indices [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Dimension indices for computing frequencies
        # div_term = 10000^(2i/d_model) computed in log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even dimensions, cos to odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add batch dimension: [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            x + positional_encoding [batch_size, seq_len, d_model]
        """
        # Add positional encoding (broadcasts over batch dimension)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Visualization helper
def visualize_positional_encoding(d_model=64, max_len=100):
    """Visualize positional encoding as a heatmap pattern."""
    pe = SinusoidalPositionalEncoding(d_model, max_len)

    # Get the encoding matrix
    encoding = pe.pe[0].numpy()  # [max_len, d_model]

    print("Positional Encoding Visualization")
    print("=" * 60)
    print(f"Shape: {encoding.shape}")
    print(f"Position 0: {encoding[0, :8].round(2)}")  # First 8 dims
    print(f"Position 1: {encoding[1, :8].round(2)}")
    print(f"Position 2: {encoding[2, :8].round(2)}")

    return encoding
```

### Learned Positional Embeddings

Many modern models (BERT, GPT) use learned position embeddings:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEARNED POSITIONAL EMBEDDINGS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Instead of fixed sinusoidal patterns, LEARN position embeddings            │
│  just like we learn word embeddings.                                        │
│                                                                             │
│  Position Embedding Table:                                                  │
│  ┌──────────┬────────────────────────────────────────┐                      │
│  │ Position │  Learned Embedding Vector              │                      │
│  ├──────────┼────────────────────────────────────────┤                      │
│  │    0     │  [0.12, -0.34, 0.56, ...]             │                      │
│  │    1     │  [0.23, -0.45, 0.67, ...]             │                      │
│  │    2     │  [0.34, -0.56, 0.78, ...]             │                      │
│  │   ...    │  ...                                   │                      │
│  │   511    │  [-0.11, 0.22, -0.33, ...]            │                      │
│  └──────────┴────────────────────────────────────────┘                      │
│                                                                             │
│  COMPARISON:                                                                │
│  ───────────                                                                │
│                                                                             │
│  ┌─────────────────────┬─────────────────────────────────────────────────┐  │
│  │     Sinusoidal      │          Learned                                │  │
│  ├─────────────────────┼─────────────────────────────────────────────────┤  │
│  │ ✓ No parameters     │ ✗ max_len × d_model parameters                  │  │
│  │ ✓ Extrapolates      │ ✗ Cannot exceed max_len                         │  │
│  │ ✓ Mathematical      │ ✓ Can learn task-specific patterns              │  │
│  │   properties        │ ✓ Often slightly better empirically             │  │
│  └─────────────────────┴─────────────────────────────────────────────────┘  │
│                                                                             │
│  Most modern models use LEARNED embeddings (despite extrapolation limit)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings as used in BERT, GPT, etc.

    Simply looks up position index in a learned embedding table.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings [max_len, d_model]
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)

        # Look up position embeddings and add to input
        x = x + self.pos_embedding(positions)

        return self.dropout(x)
```

---

## 5.6 The Transformer Architecture

### Complete Transformer Architecture

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), combines all the components we've covered:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      THE COMPLETE TRANSFORMER ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────────────────────┐        ┌──────────────────────────────┐          │
│  │         ENCODER              │        │         DECODER              │          │
│  │         (×N layers)          │        │         (×N layers)          │          │
│  │                              │        │                              │          │
│  │  ┌────────────────────────┐  │        │  ┌────────────────────────┐  │          │
│  │  │   INPUT EMBEDDING      │  │        │  │   OUTPUT EMBEDDING     │  │          │
│  │  │   + Positional Enc     │  │        │  │   + Positional Enc     │  │          │
│  │  └───────────┬────────────┘  │        │  └───────────┬────────────┘  │          │
│  │              │               │        │              │               │          │
│  │              ▼               │        │              ▼               │          │
│  │  ┌────────────────────────┐  │        │  ┌────────────────────────┐  │          │
│  │  │                        │  │        │  │  MASKED SELF-ATTENTION │  │          │
│  │  │    SELF-ATTENTION      │  │        │  │  (Causal: can't see    │  │          │
│  │  │    (Bidirectional)     │  │        │  │   future tokens)       │  │          │
│  │  │                        │  │        │  │                        │  │          │
│  │  └───────────┬────────────┘  │        │  └───────────┬────────────┘  │          │
│  │              │               │        │              │               │          │
│  │      ┌───────┴───────┐       │        │      ┌───────┴───────┐       │          │
│  │      │  Add & Norm   │       │        │      │  Add & Norm   │       │          │
│  │      └───────┬───────┘       │        │      └───────┬───────┘       │          │
│  │              │               │        │              │               │          │
│  │              ▼               │        │              ▼               │          │
│  │  ┌────────────────────────┐  │        │  ┌────────────────────────┐  │          │
│  │  │                        │  │   ┌────┼──│   CROSS-ATTENTION      │  │          │
│  │  │    FEED-FORWARD        │  │   │    │  │                        │  │          │
│  │  │    NETWORK (FFN)       │  │   │    │  │   Q: from decoder      │  │          │
│  │  │                        │  │   │    │  │   K,V: from encoder ◄──┼──┼──────┐   │
│  │  └───────────┬────────────┘  │   │    │  │                        │  │      │   │
│  │              │               │   │    │  └───────────┬────────────┘  │      │   │
│  │      ┌───────┴───────┐       │   │    │              │               │      │   │
│  │      │  Add & Norm   │       │   │    │      ┌───────┴───────┐       │      │   │
│  │      └───────┬───────┘       │   │    │      │  Add & Norm   │       │      │   │
│  │              │               │   │    │      └───────┬───────┘       │      │   │
│  └──────────────┼───────────────┘   │    │              │               │      │   │
│                 │                   │    │              ▼               │      │   │
│                 │                   │    │  ┌────────────────────────┐  │      │   │
│                 │                   │    │  │                        │  │      │   │
│                 │                   │    │  │    FEED-FORWARD        │  │      │   │
│                 └───────────────────┘    │  │    NETWORK (FFN)       │  │      │   │
│                                          │  │                        │  │      │   │
│                 Encoder output           │  └───────────┬────────────┘  │      │   │
│                 provides K,V             │              │               │      │   │
│                 for cross-attention      │      ┌───────┴───────┐       │      │   │
│                                          │      │  Add & Norm   │       │      │   │
│                                          │      └───────┬───────┘       │      │   │
│                                          │              │               │      │   │
│                                          └──────────────┼───────────────┘      │   │
│                                                         │                      │   │
│                                                         ▼                      │   │
│                                              ┌────────────────────┐            │   │
│                                              │  LINEAR + SOFTMAX  │            │   │
│                                              │  (Vocabulary size) │            │   │
│                                              └─────────┬──────────┘            │   │
│                                                        │                       │   │
│                                                        ▼                       │   │
│                                              Output Probabilities              │   │
│                                              (next token prediction)           │   │
│                                                                                │   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Encoder Block

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRANSFORMER ENCODER BLOCK (DETAIL)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    Input: X ∈ ℝ^{n × d_model}                               │
│                              │                                              │
│                              │                                              │
│      ┌───────────────────────┼───────────────────────┐                      │
│      │                       │                       │                      │
│      │                       ▼                       │                      │
│      │       ┌───────────────────────────────┐       │                      │
│      │       │    MULTI-HEAD SELF-ATTENTION  │       │                      │
│      │       │                               │       │                      │
│      │       │   Q = X·W_Q                   │       │                      │
│      │       │   K = X·W_K   (all from X)    │       │                      │
│      │       │   V = X·W_V                   │       │                      │
│      │       │                               │       │                      │
│      │       │   Output = Attention(Q,K,V)   │       │                      │
│      │       └───────────────┬───────────────┘       │                      │
│      │                       │                       │                      │
│      │                       ▼                       │                      │
│      │              ┌─────────────────┐              │                      │
│      │              │     Dropout     │              │                      │
│      │              └────────┬────────┘              │                      │
│      │                       │                       │                      │
│      └──────────────────────►+◄──────────────────────┘                      │
│                   (Residual Connection)                                     │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────┐                                      │
│                    │   Layer Norm    │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│      ┌──────────────────────┼──────────────────────┐                        │
│      │                      │                      │                        │
│      │                      ▼                      │                        │
│      │      ┌───────────────────────────────┐      │                        │
│      │      │     FEED-FORWARD NETWORK      │      │                        │
│      │      │                               │      │                        │
│      │      │   FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂   │                        │
│      │      │                               │      │                        │
│      │      │   W₁: [d_model × d_ff]        │      │                        │
│      │      │   W₂: [d_ff × d_model]        │      │                        │
│      │      │   d_ff = 4 × d_model typically│      │                        │
│      │      │                               │      │                        │
│      │      └───────────────┬───────────────┘      │                        │
│      │                      │                      │                        │
│      │                      ▼                      │                        │
│      │             ┌─────────────────┐             │                        │
│      │             │     Dropout     │             │                        │
│      │             └────────┬────────┘             │                        │
│      │                      │                      │                        │
│      └─────────────────────►+◄─────────────────────┘                        │
│                  (Residual Connection)                                      │
│                             │                                               │
│                             ▼                                               │
│                   ┌─────────────────┐                                       │
│                   │   Layer Norm    │                                       │
│                   └────────┬────────┘                                       │
│                            │                                                │
│                            ▼                                                │
│                    Output: ∈ ℝ^{n × d_model}                                │
│                    (same shape as input!)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Residual Connections?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RESIDUAL CONNECTIONS IN TRANSFORMERS                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WITHOUT Residual Connections:                                              │
│  ────────────────────────────                                               │
│                                                                             │
│    x ──▶ [Layer 1] ──▶ [Layer 2] ──▶ ... ──▶ [Layer N] ──▶ output          │
│                                                                             │
│    Problem: Gradients must flow through EVERY layer                         │
│    • Gradients can vanish or explode                                        │
│    • Deep networks become very hard to train                                │
│    • Each layer must learn the ENTIRE transformation                        │
│                                                                             │
│  WITH Residual Connections:                                                 │
│  ─────────────────────────                                                  │
│                                                                             │
│    x ──┬──▶ [Layer] ──┬──▶ output                                           │
│        │              │                                                     │
│        └──────────────┘                                                     │
│           (skip connection)                                                 │
│                                                                             │
│    output = x + Layer(x)                                                    │
│                                                                             │
│  Benefits:                                                                  │
│  ─────────                                                                  │
│                                                                             │
│  1. GRADIENT HIGHWAY                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  ∂L/∂x = ∂L/∂output × ∂output/∂x                                │     │
│     │        = ∂L/∂output × (1 + ∂Layer(x)/∂x)                        │     │
│     │                        ↑                                        │     │
│     │                        This "1" provides direct gradient path   │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. LEARNING RESIDUAL FUNCTIONS                                             │
│     Layer only needs to learn the DIFFERENCE from identity                  │
│     • Easier to learn "do nothing" (output zeros) if needed                 │
│     • Small refinements are easier than full transformations                │
│                                                                             │
│  3. ENABLES VERY DEEP NETWORKS                                              │
│     GPT-3: 96 layers                                                        │
│     Without residuals, this would be nearly impossible to train             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Normalization vs Batch Normalization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                LAYER NORM vs BATCH NORM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Tensor shape: [Batch, Sequence, Features]                                  │
│                                                                             │
│  BATCH NORMALIZATION:                                                       │
│  ─────────────────────                                                      │
│                                                                             │
│        Feature 1   Feature 2   Feature 3                                    │
│           │           │           │                                         │
│    ┌──────┼───────────┼───────────┼──────┐                                  │
│    │  ████████████│███████████│██████████│  Batch 1                         │
│    │  ████████████│███████████│██████████│  Batch 2                         │
│    │  ████████████│███████████│██████████│  Batch 3                         │
│    │  ████████████│███████████│██████████│  Batch 4                         │
│    └──────┼───────────┼───────────┼──────┘                                  │
│           │           │           │                                         │
│    Normalize down columns (across batch)                                    │
│                                                                             │
│    Problem for sequences:                                                   │
│    • Variable sequence lengths                                              │
│    • Batch statistics unreliable at inference                               │
│    • Doesn't work well for RNNs/Transformers                                │
│                                                                             │
│  LAYER NORMALIZATION:                                                       │
│  ─────────────────────                                                      │
│                                                                             │
│    ┌──────────────────────────────────────┐                                 │
│    │  ████████████████████████████████████│──▶ Normalize this sample        │
│    │  ████████████████████████████████████│──▶ Normalize this sample        │
│    │  ████████████████████████████████████│──▶ Normalize this sample        │
│    │  ████████████████████████████████████│──▶ Normalize this sample        │
│    └──────────────────────────────────────┘                                 │
│       Feature 1  Feature 2  Feature 3                                       │
│                                                                             │
│    Normalize across features (within each sample)                           │
│                                                                             │
│    Benefits:                                                                │
│    • Works the same at training and inference                               │
│    • Independent of batch size                                              │
│    • Works naturally with variable-length sequences                         │
│                                                                             │
│  FORMULA:                                                                   │
│  ────────                                                                   │
│                                                                             │
│    LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β                               │
│                                                                             │
│    where μ, σ computed over feature dimension                               │
│    γ, β are learned scale and shift parameters                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feed-Forward Network (FFN)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEED-FORWARD NETWORK IN TRANSFORMERS                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FFN(x) = Activation(x·W₁ + b₁)·W₂ + b₂                                     │
│                                                                             │
│  Typical dimensions:                                                        │
│    d_model = 512 (or 768, 1024, etc.)                                       │
│    d_ff = 4 × d_model = 2048 (expansion factor of 4)                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │   Input x                                                           │    │
│  │   [seq_len × d_model]                                               │    │
│  │        │                                                            │    │
│  │        │  W₁: [d_model × d_ff]                                      │    │
│  │        ▼                                                            │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │              LINEAR PROJECTION #1                           │   │    │
│  │   │                                                             │   │    │
│  │   │   x·W₁ + b₁ → [seq_len × d_ff]                              │   │    │
│  │   │                                                             │   │    │
│  │   │   512 → 2048 (EXPAND by 4×)                                 │   │    │
│  │   │                                                             │   │    │
│  │   └─────────────────────────┬───────────────────────────────────┘   │    │
│  │                             │                                       │    │
│  │                             ▼                                       │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │              ACTIVATION (GELU/ReLU)                         │   │    │
│  │   │                                                             │   │    │
│  │   │   GELU(x) = x × Φ(x)   (smooth approximation of ReLU)       │   │    │
│  │   │                                                             │   │    │
│  │   │   ReLU                    GELU                              │   │    │
│  │   │     │                       │                               │   │    │
│  │   │   ──┼───────              ──┼─────── (smoother)             │   │    │
│  │   │     │     /                 │    ╱                          │   │    │
│  │   │     │    /                  │  ╱                            │   │    │
│  │   │     │   /                   │╱                              │   │    │
│  │   │   ──┼──/──                ─╱┼─────                          │   │    │
│  │   │     │                    ╱  │                               │   │    │
│  │   │                                                             │   │    │
│  │   └─────────────────────────┬───────────────────────────────────┘   │    │
│  │                             │                                       │    │
│  │                             ▼                                       │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │              LINEAR PROJECTION #2                           │   │    │
│  │   │                                                             │   │    │
│  │   │   ×W₂ + b₂ → [seq_len × d_model]                            │   │    │
│  │   │                                                             │   │    │
│  │   │   2048 → 512 (COMPRESS back to d_model)                     │   │    │
│  │   │                                                             │   │    │
│  │   │   W₂: [d_ff × d_model]                                      │   │    │
│  │   │                                                             │   │    │
│  │   └─────────────────────────┬───────────────────────────────────┘   │    │
│  │                             │                                       │    │
│  │                             ▼                                       │    │
│  │   Output                                                            │    │
│  │   [seq_len × d_model]                                               │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  WHY EXPAND THEN COMPRESS?                                                  │
│  ──────────────────────────                                                 │
│                                                                             │
│  • Higher-dimensional space allows more expressive transformations          │
│  • Non-linearity in high dimensions captures complex patterns               │
│  • Most FFN parameters in Transformers (2/3 of total!)                      │
│  • Applied INDEPENDENTLY to each position (position-wise FFN)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Causal (Masked) Self-Attention

For autoregressive generation, we need to prevent the model from "seeing the future":

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CAUSAL (MASKED) SELF-ATTENTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: During generation, we predict one token at a time                 │
│                                                                             │
│    "The cat sat on the [???]"                                               │
│                         ↑                                                   │
│              Predicting this token                                          │
│                                                                             │
│  We can't let the model see future tokens during training,                  │
│  or it will just copy them instead of learning to predict!                  │
│                                                                             │
│  SOLUTION: Mask out future positions in attention                           │
│  ─────────────────────────────────────────────────                          │
│                                                                             │
│  Without mask (bidirectional):       With causal mask:                      │
│                                                                             │
│       1   2   3   4   5                   1   2   3   4   5                  │
│    ┌─────────────────────┐            ┌─────────────────────┐               │
│  1 │ ✓   ✓   ✓   ✓   ✓  │          1 │ ✓   ✗   ✗   ✗   ✗  │               │
│  2 │ ✓   ✓   ✓   ✓   ✓  │          2 │ ✓   ✓   ✗   ✗   ✗  │               │
│  3 │ ✓   ✓   ✓   ✓   ✓  │          3 │ ✓   ✓   ✓   ✗   ✗  │               │
│  4 │ ✓   ✓   ✓   ✓   ✓  │          4 │ ✓   ✓   ✓   ✓   ✗  │               │
│  5 │ ✓   ✓   ✓   ✓   ✓  │          5 │ ✓   ✓   ✓   ✓   ✓  │               │
│    └─────────────────────┘            └─────────────────────┘               │
│                                                                             │
│  ✓ = can attend, ✗ = cannot attend (masked with -∞)                        │
│                                                                             │
│  Position 3 can only attend to positions 1, 2, 3                            │
│  (past and present, not future)                                             │
│                                                                             │
│  IMPLEMENTATION:                                                            │
│  ───────────────                                                            │
│                                                                             │
│  1. Create lower triangular mask:                                           │
│                                                                             │
│     mask = [[1, 0, 0, 0, 0],                                                │
│             [1, 1, 0, 0, 0],                                                │
│             [1, 1, 1, 0, 0],                                                │
│             [1, 1, 1, 1, 0],                                                │
│             [1, 1, 1, 1, 1]]                                                │
│                                                                             │
│  2. Apply to attention scores:                                              │
│                                                                             │
│     scores = scores.masked_fill(mask == 0, -inf)                            │
│                                                                             │
│  3. Softmax turns -inf into 0:                                              │
│                                                                             │
│     softmax([1.2, 0.5, -inf, -inf]) = [0.67, 0.33, 0, 0]                    │
│                                                                             │
│  Result: No gradient flows from future positions                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complete Transformer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Consists of:
    1. Multi-head self-attention (bidirectional)
    2. Feed-forward network
    Both with residual connections and layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.

    Consists of:
    1. Masked multi-head self-attention (causal)
    2. Multi-head cross-attention (to encoder output)
    3. Feed-forward network
    All with residual connections and layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-attention to encoder output
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: torch.Tensor = None,
        cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Masked self-attention
        self_attn_out = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention to encoder
        cross_attn_out = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode source sequence."""
        # Embed and scale
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode target sequence given encoder output."""
        # Embed and scale
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: Source sequence [batch, src_len]
            tgt: Target sequence [batch, tgt_len]
            src_mask: Padding mask for source
            tgt_mask: Causal mask for target

        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_proj(decoder_output)

        return logits


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal (look-ahead) mask for decoder self-attention.

    Returns:
        mask: [1, 1, seq_len, seq_len] with 1s in lower triangle, 0s elsewhere
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask (1 for real tokens, 0 for padding).

    Args:
        seq: Input sequence [batch, seq_len]
        pad_idx: Padding token index

    Returns:
        mask: [batch, 1, 1, seq_len]
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```

---

## 5.7 Types of Attention

### Cross-Attention vs Self-Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TYPES OF ATTENTION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SELF-ATTENTION:                                                            │
│  ───────────────                                                            │
│    Q, K, V all come from the SAME sequence                                  │
│                                                                             │
│    Input X ──┬──▶ W_Q ──▶ Q ─┐                                              │
│              │               │                                              │
│              ├──▶ W_K ──▶ K ─┼──▶ Attention ──▶ Output                      │
│              │               │                                              │
│              └──▶ W_V ──▶ V ─┘                                              │
│                                                                             │
│    Used in: Both encoder and decoder                                        │
│    Purpose: Let each position gather information from all other positions   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  CROSS-ATTENTION:                                                           │
│  ────────────────                                                           │
│    Q from one sequence, K and V from ANOTHER sequence                       │
│                                                                             │
│    Decoder state ──▶ W_Q ──▶ Q ─┐                                           │
│                                 │                                           │
│    Encoder output ──┬──▶ W_K ──▶ K ─┼──▶ Attention ──▶ Output               │
│                     │               │                                       │
│                     └──▶ W_V ──▶ V ─┘                                       │
│                                                                             │
│    Used in: Decoder (attending to encoder output)                           │
│    Purpose: Let decoder "look at" the source sequence                       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  EXAMPLE - Translation "The cat" → "Le chat":                               │
│                                                                             │
│    Encoder (self-attention):                                                │
│      "The" attends to "The" and "cat"                                       │
│      "cat" attends to "The" and "cat"                                       │
│                                                                             │
│    Decoder (cross-attention when generating "chat"):                        │
│      "chat" (Q) attends to encoder's "The" and "cat" (K, V)                 │
│      → High attention to "cat" because "chat" = French for "cat"            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Bidirectional vs Causal Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                BIDIRECTIONAL vs CAUSAL ATTENTION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BIDIRECTIONAL (Full) Self-Attention:                                       │
│  ─────────────────────────────────────                                      │
│                                                                             │
│    Every position can attend to every other position                        │
│                                                                             │
│    Position:  1   2   3   4   5                                             │
│               │╲ │╲ │╲ │╲ │                                                 │
│               │ ╲│ ╲│ ╲│ ╲│                                                 │
│               │  ╳  ╳  ╳  │                                                 │
│               │ ╱│ ╱│ ╱│ ╱│                                                 │
│               │╱ │╱ │╱ │╱ │                                                 │
│               1   2   3   4   5                                             │
│                                                                             │
│    Used in: BERT, encoder-only models                                       │
│    Tasks: Classification, NER, extractive QA                                │
│    Cannot be used for: Generation (would see answers)                       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  CAUSAL (Masked) Self-Attention:                                            │
│  ────────────────────────────────                                           │
│                                                                             │
│    Each position can only attend to itself and previous positions           │
│                                                                             │
│    Position:  1   2   3   4   5                                             │
│               ●                                                             │
│               │╲                                                            │
│               │ ●                                                           │
│               │ │╲                                                          │
│               │ │ ●                                                         │
│               │ │ │╲                                                        │
│               │ │ │ ●                                                       │
│               │ │ │ │╲                                                      │
│               │ │ │ │ ●                                                     │
│               1 2 3 4 5                                                     │
│                                                                             │
│    Used in: GPT, decoder-only models, language modeling                     │
│    Tasks: Text generation, completion, chat                                 │
│    Key property: Autoregressive (generate one token at a time)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.8 Transformer Variants

### The Three Main Architectures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE VARIANTS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ENCODER-ONLY (BERT-style)                                               │
│  ────────────────────────────                                               │
│                                                                             │
│    Input ──▶ [Embedding + PE] ──▶ [Encoder × N] ──▶ Representations         │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │  Bidirectional self-attention (full context)                    │      │
│    │  Pre-training: Masked Language Modeling (MLM)                   │      │
│    │                                                                 │      │
│    │  "The [MASK] sat on the mat"                                    │      │
│    │           ↓                                                     │      │
│    │       "cat" (predict masked token)                              │      │
│    │                                                                 │      │
│    │  Models: BERT, RoBERTa, ALBERT, DistilBERT, DeBERTa            │      │
│    │  Tasks: Classification, NER, QA (extractive), similarity        │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  2. DECODER-ONLY (GPT-style)                                                │
│  ───────────────────────────                                                │
│                                                                             │
│    Input ──▶ [Embedding + PE] ──▶ [Decoder × N] ──▶ Next Token              │
│                                    (causal mask)                            │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │  Causal self-attention (only look back)                         │      │
│    │  Pre-training: Next Token Prediction (autoregressive LM)        │      │
│    │                                                                 │      │
│    │  "The cat sat on the" ──▶ "mat"                                 │      │
│    │                                                                 │      │
│    │  Models: GPT-2, GPT-3, GPT-4, Claude, LLaMA, Mistral           │      │
│    │  Tasks: Text generation, completion, chat, code generation      │      │
│    │                                                                 │      │
│    │  NOTE: Most modern LLMs use this architecture                   │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  3. ENCODER-DECODER (T5-style)                                              │
│  ─────────────────────────────                                              │
│                                                                             │
│    Source ──▶ [Encoder] ──┬──▶ [Decoder] ──▶ Target                         │
│                          │                                                  │
│                      cross-attention                                        │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │  Encoder: Bidirectional self-attention                          │      │
│    │  Decoder: Causal self-attention + cross-attention               │      │
│    │  Pre-training: Span corruption / denoising                      │      │
│    │                                                                 │      │
│    │  Input: "The <X> sat on <Y> mat"                                │      │
│    │  Output: "<X> cat <Y> the"                                      │      │
│    │                                                                 │      │
│    │  Models: T5, BART, mT5, FLAN-T5                                 │      │
│    │  Tasks: Translation, summarization, generative QA               │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  COMPARISON TABLE:                                                          │
│  ┌─────────────────┬───────────────┬───────────────┬───────────────┐        │
│  │   Aspect        │ Encoder-Only  │ Decoder-Only  │ Enc-Dec       │        │
│  ├─────────────────┼───────────────┼───────────────┼───────────────┤        │
│  │ Context         │ Bidirectional │ Causal (left) │ Both          │        │
│  │ Pre-training    │ MLM           │ Next token    │ Denoising     │        │
│  │ Best for        │ Understanding │ Generation    │ Seq2Seq       │        │
│  │ Can generate?   │ No*           │ Yes           │ Yes           │        │
│  │ Popular models  │ BERT, RoBERTa │ GPT, Claude   │ T5, BART      │        │
│  └─────────────────┴───────────────┴───────────────┴───────────────┘        │
│                                                                             │
│  * Encoder-only can be adapted for generation but not its natural strength  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.9 Vision Transformer (ViT)

### Applying Transformers to Images

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VISION TRANSFORMER (ViT)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KEY INSIGHT: Treat image patches as tokens!                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  Original Image (224 × 224 × 3)                                     │    │
│  │  ┌───┬───┬───┬───┬───┬───┬───┐                                      │    │
│  │  │ P1│ P2│ P3│ P4│ P5│ P6│ P7│                                      │    │
│  │  ├───┼───┼───┼───┼───┼───┼───┤                                      │    │
│  │  │ P8│ P9│P10│P11│P12│P13│P14│   14×14 = 196 patches               │    │
│  │  ├───┼───┼───┼───┼───┼───┼───┤   Each patch: 16×16 pixels          │    │
│  │  │...│...│...│...│...│...│...│                                      │    │
│  │  ├───┼───┼───┼───┼───┼───┼───┤                                      │    │
│  │  │P..│P..│P..│P..│P..│P..│196│                                      │    │
│  │  └───┴───┴───┴───┴───┴───┴───┘                                      │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              FLATTEN & LINEAR PROJECTION                    │   │    │
│  │  │                                                             │   │    │
│  │  │  Each 16×16×3 patch = 768 values                            │   │    │
│  │  │  Linear projection to d_model (e.g., 768)                   │   │    │
│  │  │                                                             │   │    │
│  │  │  Patch → flatten → [768] → Linear → [d_model]               │   │    │
│  │  │                                                             │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              ADD [CLS] TOKEN + POSITION EMBEDDINGS          │   │    │
│  │  │                                                             │   │    │
│  │  │  [CLS] P1  P2  P3  ...  P196                                │   │    │
│  │  │    │   │   │   │         │                                  │   │    │
│  │  │    +   +   +   +         +    (add position embeddings)     │   │    │
│  │  │   pos0 pos1 pos2 pos3   pos196                              │   │    │
│  │  │                                                             │   │    │
│  │  │  Total: 197 tokens (196 patches + 1 CLS)                    │   │    │
│  │  │                                                             │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              TRANSFORMER ENCODER (× L layers)               │   │    │
│  │  │                                                             │   │    │
│  │  │  Standard self-attention + FFN                              │   │    │
│  │  │  All patches can attend to all other patches                │   │    │
│  │  │                                                             │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                           │    │
│  │         ▼                                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              CLASSIFICATION HEAD                            │   │    │
│  │  │                                                             │   │    │
│  │  │  Use [CLS] token output → Linear → num_classes             │   │    │
│  │  │                                                             │   │    │
│  │  │  [CLS] captures global image information                    │   │    │
│  │  │                                                             │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  WHY IT WORKS:                                                              │
│  ─────────────                                                              │
│                                                                             │
│  • Image patches are analogous to word tokens                               │
│  • Self-attention can capture GLOBAL relationships (unlike CNN's local)    │
│  • Position embeddings tell model where patches are spatially              │
│  • With enough data, ViT matches or exceeds CNN performance                │
│                                                                             │
│  KEY FINDING: ViT needs LOTS of data (or pre-training on ImageNet-21k)     │
│  With limited data, CNN's inductive bias (locality) helps more             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ViT Implementation

```python
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    "An Image is Worth 16x16 Words" - Dosovitskiy et al., 2020
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.d_model = d_model
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding: Conv2d with kernel=stride=patch_size
        # This is equivalent to: flatten patch → linear projection
        self.patch_embed = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable position embeddings (for CLS + all patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Linear(d_model, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images [batch, channels, height, width]

        Returns:
            logits: [batch, num_classes]
        """
        batch_size = x.size(0)

        # Step 1: Create patch embeddings
        # [B, 3, 224, 224] -> [B, d_model, 14, 14]
        x = self.patch_embed(x)

        # Flatten spatial dimensions: [B, d_model, 14, 14] -> [B, d_model, 196]
        x = x.flatten(2)

        # Transpose: [B, d_model, 196] -> [B, 196, d_model]
        x = x.transpose(1, 2)

        # Step 2: Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, d_model]

        # Step 3: Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Step 4: Pass through Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        # Step 5: Classification using [CLS] token
        cls_output = x[:, 0]  # [B, d_model]
        logits = self.head(cls_output)  # [B, num_classes]

        return logits


# Example usage
if __name__ == "__main__":
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        d_model=768,
        num_heads=12,
        num_layers=12
    )

    # Random image batch
    images = torch.randn(4, 3, 224, 224)
    logits = model(images)

    print(f"Input shape: {images.shape}")
    print(f"Output shape: {logits.shape}")  # [4, 1000]
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 5.10 KV Cache for Efficient Generation

### The Generation Problem

During autoregressive generation, naive implementation recomputes K and V for all previous tokens at each step:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE KV CACHE OPTIMIZATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NAIVE GENERATION (inefficient):                                            │
│  ────────────────────────────────                                           │
│                                                                             │
│  Step 1: Generate token 1                                                   │
│    Input: [START]                                                           │
│    Compute K₁, V₁ for [START]                                               │
│    Output: token_1                                                          │
│                                                                             │
│  Step 2: Generate token 2                                                   │
│    Input: [START, token_1]                                                  │
│    Recompute K₁, V₁ for [START]     ← WASTED WORK!                         │
│    Compute K₂, V₂ for [token_1]                                             │
│    Output: token_2                                                          │
│                                                                             │
│  Step 3: Generate token 3                                                   │
│    Input: [START, token_1, token_2]                                         │
│    Recompute K₁, V₁ for [START]     ← WASTED WORK!                         │
│    Recompute K₂, V₂ for [token_1]   ← WASTED WORK!                         │
│    Compute K₃, V₃ for [token_2]                                             │
│    Output: token_3                                                          │
│                                                                             │
│  Complexity: O(n²) for n tokens (each step processes all previous)          │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  WITH KV CACHE (efficient):                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  Step 1: Generate token 1                                                   │
│    Input: [START]                                                           │
│    Compute K₁, V₁ → CACHE: K=[K₁], V=[V₁]                                  │
│    Output: token_1                                                          │
│                                                                             │
│  Step 2: Generate token 2                                                   │
│    Input: [token_1] only (just the new token!)                              │
│    Compute K₂, V₂                                                           │
│    Append to cache: K=[K₁,K₂], V=[V₁,V₂]                                    │
│    Attention uses cached K, V                                               │
│    Output: token_2                                                          │
│                                                                             │
│  Step 3: Generate token 3                                                   │
│    Input: [token_2] only                                                    │
│    Compute K₃, V₃                                                           │
│    Append to cache: K=[K₁,K₂,K₃], V=[V₁,V₂,V₃]                              │
│    Attention uses cached K, V                                               │
│    Output: token_3                                                          │
│                                                                             │
│  Complexity: O(n) for n tokens (each step only processes 1 new token)       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  VISUALIZATION:                                                             │
│                                                                             │
│   Cache                New query                                            │
│   ┌────────────────┐   ┌────┐                                               │
│   │ K₁  K₂  K₃  K₄ │ + │ K₅ │  → Concat → [K₁ K₂ K₃ K₄ K₅]               │
│   │ V₁  V₂  V₃  V₄ │   │ V₅ │            [V₁ V₂ V₃ V₄ V₅]               │
│   └────────────────┘   └────┘                                               │
│                             │                                               │
│                             ▼                                               │
│                     Q₅ attends to all K                                     │
│                     Output weighted sum of all V                            │
│                                                                             │
│  MEMORY COST:                                                               │
│  ─────────────                                                              │
│    Per layer: 2 × seq_len × num_heads × d_k × batch_size × 2 bytes (fp16)   │
│    For GPT-3 (96 layers, 96 heads, d_k=128, seq=2048, batch=1):            │
│      = 96 × 2 × 2048 × 96 × 128 × 1 × 2 ≈ 9.6 GB                           │
│                                                                             │
│    KV cache is often the memory bottleneck for long sequences!              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### KV Cache Implementation

```python
class CachedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with KV caching for efficient generation.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple = None,
        use_cache: bool = False
    ) -> tuple:
        """
        Args:
            x: Input [batch, seq_len, d_model]
               During generation with cache: seq_len = 1 (just the new token)
            kv_cache: Tuple of (cached_k, cached_v) from previous steps
            use_cache: Whether to return updated cache

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated (k_cache, v_cache) if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V for new token(s)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Concatenate with cached K, V
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            K = torch.cat([cached_k, K], dim=2)  # Append along sequence dim
            V = torch.cat([cached_v, V], dim=2)

        # Store updated cache
        new_cache = (K, V) if use_cache else None

        # Standard attention computation
        # Q only has 1 token during generation, K/V have all previous + current
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, new_cache


def generate_with_cache(model, prompt_ids, max_new_tokens, temperature=1.0):
    """
    Efficient text generation using KV cache.
    """
    # Process prompt (prefill phase)
    # All prompt tokens processed together, cache initialized
    output, kv_cache = model(prompt_ids, kv_cache=None, use_cache=True)

    generated_ids = prompt_ids.tolist()[0]

    # Generate new tokens one at a time (decode phase)
    for _ in range(max_new_tokens):
        # Get logits for last position
        logits = output[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids.append(next_token.item())

        # Forward pass with just the new token + cached K, V
        next_input = next_token.unsqueeze(0)  # [1, 1]
        output, kv_cache = model(next_input, kv_cache=kv_cache, use_cache=True)

    return generated_ids
```

---

## 5.11 Summary

### Key Concepts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 5 KEY CONCEPTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ATTENTION MECHANISM                                                     │
│     • Soft dictionary lookup: Query finds relevant Keys, retrieves Values   │
│     • Scaled dot-product: Attention(Q,K,V) = softmax(QK^T/√d_k)V           │
│     • Solves the Seq2Seq bottleneck problem                                 │
│                                                                             │
│  2. SELF-ATTENTION                                                          │
│     • Q, K, V all from the same sequence                                    │
│     • O(1) path length between any two positions                            │
│     • Fully parallelizable (unlike RNNs)                                    │
│     • O(n²) complexity (limitation for long sequences)                      │
│                                                                             │
│  3. MULTI-HEAD ATTENTION                                                    │
│     • Multiple attention "heads" capture different relationships            │
│     • Heads specialize (syntax, coreference, semantics, etc.)               │
│     • Concatenate and project outputs                                       │
│                                                                             │
│  4. POSITIONAL ENCODING                                                     │
│     • Self-attention is permutation equivariant (no position info)          │
│     • Sinusoidal: sin/cos functions at different frequencies                │
│     • Learned: position embedding table (most common today)                 │
│                                                                             │
│  5. TRANSFORMER ARCHITECTURE                                                │
│     • Stack of attention + FFN blocks                                       │
│     • Residual connections + Layer Normalization                            │
│     • Encoder (bidirectional), Decoder (causal + cross-attention)           │
│                                                                             │
│  6. CAUSAL MASKING                                                          │
│     • Prevents attending to future positions                                │
│     • Essential for autoregressive generation                               │
│     • Lower triangular attention mask                                       │
│                                                                             │
│  7. TRANSFORMER VARIANTS                                                    │
│     • Encoder-only: BERT (understanding)                                    │
│     • Decoder-only: GPT (generation) ← most LLMs                           │
│     • Encoder-Decoder: T5 (seq2seq)                                         │
│                                                                             │
│  8. VISION TRANSFORMER (ViT)                                                │
│     • Image patches as tokens                                               │
│     • Transformers for vision tasks                                         │
│     • Needs lots of data or pretraining                                     │
│                                                                             │
│  9. KV CACHE                                                                │
│     • Cache K, V from previous tokens during generation                     │
│     • Reduces complexity from O(n²) to O(n) per token                       │
│     • Memory vs compute tradeoff                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Glossary Terms Covered

- **Attention Mechanism**: Soft lookup allowing models to focus on relevant parts
- **Self-Attention**: Attention where Q, K, V come from the same sequence
- **Cross-Attention**: Q from one sequence, K/V from another
- **Query, Key, Value (Q, K, V)**: Components of attention computation
- **Scaled Dot-Product Attention**: QK^T/√d_k with softmax
- **Multi-Head Attention**: Parallel attention heads for different relationships
- **Positional Encoding**: Injecting position information into embeddings
- **Transformer**: Architecture built on self-attention
- **Encoder-Decoder**: Architecture with separate encode/decode stages
- **Causal Masking**: Preventing attention to future positions
- **Layer Normalization**: Normalizing across features within each sample
- **Residual Connection**: Skip connection adding input to output
- **Feed-Forward Network (FFN)**: Two-layer MLP in Transformer blocks
- **KV Cache**: Caching keys and values for efficient generation
- **Vision Transformer (ViT)**: Transformer architecture for images

### What's Next

Module 6 builds on Transformers to cover **Large Language Models**: tokenization, pre-training, scaling laws, fine-tuning techniques (LoRA, QLoRA), RLHF, prompting strategies, and the path from base models to helpful assistants.

---

## 5.12 Exercises

1. **Attention by Hand**: Given Q=[1,0,1], K=[[1,0,0],[0,1,0],[1,1,0]], V=[[1],[2],[3]], compute the attention output (without scaling).

2. **Complexity Analysis**: For a Transformer with d_model=512, 8 heads, FFN expansion 4×, sequence length 1000:
   - How many parameters in one encoder layer?
   - What's the memory for KV cache for 12 layers?

3. **Causal Mask**: Write code to create a causal mask and show how it affects attention scores.

4. **Position Encoding**: Plot the sinusoidal positional encoding for the first 100 positions and 64 dimensions. What patterns do you see?

5. **Implementation**: Implement a complete Transformer encoder from scratch and verify it works on a simple sequence.

---

## 5.13 References

### Primary Papers

- **Vaswani et al. (2017)** - "Attention Is All You Need"
  - Original Transformer paper
  - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- **Bahdanau et al. (2015)** - "Neural Machine Translation by Jointly Learning to Align and Translate"
  - Introduced attention for NMT
  - [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)

- **Dosovitskiy et al. (2020)** - "An Image is Worth 16x16 Words"
  - Vision Transformer (ViT)
  - [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

### Model Papers

- **Devlin et al. (2019)** - "BERT: Pre-training of Deep Bidirectional Transformers"
  - Encoder-only architecture
  - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

- **Radford et al. (2019)** - "Language Models are Unsupervised Multitask Learners"
  - GPT-2, decoder-only architecture
  - [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- **Raffel et al. (2020)** - "Exploring the Limits of Transfer Learning with T5"
  - T5, encoder-decoder architecture
  - [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

### Course Lectures

- **CS231n** - Attention and Transformers
  - [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)

- **CS224N** - Transformers and Pretraining
  - Lecture 9: Self-Attention and Transformers
  - [https://web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/)

### Interactive Resources

- **The Illustrated Transformer** - Jay Alammar
  - Excellent visual explanations
  - [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

- **The Annotated Transformer** - Harvard NLP
  - Line-by-line implementation with explanations
  - [https://nlp.seas.harvard.edu/annotated-transformer/](https://nlp.seas.harvard.edu/annotated-transformer/)

- **Attention? Attention!** - Lilian Weng
  - Comprehensive attention survey
  - [https://lilianweng.github.io/posts/2018-06-24-attention/](https://lilianweng.github.io/posts/2018-06-24-attention/)

### Books

- **"Speech and Language Processing"** - Jurafsky & Martin
  - Chapter on Transformers and attention
  - [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)

- **"Deep Learning"** - Goodfellow, Bengio, Courville
  - Foundations for understanding attention
  - [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

### Implementation References

- **PyTorch Transformer Documentation**
  - [https://pytorch.org/docs/stable/nn.html#transformer-layers](https://pytorch.org/docs/stable/nn.html#transformer-layers)

- **Hugging Face Transformers**
  - Library for pre-trained models
  - [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
