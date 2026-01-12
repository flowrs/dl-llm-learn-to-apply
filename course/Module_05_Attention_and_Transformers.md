# Module 5: Attention & Transformers

## Learning Objectives

By the end of this module, you will understand:
- The attention mechanism and why it revolutionized deep learning
- Self-attention and its computational properties
- Multi-head attention for richer representations
- The complete Transformer architecture
- Positional encoding for sequence order
- Vision Transformers (ViT)

---

## 5.1 Motivation: The Bottleneck Problem

### The Seq2Seq Limitation

In basic Seq2Seq, all source information flows through one context vector:

```
"The cat sat on the mat" → [Encoder] → c → [Decoder] → "Le chat..."
                                       ↑
                           Single vector must encode everything!
```

For long sequences, this bottleneck loses information.

### The Key Insight

Instead of one fixed context, let the decoder **look at all encoder states** and focus on relevant parts:

```
"The cat sat"
  ↓   ↓   ↓
 h₁  h₂  h₃   (encoder states)
  ↘  ↓  ↙
    [Attention] ← "Which encoder states matter for this decoder step?"
      ↓
  Weighted sum = context for current step
```

---

## 5.2 The Attention Mechanism

### Query, Key, Value

Attention can be thought of as a **soft dictionary lookup**:

```
Query (q):  "What am I looking for?"
Keys (K):   "What does each item contain?"
Values (V): "What information to retrieve?"

Regular dictionary: key → exact match → one value
Attention: query → similarity to all keys → weighted sum of values
```

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

**Step by step**:

1. **Compute scores**: How well does each key match the query?
   ```
   scores = Q · Kᵀ    # [seq_len × seq_len]
   ```

2. **Scale**: Prevent dot products from growing too large
   ```
   scores = scores / √d_k    # d_k = key dimension
   ```

3. **Softmax**: Convert to probabilities (attention weights)
   ```
   weights = softmax(scores)  # Each row sums to 1
   ```

4. **Weighted sum**: Retrieve values according to attention
   ```
   output = weights · V
   ```

### Visual Representation

```
Query: "sat"
                    ┌─────────────────────────┐
                    │  "The"  "cat"  "sat"    │
         ┌─────────▶│   0.1    0.7   0.2     │◀── Attention weights
         │          └─────────────────────────┘
         │                     │
         │                     ▼
         │            Weighted sum of values
         │                     │
         │                     ▼
     Query "sat" ───────────► Output context
```

### Why Scale by √d_k?

For large d_k, dot products have large variance:
```
If q_i, k_i ~ N(0, 1), then q·k ~ N(0, d_k)

Large values → softmax → very peaked distribution
→ gradients become very small
```

Scaling by √d_k keeps variance at 1.

### Python Implementation

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, seq_len_q, d_k]
    K: [batch, seq_len_k, d_k]
    V: [batch, seq_len_k, d_v]
    """
    d_k = K.shape[-1]

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

---

## 5.3 Self-Attention

### What is Self-Attention?

In self-attention, Q, K, V all come from the **same sequence**:

```
Input: x₁, x₂, x₃
           ↓
       [Linear] → Q₁, Q₂, Q₃
       [Linear] → K₁, K₂, K₃
       [Linear] → V₁, V₂, V₃
           ↓
      Self-Attention
           ↓
       y₁, y₂, y₃
```

Each position can attend to all other positions (including itself).

### Why Self-Attention?

1. **Constant path length**: Any two positions are 1 step apart
   - RNN: O(n) steps for distant positions
   - Self-attention: O(1) steps

2. **Parallelizable**: All positions computed simultaneously
   - RNN: Sequential, must wait for previous step

3. **Interpretable**: Attention weights show what the model "looks at"

### Complexity Comparison

| Operation | Sequential Ops | Maximum Path | Computation |
|-----------|----------------|--------------|-------------|
| RNN | O(n) | O(n) | O(n·d²) |
| Self-Attention | O(1) | O(1) | O(n²·d) |
| Convolution | O(1) | O(log_k(n)) | O(k·n·d²) |

Self-attention is O(n²) in sequence length—a key limitation for long sequences.

### Permutation Equivariance

Self-attention is **permutation equivariant**: reordering inputs reorders outputs the same way.

```
Input: [A, B, C] → Output: [A', B', C']
Input: [B, A, C] → Output: [B', A', C']
```

This means **order information is lost** without positional encoding.

---

## 5.4 Multi-Head Attention

### Why Multiple Heads?

Single attention captures one type of relationship. Multiple heads capture different aspects:

```
Head 1: Might focus on syntactic relationships
Head 2: Might focus on semantic similarity
Head 3: Might focus on positional proximity
...
```

### Implementation

```
Multi-Head Attention:
1. Project Q, K, V to h different subspaces
2. Apply attention in parallel for each head
3. Concatenate results
4. Project back to model dimension

MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

```
Input → [Split into h heads]
            ↓
        ┌───┬───┬───┐
        │ H₁│ H₂│ H₃│  (parallel attention)
        └───┴───┴───┘
            ↓
        [Concatenate]
            ↓
        [Linear projection]
            ↓
         Output
```

### Code Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q)  # [batch, seq, d_model]
        K = self.W_k(K)
        V = self.W_v(V)

        # Reshape to [batch, num_heads, seq, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back: [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Final projection
        return self.W_o(attn_output)
```

---

## 5.5 Positional Encoding

### The Problem

Self-attention is permutation equivariant—it doesn't know position!

```
"Dog bites man" vs "Man bites dog"
→ Same bag of words, different meaning!
```

### Solution: Add Position Information

```
Input embedding + Positional encoding → Model input
```

### Sinusoidal Positional Encoding

The original Transformer uses sine and cosine functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = position in sequence (0, 1, 2, ...)
- `i` = dimension index
- `d_model` = model dimension

**Properties**:
- Unique encoding for each position
- Can extrapolate to longer sequences
- Relative positions have consistent patterns

### Learned Positional Embeddings

Alternative: Learn position embeddings like word embeddings:

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embedding(positions)
```

Used in BERT, GPT, and most modern models.

---

## 5.6 The Transformer Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                        ENCODER                          │
│  ┌────────────────────────────────────────────────────┐│
│  │  Input Embedding + Positional Encoding             ││
│  └────────────────────────────────────────────────────┘│
│                           ↓                             │
│  ┌────────────────────────────────────────────────────┐│
│  │  Multi-Head Self-Attention                         ││
│  │           ↓                                        ││
│  │  Add & Norm (Residual + LayerNorm)                 ││
│  │           ↓                                        ││
│  │  Feed-Forward Network                              ││
│  │           ↓                                        ││
│  │  Add & Norm                                        ││
│  └────────────────────────────────────────────────────┘│
│              × N layers                                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                        DECODER                          │
│  ┌────────────────────────────────────────────────────┐│
│  │  Output Embedding + Positional Encoding            ││
│  └────────────────────────────────────────────────────┘│
│                           ↓                             │
│  ┌────────────────────────────────────────────────────┐│
│  │  Masked Multi-Head Self-Attention                  ││
│  │           ↓                                        ││
│  │  Add & Norm                                        ││
│  │           ↓                                        ││
│  │  Multi-Head Cross-Attention (Q from decoder,       ││
│  │                             K, V from encoder)     ││
│  │           ↓                                        ││
│  │  Add & Norm                                        ││
│  │           ↓                                        ││
│  │  Feed-Forward Network                              ││
│  │           ↓                                        ││
│  │  Add & Norm                                        ││
│  └────────────────────────────────────────────────────┘│
│              × N layers                                 │
└─────────────────────────────────────────────────────────┘
                           ↓
                  Linear + Softmax → Output probabilities
```

### Key Components

**1. Residual Connections**
```
output = LayerNorm(x + Sublayer(x))
```
Helps gradient flow, enables deep networks.

**2. Layer Normalization**
```python
def layer_norm(x, gamma, beta):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return gamma * (x - mean) / (std + eps) + beta
```
Normalizes across features (not batch).

**3. Feed-Forward Network**
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```
Usually d_ff = 4 × d_model (e.g., 2048 for d_model=512).

**4. Masked Self-Attention**
Prevents decoder from seeing future tokens:

```
Attention mask:
     Position 1  2  3  4
        ┌─────────────────┐
      1 │  1    0  0  0   │   (can only see pos 1)
      2 │  1    1  0  0   │   (can see pos 1-2)
      3 │  1    1  1  0   │   (can see pos 1-3)
      4 │  1    1  1  1   │   (can see pos 1-4)
        └─────────────────┘
```

### Transformer Code

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

---

## 5.7 Types of Attention

### Cross-Attention

Q from one sequence, K and V from another:

```
Decoder cross-attention:
  Q ← decoder hidden states
  K, V ← encoder outputs

"Translate to French" → Cross-attention lets decoder
                        look at English source
```

### Causal (Masked) Self-Attention

Prevents attending to future positions:

```python
def create_causal_mask(seq_len):
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 = can attend, 0 = cannot
```

Used in: GPT, language modeling, autoregressive generation.

### Bidirectional Self-Attention

No masking—each position attends to all positions:
- Used in: BERT, encoder-only models
- Good for: Understanding, classification
- Cannot be used for: Generation (would see future)

---

## 5.8 Transformer Variants

### Encoder-Only (BERT-style)

```
Input → [Encoder] → Representations

Used for:
- Classification
- Named entity recognition
- Question answering (extractive)
```

### Decoder-Only (GPT-style)

```
Input → [Decoder with causal mask] → Next token

Used for:
- Text generation
- Language modeling
- Most modern LLMs
```

### Encoder-Decoder (T5-style)

```
Input → [Encoder] → [Decoder] → Output

Used for:
- Translation
- Summarization
- Any seq2seq task
```

---

## 5.9 Vision Transformer (ViT)

### Applying Transformers to Images

Images aren't sequences—how to use Transformers?

**Solution**: Split image into patches and treat as tokens.

```
Image (224×224)
       ↓
   [Split into 16×16 patches]
       ↓
   14×14 = 196 patches
       ↓
   [Flatten each patch to vector]
       ↓
   [Linear projection to d_model]
       ↓
   [Add position embeddings]
       ↓
   [Transformer Encoder]
       ↓
   [Classification head]
```

### ViT Architecture

```python
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model, num_heads, num_layers):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)

        # [CLS] token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads),
            num_layers
        )

        # Classification head
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x)  # [B, d_model, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classify using [CLS] token
        return self.head(x[:, 0])
```

---

## 5.10 KV Cache for Efficient Generation

### The Problem

During autoregressive generation, we recompute K and V for all previous tokens:

```
Step 1: K, V for [token1]
Step 2: K, V for [token1, token2]  ← recomputes token1!
Step 3: K, V for [token1, token2, token3]  ← recomputes token1, token2!
```

### Solution: Cache K and V

```python
class CachedAttention:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, q, k, v, use_cache=False):
        if use_cache and self.k_cache is not None:
            # Append new K, V to cache
            k = torch.cat([self.k_cache, k], dim=1)
            v = torch.cat([self.v_cache, v], dim=1)

        if use_cache:
            self.k_cache = k
            self.v_cache = v

        # Only query with new token(s)
        return attention(q, k, v)
```

Reduces generation complexity from O(n²) to O(n) per token.

---

## 5.11 Summary

### Key Concepts

1. **Attention** allows models to focus on relevant parts of input
2. **Self-attention** lets each position attend to all others in O(1) path length
3. **Multi-head attention** captures different types of relationships
4. **Positional encoding** provides sequence order information
5. **Transformers** stack attention with FFN, using residuals and normalization
6. **Causal masking** enables autoregressive generation
7. **KV cache** makes generation efficient

### Glossary Terms Covered

- Attention Mechanism
- Self-Attention
- Cross-Attention
- Query, Key, Value (Q, K, V)
- Multi-Head Attention
- Transformer
- Positional Encoding
- Encoder-Decoder Architecture
- Causal Masking
- KV Cache

### What's Next

Module 6 builds on Transformers to cover **Large Language Models**: pre-training, fine-tuning, RLHF, and prompting.

---

## Exercises

1. **Attention computation**: Given Q=[1,0], K=[[1,0],[0,1],[1,1]], V=[[1],[2],[3]], compute attention output.

2. **Complexity**: For sequence length n=1000, d_model=512, how many operations for self-attention?

3. **Masking**: Why can't we use bidirectional attention for text generation?

4. **Code**: Implement a single Transformer encoder layer from scratch.

---

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- CS231n: Attention
- CS224N: Transformers
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT)
