# Module 4: Sequence Models

## Learning Objectives

By the end of this module, you will understand:
- Language modeling and sequence processing
- Recurrent Neural Networks (RNNs) and their limitations
- LSTM and GRU architectures
- Sequence-to-sequence models
- The vanishing gradient problem and solutions

---

## 4.1 Sequential Data and Language Modeling

### Types of Sequential Data

- **Text**: "The cat sat on the mat"
- **Time series**: Stock prices, sensor readings
- **Audio**: Speech waveforms
- **Video**: Sequences of frames

### Language Modeling

A **language model** predicts the probability of the next word:

```
P(w_t | w_1, w_2, ..., w_{t-1})

"The cat sat on the ___"
  ↓
P(mat) = 0.15
P(floor) = 0.12
P(dog) = 0.01
```

**Why it matters**:
- Foundation for text generation
- Basis for machine translation
- Key pre-training objective for LLMs

### Why Not Feedforward Networks?

Problems with fixed-size windows:

```
Input: "The cat that I saw yesterday in the park sat on the ___"
                                                        ↑
Window of 5 words: "park sat on the" → misses "cat"!
```

We need models that handle **variable-length context**.

---

## 4.2 Recurrent Neural Networks (RNNs)

### Core Idea

Process sequences one element at a time, maintaining a **hidden state**:

```
     x_0         x_1         x_2         x_3
      │           │           │           │
      ↓           ↓           ↓           ↓
   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
h_0│ RNN │→h_1│ RNN │→h_2│ RNN │→h_3│ RNN │→ h_4
   └─────┘    └─────┘    └─────┘    └─────┘
      │           │           │           │
      ↓           ↓           ↓           ↓
     y_0         y_1         y_2         y_3
```

**Same weights** applied at every time step (parameter sharing).

### RNN Equations

At each time step t:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

Where:
- `h_t`: Hidden state at time t (memory)
- `x_t`: Input at time t
- `y_t`: Output at time t
- `W_hh`: Hidden-to-hidden weights (recurrent)
- `W_xh`: Input-to-hidden weights
- `W_hy`: Hidden-to-output weights

### Python Implementation

```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        """
        inputs: list of input vectors
        h_prev: initial hidden state
        """
        h = h_prev
        outputs = []

        for x in inputs:
            # Update hidden state
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            # Compute output
            y = self.Why @ h + self.by
            outputs.append(y)

        return outputs, h
```

### Types of RNN Architectures

**One-to-Many**: Image captioning
```
Image → [RNN] → "A" → [RNN] → "cat" → [RNN] → "sitting"
```

**Many-to-One**: Sentiment classification
```
"I" → [RNN] → "love" → [RNN] → "this" → [RNN] → "movie" → [RNN] → Positive
```

**Many-to-Many (same length)**: Part-of-speech tagging
```
"The"  → [RNN] → DET
"cat"  → [RNN] → NOUN
"sat"  → [RNN] → VERB
```

**Many-to-Many (different length)**: Translation (Seq2Seq)
```
"The cat" → [Encoder] → [Context] → [Decoder] → "Le chat"
```

---

## 4.3 The Vanishing Gradient Problem

### Why RNNs Struggle with Long Sequences

During backpropagation through time (BPTT):

```
∂L/∂W = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_2/∂h_1 × ∂h_1/∂W
                   └────────────────────────────────┘
                        Product of many terms
```

Each term `∂h_t/∂h_{t-1}` involves `tanh'` and `W_hh`:

```
∂h_t/∂h_{t-1} = tanh'(...) × W_hh
```

**Problem 1: Vanishing**
- `|tanh'(x)| ≤ 1`, often < 1
- If `||W_hh|| < 1`, products → 0

**Problem 2: Exploding**
- If `||W_hh|| > 1`, products → ∞

### Consequences

```
Sequence: "The cat that I saw yesterday sat on the ___"
           ↑                              ↑
           word 1                         word 9

Gradient from word 9 barely reaches word 1!
→ Model can't learn long-range dependencies
```

### Solutions

1. **Gradient clipping**: Cap gradient magnitude
2. **Better architectures**: LSTM, GRU
3. **Skip connections**: Direct paths for gradients
4. **Transformers**: No sequential dependencies (Module 5)

---

## 4.4 Long Short-Term Memory (LSTM)

### Key Innovation: Cell State

LSTMs add a **cell state** `c_t` that flows through time with minimal modification:

```
      c_{t-1} ────────────────────────→ c_t
                    (information highway)
       │                                  ↑
       │    ┌─────────────────────────────┘
       │    │
       │    │  ┌─────┐  ┌─────┐  ┌─────┐
       └────┼──│  f  │──│  i  │──│  o  │
            │  └─────┘  └─────┘  └─────┘
            │  forget   input    output
            │   gate    gate      gate
            │
       h_{t-1} ─┘
```

### LSTM Equations

**Step 1: Forget Gate** - What to remove from cell state
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Step 2: Input Gate** - What new information to store
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # Candidate values
```

**Step 3: Update Cell State**
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
      └─────┬─────┘   └────┬────┘
       what to keep    what to add
```

**Step 4: Output Gate** - What to output
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

Where `⊙` is element-wise multiplication.

### Why LSTMs Work

The cell state gradient:
```
∂c_t/∂c_{t-1} = f_t
```

If `f_t ≈ 1` (forget gate open):
- Gradient flows directly through
- No vanishing!

The gates learn when to:
- **Remember** (f close to 1): Keep information
- **Forget** (f close to 0): Clear information
- **Write** (i close to 1): Store new info
- **Read** (o close to 1): Output cell content

### Python LSTM

```python
def lstm_step(x_t, h_prev, c_prev, params):
    """One step of LSTM"""
    # Concatenate input and previous hidden
    combined = np.concatenate([h_prev, x_t])

    # Gates (all use sigmoid)
    f_t = sigmoid(params['Wf'] @ combined + params['bf'])  # Forget
    i_t = sigmoid(params['Wi'] @ combined + params['bi'])  # Input
    o_t = sigmoid(params['Wo'] @ combined + params['bo'])  # Output

    # Candidate cell state
    g_t = np.tanh(params['Wg'] @ combined + params['bg'])

    # Update cell state
    c_t = f_t * c_prev + i_t * g_t

    # Update hidden state
    h_t = o_t * np.tanh(c_t)

    return h_t, c_t
```

---

## 4.5 Gated Recurrent Unit (GRU)

### Simplified Gating

GRU combines forget and input gates into one **update gate**:

```
z_t = σ(W_z · [h_{t-1}, x_t])         # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])         # Reset gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])  # Candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # New hidden state
```

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More (separate cell state) | Fewer |
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Performance | Slightly better on some tasks | Similar, faster training |
| Usage | More common in practice | Good alternative |

---

## 4.6 Sequence-to-Sequence (Seq2Seq)

### Architecture

**Encoder-Decoder** structure for variable-length input/output:

```
Encoder:
"The cat sat" → [LSTM] → [LSTM] → [LSTM] → context vector c
                  ↑         ↑         ↑
                "The"     "cat"     "sat"

Decoder:
            c → [LSTM] → "Le"
                  ↓
                [LSTM] → "chat"
                  ↓
                [LSTM] → "assis"
                  ↓
                [LSTM] → <EOS>
```

### The Bottleneck Problem

All source information must fit in context vector `c`:
- Works for short sequences
- Struggles with long sequences
- **Solution**: Attention (Module 5)

### Teacher Forcing

During training, feed **ground truth** tokens to decoder:

```
Training:
c → [LSTM] → "Le"   (given ground truth "<START>")
      ↓
    [LSTM] → "chat" (given ground truth "Le")
      ↓
    [LSTM] → "assis" (given ground truth "chat")

Inference:
c → [LSTM] → "Le"   (no ground truth available)
      ↓
    [LSTM] → "chat" (use predicted "Le")
```

**Exposure bias**: Model never sees its own mistakes during training.

---

## 4.7 Bidirectional RNNs

### Motivation

Forward RNN only sees past context:

```
"The movie was ___ even though the acting was great"

Forward RNN at "___" has seen: "The movie was"
But "even though" suggests contrast!
```

### Solution: Two RNNs

```
Forward:   x_1 → [→] → x_2 → [→] → x_3 → [→] → x_4
                  ↓          ↓          ↓          ↓
Backward:  x_1 ← [←] ← x_2 ← [←] ← x_3 ← [←] ← x_4

Output: h_t = [h_forward_t ; h_backward_t]  (concatenate)
```

Each position has context from **both directions**.

**Note**: Cannot be used for generation (need to see future).

---

## 4.8 Multi-layer RNNs

### Stacking for Depth

```
Layer 3:  [RNN] → [RNN] → [RNN] → [RNN]
            ↑        ↑        ↑        ↑
Layer 2:  [RNN] → [RNN] → [RNN] → [RNN]
            ↑        ↑        ↑        ↑
Layer 1:  [RNN] → [RNN] → [RNN] → [RNN]
            ↑        ↑        ↑        ↑
           x_1      x_2      x_3      x_4
```

- Deeper networks learn more abstract representations
- Typical: 2-4 layers
- More layers → more dropout needed

---

## 4.9 Practical Considerations

### Word Embeddings

Don't use one-hot vectors! Use learned embeddings:

```python
# One-hot: sparse, no similarity information
"cat" → [0, 0, 1, 0, ..., 0]  # V-dimensional

# Embedding: dense, captures meaning
"cat" → [0.2, -0.5, 0.1, ...]  # d-dimensional (e.g., 300)

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
x = embedding_layer(token_ids)  # [batch, seq_len, embed_dim]
```

### Handling Variable Length

**Padding**: Add special tokens to make sequences same length

```
"The cat"     → [The, cat, <PAD>, <PAD>]
"A big dog"   → [A, big, dog, <PAD>]
```

**Packing**: PyTorch's `pack_padded_sequence` for efficiency

### Common PyTorch Pattern

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                           num_layers=2, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x: [batch, seq_len] token IDs
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        output, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state from both directions
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden_cat)
```

---

## 4.10 Summary

### Key Concepts

1. **RNNs** process sequences by maintaining hidden state across time steps
2. **Vanishing gradients** prevent learning long-range dependencies
3. **LSTMs** use gating and cell state to solve vanishing gradients
4. **GRUs** are a simplified alternative to LSTMs
5. **Seq2Seq** enables variable-length input/output
6. **Bidirectional RNNs** capture context from both directions

### Glossary Terms Covered

- Recurrent Neural Network (RNN)
- Hidden State
- Sequence-to-Sequence (Seq2Seq)
- Encoder-Decoder
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Language Model
- Embedding
- Teacher Forcing
- Vanishing Gradient (revisited)

### Limitations of RNNs

1. **Sequential processing** → slow, hard to parallelize
2. **Long-range dependencies** → still difficult despite LSTM
3. **Fixed-size bottleneck** in Seq2Seq

These limitations motivate **attention mechanisms** (Module 5).

---

## Exercises

1. **RNN states**: An RNN has hidden size 128. Input sequence length is 50. How many hidden states are computed?

2. **LSTM gates**: The forget gate outputs 0.9 for all positions. What happens to information in the cell state?

3. **Parameter count**: LSTM with input size 256, hidden size 512. How many parameters in the LSTM cell? (Hint: 4 gate computations)

4. **Code**: Implement a character-level language model using LSTM.

---

## References

- CS231n: Recurrent Neural Networks
- CS224N: Language Models and RNNs
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (GRU)
