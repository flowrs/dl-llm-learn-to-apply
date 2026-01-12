# Module 4: Sequence Models

## Learning Objectives

By the end of this module, you will understand:
- Language modeling and sequence processing fundamentals
- Recurrent Neural Networks (RNNs) architecture and mechanics
- The vanishing gradient problem and why it matters
- LSTM and GRU architectures that solve gradient issues
- Sequence-to-sequence models for translation
- Bidirectional and stacked RNN architectures
- Practical implementation patterns

---

## 4.1 Sequential Data and Language Modeling

### Types of Sequential Data

Sequential data is everywhere in the real world:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         TYPES OF SEQUENTIAL DATA                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  TEXT (Variable length sequences of tokens)                                │
│  ─────────────────────────────────────────                                │
│  "The cat sat on the mat" → [The, cat, sat, on, the, mat]                 │
│                                                                            │
│  Each token depends on previous tokens for meaning:                        │
│  "bank" in "river bank" vs "bank account"                                  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  TIME SERIES (Regular intervals)                                           │
│  ─────────────────────────────────                                        │
│  Stock prices:    $150 → $152 → $148 → $155 → ?                           │
│  Temperature:     72°F → 75°F → 78°F → 80°F → ?                           │
│  Heart rate:      72 → 75 → 71 → 73 → ?                                   │
│                                                                            │
│  Values at time t depend on values at times t-1, t-2, ...                  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  AUDIO (Waveform samples)                                                  │
│  ─────────────────────────                                                │
│  Speech waveform (16kHz = 16,000 samples per second):                      │
│                                                                            │
│      amplitude                                                             │
│          ↑    ∿∿∿∿                                                        │
│          │  ∿      ∿∿∿                                                    │
│          │∿           ∿∿∿                                                 │
│          └─────────────────→ time                                         │
│                                                                            │
│  Must process long sequences (1 second = 16,000 steps!)                    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  VIDEO (Sequences of frames)                                               │
│  ──────────────────────────                                               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                 │
│  │Frame│→│Frame│→│Frame│→│Frame│→│Frame│                                 │
│  │  1  │ │  2  │ │  3  │ │  4  │ │  5  │                                 │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                                 │
│                                                                            │
│  Action recognition: What is happening in this video?                      │
│  Each frame is a 2D image; sequence adds temporal dimension                │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DNA/PROTEIN SEQUENCES (Biological sequences)                              │
│  ─────────────────────────────────────────────                            │
│  DNA:     A-T-G-C-C-T-A-G-...                                             │
│  Protein: Met-Ala-Gly-Ser-...                                              │
│                                                                            │
│  Predict protein structure, function, or binding sites                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Language Modeling

A **language model** learns to predict the probability of the next word given previous context:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          LANGUAGE MODELING                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Mathematical formulation:                                                 │
│  ─────────────────────────                                                │
│  P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁...wₙ₋₁)│
│                                                                            │
│  Or using chain rule:                                                      │
│              n                                                             │
│  P(w₁:ₙ) = ∏ P(wᵢ | w₁:ᵢ₋₁)                                               │
│             i=1                                                            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Example: "The cat sat on the ___"                                        │
│  ──────────────────────────────────                                       │
│                                                                            │
│  Context: [The, cat, sat, on, the]                                         │
│           ↓                                                                │
│  ┌─────────────────────────┐                                              │
│  │    Language Model       │                                              │
│  └───────────┬─────────────┘                                              │
│              ↓                                                             │
│  ┌─────────────────────────────────────┐                                  │
│  │ Probability Distribution over V:    │                                  │
│  │                                     │                                  │
│  │ P(mat)   = 0.15  ████████████████   │                                  │
│  │ P(floor) = 0.12  █████████████      │                                  │
│  │ P(table) = 0.08  ████████           │                                  │
│  │ P(chair) = 0.05  █████              │                                  │
│  │ P(dog)   = 0.01  █                  │                                  │
│  │ ...                                 │                                  │
│  │ Sum over all V = 1.0                │                                  │
│  └─────────────────────────────────────┘                                  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Why Language Modeling Matters:                                            │
│  ───────────────────────────────                                          │
│                                                                            │
│  1. TEXT GENERATION                                                        │
│     Sample from P(w|context) repeatedly to generate text                   │
│     "The cat sat on the" → "mat" → "." → <END>                            │
│                                                                            │
│  2. SCORING/RANKING                                                        │
│     P("The cat sat on the mat") > P("Mat the cat the on sat")             │
│     Higher probability = more fluent/natural                               │
│                                                                            │
│  3. PRE-TRAINING FOR NLP                                                   │
│     GPT, BERT, etc. use language modeling as pre-training objective       │
│     Learn general language understanding, then fine-tune                   │
│                                                                            │
│  4. MACHINE TRANSLATION (indirectly)                                       │
│     Decoder in translation models is a conditional language model          │
│     P(target | source)                                                     │
│                                                                            │
│  5. SPEECH RECOGNITION                                                     │
│     Combine acoustic model with language model for better transcription    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Perplexity: Evaluating Language Models

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          PERPLEXITY (PPL)                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Definition:                                                               │
│  ───────────                                                              │
│                    1   n                                                   │
│  PPL = exp( - ─── Σ  log P(wᵢ | w₁:ᵢ₋₁) )                                │
│                n  i=1                                                      │
│                                                                            │
│  Equivalently:                                                             │
│  PPL = exp(cross-entropy loss)                                             │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Intuition: "Average branching factor"                                     │
│  ─────────────────────────────────────                                    │
│                                                                            │
│  PPL = 100 means: On average, the model is as uncertain about the         │
│                   next word as if choosing from 100 equally likely words   │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │ PPL = 1     Perfect prediction (knows exactly what comes next)│         │
│  │ PPL = 10    Good model (10-way uncertainty on average)        │         │
│  │ PPL = 100   Decent model                                      │         │
│  │ PPL = 1000  Poor model                                        │         │
│  │ PPL = |V|   Random guessing (worst case)                      │         │
│  └──────────────────────────────────────────────────────────────┘         │
│                                                                            │
│  Lower perplexity = better language model                                  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Example Calculation:                                                      │
│  ────────────────────                                                     │
│  Sentence: "The cat sat" (3 tokens)                                        │
│                                                                            │
│  P(The|<start>) = 0.1                                                      │
│  P(cat|The) = 0.05                                                         │
│  P(sat|The cat) = 0.02                                                     │
│                                                                            │
│  Log probabilities: log(0.1) + log(0.05) + log(0.02) = -2.3 - 3.0 - 3.9   │
│                   = -9.2                                                   │
│                                                                            │
│  Average: -9.2 / 3 = -3.07                                                 │
│  PPL = exp(3.07) ≈ 21.5                                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why Not Feedforward Networks for Sequences?

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   WHY FEEDFORWARD NETWORKS FAIL                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Problem 1: FIXED WINDOW SIZE                                              │
│  ────────────────────────────────                                         │
│                                                                            │
│  Feedforward approach: Use last k words as input (n-gram model)            │
│                                                                            │
│  "The cat that I saw yesterday in the park sat on the ___"                │
│                                                                            │
│  Window of 5:  [park, sat, on, the, ___]                                   │
│                                                                            │
│  Problem: "cat" is 10 words back! Model can't see it.                      │
│           "The ___ sat" should predict something that "sat"                │
│           But "park" is just a prepositional phrase...                     │
│                                                                            │
│  Making window bigger:                                                     │
│  - Window of 20? Still might not be enough                                 │
│  - Window of 100? Huge number of parameters                                │
│  - And still can't handle variable-length!                                 │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Problem 2: NO PARAMETER SHARING                                           │
│  ─────────────────────────────────                                        │
│                                                                            │
│  Feedforward:                                                              │
│  ┌─────┬─────┬─────┬─────┬─────┐                                          │
│  │pos 1│pos 2│pos 3│pos 4│pos 5│                                          │
│  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                                          │
│     │     │     │     │                                                    │
│    W₁    W₂    W₃    W₄    W₅   ← Different weights for each position!    │
│     │     │     │     │     │                                              │
│     └─────┴─────┴─────┴─────┘                                              │
│               │                                                            │
│           [Hidden]                                                         │
│                                                                            │
│  What model learns at position 1 doesn't transfer to position 2!           │
│  "cat" as first word vs "cat" as fifth word = different parameters         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Problem 3: VARIABLE LENGTH INPUT                                          │
│  ─────────────────────────────────                                        │
│                                                                            │
│  Sentences have different lengths:                                         │
│  - "Hi"                    (2 characters)                                  │
│  - "The quick brown fox"   (19 characters)                                 │
│  - [An entire paragraph]   (1000+ characters)                              │
│                                                                            │
│  Feedforward networks need fixed-size input!                               │
│                                                                            │
│  Padding solutions are wasteful and don't really solve the problem.        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE SOLUTION: RECURRENT NEURAL NETWORKS                                   │
│  ───────────────────────────────────────                                  │
│                                                                            │
│  ✓ Process one token at a time (handle any length)                        │
│  ✓ Same weights at every time step (parameter sharing)                    │
│  ✓ Hidden state accumulates information from all past tokens              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 Recurrent Neural Networks (RNNs)

### Core Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RNN ARCHITECTURE                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  COMPACT VIEW (Folded):                                                    │
│  ─────────────────────                                                    │
│                   ┌───────────────┐                                       │
│                   │               │                                       │
│             ┌─────┴─────┐         │                                       │
│     x_t ───►│    RNN    │────────►│ h_t                                   │
│             │   Cell    │         │                                       │
│             └─────┬─────┘         │                                       │
│                   │               │                                       │
│                   └───────────────┘                                       │
│                    h_{t-1} (feedback)                                      │
│                                                                            │
│  Output y_t can be computed from h_t                                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  UNFOLDED VIEW (Across time):                                              │
│  ───────────────────────────                                              │
│                                                                            │
│     x₀          x₁          x₂          x₃          x₄                    │
│      │           │           │           │           │                     │
│      ↓           ↓           ↓           ↓           ↓                     │
│   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                     │
│ h₀│ RNN │─h₁►│ RNN │─h₂►│ RNN │─h₃►│ RNN │─h₄►│ RNN │─h₅►                 │
│   │     │    │     │    │     │    │     │    │     │                     │
│   └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                     │
│      │          │          │          │          │                        │
│      ↓          ↓          ↓          ↓          ↓                        │
│     y₀         y₁         y₂         y₃         y₄                        │
│                                                                            │
│  Key insight: SAME WEIGHTS (same RNN cell) applied at every time step!    │
│                                                                            │
│  Parameters: W_xh, W_hh, W_hy (shared across all time steps)              │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INFORMATION FLOW:                                                         │
│  ─────────────────                                                        │
│                                                                            │
│  Time t=0: h₀ = 0 (or learned initial state)                              │
│            x₀ = first input                                                │
│            h₁ = f(x₀, h₀) = tanh(W_xh·x₀ + W_hh·h₀ + b)                   │
│                                                                            │
│  Time t=1: h₂ = f(x₁, h₁)  ← h₁ carries info about x₀                     │
│                                                                            │
│  Time t=2: h₃ = f(x₂, h₂)  ← h₂ carries info about x₀, x₁                 │
│                                                                            │
│  Time t=n: h_{n+1} = f(x_n, h_n)  ← h_n carries info about all x₀...x_{n-1}│
│                                                                            │
│  Hidden state h is the "MEMORY" of the network!                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### RNN Equations in Detail

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RNN EQUATIONS                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  At each time step t:                                                      │
│  ────────────────────                                                     │
│                                                                            │
│  1. HIDDEN STATE UPDATE:                                                   │
│                                                                            │
│     h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)                         │
│            ↑          ↑             ↑          ↑                           │
│         activation  recurrent    input      bias                           │
│                     connection  connection                                 │
│                                                                            │
│  2. OUTPUT COMPUTATION (optional, task-dependent):                         │
│                                                                            │
│     y_t = W_hy · h_t + b_y                                                │
│                                                                            │
│     For classification: y_t = softmax(W_hy · h_t + b_y)                   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DIMENSIONS:                                                               │
│  ───────────                                                              │
│                                                                            │
│  Let:                                                                      │
│    d_x = input dimension (e.g., embedding size)                           │
│    d_h = hidden state dimension                                           │
│    d_y = output dimension (e.g., vocabulary size)                         │
│                                                                            │
│  Then:                                                                     │
│    x_t ∈ ℝ^{d_x}          Input vector                                    │
│    h_t ∈ ℝ^{d_h}          Hidden state vector                             │
│    y_t ∈ ℝ^{d_y}          Output vector                                   │
│                                                                            │
│    W_xh ∈ ℝ^{d_h × d_x}   Input-to-hidden weights                        │
│    W_hh ∈ ℝ^{d_h × d_h}   Hidden-to-hidden weights (recurrent)           │
│    W_hy ∈ ℝ^{d_y × d_h}   Hidden-to-output weights                       │
│    b_h ∈ ℝ^{d_h}          Hidden bias                                     │
│    b_y ∈ ℝ^{d_y}          Output bias                                     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PARAMETER COUNT:                                                          │
│  ────────────────                                                         │
│                                                                            │
│  Total parameters = d_h × d_x + d_h × d_h + d_y × d_h + d_h + d_y         │
│                   = d_h(d_x + d_h + d_y) + d_h + d_y                       │
│                                                                            │
│  Example: d_x = 300, d_h = 512, d_y = 10,000 (vocab)                      │
│                                                                            │
│  W_xh: 512 × 300 = 153,600                                                │
│  W_hh: 512 × 512 = 262,144                                                │
│  W_hy: 10,000 × 512 = 5,120,000                                           │
│  b_h + b_y: 512 + 10,000 = 10,512                                         │
│  ────────────────────────────────                                         │
│  Total: ~5.5 million parameters                                            │
│                                                                            │
│  Note: Most parameters are in output layer (W_hy)!                        │
│        Same parameters used for sequence of ANY length.                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Python Implementation

```python
import numpy as np

class SimpleRNN:
    """
    Vanilla RNN implementation for educational purposes.

    Architecture:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization for weights
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)

        # Zero initialization for biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        self.hidden_size = hidden_size

    def forward(self, inputs, h_prev=None):
        """
        Forward pass through the RNN.

        Args:
            inputs: List of input vectors, each of shape (input_size, 1)
            h_prev: Initial hidden state. If None, uses zeros.

        Returns:
            outputs: List of output vectors
            hidden_states: List of hidden states (for backprop)
            h_final: Final hidden state
        """
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))

        hidden_states = [h_prev]  # Store for backprop
        outputs = []
        h = h_prev

        for x in inputs:
            # Hidden state update
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            hidden_states.append(h)

            # Output computation
            y = self.W_hy @ h + self.b_y
            outputs.append(y)

        return outputs, hidden_states, h

    def backward(self, inputs, hidden_states, d_outputs):
        """
        Backward pass through time (BPTT).

        Args:
            inputs: List of input vectors used in forward pass
            hidden_states: List of hidden states from forward pass
            d_outputs: List of gradients w.r.t. outputs

        Returns:
            Gradients for all parameters
        """
        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        # Backward through time
        for t in reversed(range(len(inputs))):
            # Gradient from output
            dy = d_outputs[t]
            dW_hy += dy @ hidden_states[t + 1].T
            db_y += dy

            # Gradient into hidden state
            dh = self.W_hy.T @ dy + dh_next

            # Gradient through tanh
            dh_raw = dh * (1 - hidden_states[t + 1] ** 2)  # tanh derivative

            # Parameter gradients
            dW_xh += dh_raw @ inputs[t].T
            dW_hh += dh_raw @ hidden_states[t].T
            db_h += dh_raw

            # Gradient to previous hidden state
            dh_next = self.W_hh.T @ dh_raw

        return dW_xh, dW_hh, dW_hy, db_h, db_y
```

### Types of RNN Architectures

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    RNN ARCHITECTURE PATTERNS                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. ONE-TO-MANY: Single input → Sequence output                           │
│  ─────────────────────────────────────────────                            │
│                                                                            │
│     Task: Image Captioning                                                 │
│                                                                            │
│     ┌───────┐                                                              │
│     │ Image │                                                              │
│     │ (CNN) │                                                              │
│     └───┬───┘                                                              │
│         │ feature vector                                                   │
│         ↓                                                                  │
│     ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                             │
│     │ RNN │───►│ RNN │───►│ RNN │───►│ RNN │───►...                       │
│     └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                             │
│        ↓          ↓          ↓          ↓                                 │
│       "A"      "cat"    "sitting"   "on"                                  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  2. MANY-TO-ONE: Sequence input → Single output                           │
│  ─────────────────────────────────────────────                            │
│                                                                            │
│     Task: Sentiment Classification                                         │
│                                                                            │
│     "I"       "love"     "this"     "movie"                               │
│      │          │          │          │                                   │
│      ↓          ↓          ↓          ↓                                   │
│   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                               │
│   │ RNN │───►│ RNN │───►│ RNN │───►│ RNN │                               │
│   └─────┘    └─────┘    └─────┘    └──┬──┘                               │
│                                       │                                   │
│                                       ↓                                   │
│                                  [Positive]                               │
│                                                                            │
│     Only the FINAL hidden state is used for classification               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  3. MANY-TO-MANY (same length): Sequence → Sequence                       │
│  ───────────────────────────────────────────────────                      │
│                                                                            │
│     Task: Part-of-Speech Tagging                                          │
│                                                                            │
│     "The"     "cat"      "sat"      "down"                                │
│      │          │          │          │                                   │
│      ↓          ↓          ↓          ↓                                   │
│   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                               │
│   │ RNN │───►│ RNN │───►│ RNN │───►│ RNN │                               │
│   └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                               │
│      │          │          │          │                                   │
│      ↓          ↓          ↓          ↓                                   │
│    [DET]     [NOUN]     [VERB]     [ADV]                                  │
│                                                                            │
│     One output per input token                                            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  4. MANY-TO-MANY (different length): Sequence-to-Sequence                 │
│  ────────────────────────────────────────────────────────                 │
│                                                                            │
│     Task: Machine Translation (English → French)                           │
│                                                                            │
│     "The"    "cat"    "sat"   <EOS>                                       │
│      │         │        │       │                                         │
│      ↓         ↓        ↓       ↓                                         │
│   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                     │
│   │ Enc │─►│ Enc │─►│ Enc │─►│ Enc │                                     │
│   └─────┘  └─────┘  └─────┘  └──┬──┘                                     │
│                                 │ context                                 │
│            ┌────────────────────┘ vector                                  │
│            ↓                                                              │
│         ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                               │
│   <SOS>─┤ Dec │─►│ Dec │─►│ Dec │─►│ Dec │                               │
│         └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                               │
│            │        │        │        │                                   │
│            ↓        ↓        ↓        ↓                                   │
│          "Le"    "chat"   "assis"  <EOS>                                  │
│                                                                            │
│     Encoder: Read input, compress to context vector                       │
│     Decoder: Generate output from context (variable length)               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.3 The Vanishing Gradient Problem

### Understanding the Problem

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    THE VANISHING GRADIENT PROBLEM                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  During backpropagation through time (BPTT), we need:                     │
│                                                                            │
│  ∂L/∂W_hh involves a PRODUCT of many terms:                               │
│                                                                            │
│  ∂L     ∂L    ∂h_T   ∂h_{T-1}         ∂h_2   ∂h_1                         │
│  ─── = ─── × ──── × ─────── × ... × ─── × ────                           │
│  ∂W    ∂h_T  ∂h_{T-1} ∂h_{T-2}        ∂h_1   ∂W                          │
│                                                                            │
│        └────────────────────────────────────────┘                         │
│               Product of T-1 terms!                                        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Each term ∂h_t/∂h_{t-1}:                                                 │
│  ─────────────────────────                                                │
│                                                                            │
│  h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)                              │
│                                                                            │
│  ∂h_t/∂h_{t-1} = diag(tanh'(z_t)) × W_hh                                  │
│                       ↑                ↑                                   │
│                   values ≤ 1      can be > 1 or < 1                       │
│                                                                            │
│  where z_t = W_hh · h_{t-1} + W_xh · x_t + b                              │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PROBLEM VISUALIZED:                                                   │
│  ───────────────────────                                                  │
│                                                                            │
│  tanh and its derivative:                                                  │
│                                                                            │
│  tanh(x):           tanh'(x):                                             │
│       1 ┤    ─────           1 ┤   ∩                                      │
│         │   /                  │  / \                                     │
│       0 ┼──/──────           0 ┼─/───\─────                               │
│         │ /                    │/     \                                   │
│      -1 ┤/                     └──────────────                            │
│         └─────────              -3  0  3                                  │
│          -3  0  3                                                         │
│                                                                            │
│  tanh'(x) = 1 - tanh²(x)                                                  │
│  Maximum value = 1 (at x=0)                                               │
│  For large |x|, tanh'(x) → 0  (SATURATION)                               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  VANISHING: If ||W_hh|| < 1 or tanh' < 1 consistently:                    │
│  ────────────────────────────────────────────────────                     │
│                                                                            │
│  Product = 0.9 × 0.9 × 0.9 × ... × 0.9   (T times)                        │
│          = 0.9^T                                                           │
│                                                                            │
│  T=10:   0.9^10  ≈ 0.35                                                   │
│  T=50:   0.9^50  ≈ 0.005                                                  │
│  T=100:  0.9^100 ≈ 0.00003                                                │
│  T=200:  0.9^200 ≈ 10^-9  (basically ZERO!)                               │
│                                                                            │
│  Gradient from step 200 reaches step 1 with magnitude ~10^-9              │
│  → Early weights essentially DON'T UPDATE!                                │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  EXPLODING: If ||W_hh|| > 1 consistently:                                 │
│  ────────────────────────────────────────                                 │
│                                                                            │
│  Product = 1.1 × 1.1 × 1.1 × ... × 1.1   (T times)                        │
│          = 1.1^T                                                           │
│                                                                            │
│  T=10:   1.1^10  ≈ 2.6                                                    │
│  T=50:   1.1^50  ≈ 117                                                    │
│  T=100:  1.1^100 ≈ 13,781                                                 │
│  T=200:  1.1^200 ≈ 10^8   (HUGE!)                                         │
│                                                                            │
│  → Numerical overflow, NaN values, unstable training                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Real-World Consequences

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    PRACTICAL CONSEQUENCES                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Example sentence:                                                         │
│  ─────────────────                                                        │
│  "The cat that I saw yesterday at the park near the lake sat on the ___"  │
│                                                                            │
│  Position:  1   2    3   4  5     6      7  8   9    10  11   12  13 14 15│
│            The cat that  I saw yesterday at the park near the lake sat on ?│
│             ↑                                                         ↑   │
│           word 2                                                   word 15│
│                                                                            │
│  To predict "mat" after "on the", the model needs to remember "cat"       │
│  That's 13 time steps back!                                                │
│                                                                            │
│  With vanishing gradients:                                                 │
│  ─────────────────────────                                                │
│                                                                            │
│  Gradient flow:                                                            │
│                                                                            │
│  word 15 ─────► word 14 ─────► word 13 ─────► ... ─────► word 2           │
│        ×0.9         ×0.9           ×0.9           ×0.9                     │
│                                                                            │
│  After 13 steps: gradient magnitude ≈ 0.9^13 ≈ 0.25                       │
│                                                                            │
│  But if gradient starts at 1.0, by the time it reaches "cat":             │
│  - Gradient is ~0.25 of what it should be                                  │
│  - Model CANNOT learn that "cat" is important for prediction               │
│  - Long-range dependencies are lost!                                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  VISUALIZATION OF GRADIENT MAGNITUDE:                                      │
│  ─────────────────────────────────────                                    │
│                                                                            │
│  Gradient                                                                  │
│  magnitude                                                                 │
│     ↑                                                                      │
│  1.0│ █                                                                    │
│     │ █                                                                    │
│     │ █ █                                                                  │
│     │ █ █ █                                                                │
│  0.5│ █ █ █ █                                                              │
│     │ █ █ █ █ █                                                            │
│     │ █ █ █ █ █ █                                                          │
│     │ █ █ █ █ █ █ █ █                                                      │
│  0.0│─█─█─█─█─█─█─█─█─█─█─█─█─█─█─█─────────────► time step               │
│      15 14 13 12 11 10 9  8  7  6  5  4  3  2  1                          │
│       ↑                                         ↑                          │
│    current                                   earliest                      │
│                                                                            │
│  Recent words get good gradient signal.                                    │
│  Early words barely get any gradient signal.                               │
│  Model "forgets" early context!                                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Solutions to Vanishing Gradients

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    SOLUTIONS TO VANISHING GRADIENTS                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. GRADIENT CLIPPING (for exploding gradients)                           │
│  ──────────────────────────────────────────────                           │
│                                                                            │
│  if ||gradient|| > threshold:                                              │
│      gradient = gradient × (threshold / ||gradient||)                      │
│                                                                            │
│  Prevents gradients from becoming too large                                │
│  Common threshold: 1.0 to 5.0                                              │
│                                                                            │
│  PyTorch:                                                                  │
│  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  2. BETTER ARCHITECTURES: LSTM, GRU                                       │
│  ──────────────────────────────────────                                   │
│                                                                            │
│  Design special "highways" for gradient flow                               │
│  Cell state in LSTM allows gradients to flow unchanged                     │
│                                                                            │
│  Standard RNN:     LSTM:                                                   │
│  ─────────────     ─────                                                  │
│   h ──tanh──► h    c ─────────────────► c  (additive, not multiplicative) │
│       │            h ─┬─ gates ─┬──────► h                                │
│   multiplicative      └─────────┘                                         │
│       path                                                                 │
│                                                                            │
│  LSTM gradient: ∂c_t/∂c_{t-1} = f_t (forget gate)                        │
│  If f_t ≈ 1, gradient flows through unchanged!                            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  3. SKIP CONNECTIONS / RESIDUAL CONNECTIONS                               │
│  ──────────────────────────────────────────                               │
│                                                                            │
│  h_t = f(h_{t-1}, x_t) + h_{t-1}   ← Add input directly to output         │
│                         ↑                                                  │
│                    skip connection                                         │
│                                                                            │
│  Gradient through skip: ∂h_t/∂h_{t-1} = ... + 1                          │
│  The "+1" ensures gradients always flow!                                   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  4. TRANSFORMERS (no recurrence!)                                         │
│  ─────────────────────────────────                                        │
│                                                                            │
│  Attention connects ANY position directly:                                 │
│                                                                            │
│  RNN:         x₁ ──► x₂ ──► x₃ ──► ... ──► x₁₀₀                          │
│              (100 steps to connect x₁ to x₁₀₀)                            │
│                                                                            │
│  Transformer: x₁ ←──────────────────────────► x₁₀₀                        │
│              (DIRECT connection via attention)                             │
│                                                                            │
│  No sequential dependencies = no vanishing gradients across time!          │
│  See Module 5 for details.                                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.4 Long Short-Term Memory (LSTM)

### The Key Innovation

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         LSTM KEY INNOVATION                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE CELL STATE: An "Information Highway"                                  │
│  ─────────────────────────────────────────                                │
│                                                                            │
│  Standard RNN:                                                             │
│  ─────────────                                                            │
│  h_{t-1} ──[tanh]──[W]──[tanh]──► h_t                                     │
│                                                                            │
│  Every step: information is TRANSFORMED (multiplied, squashed)             │
│  Information degrades exponentially over time.                             │
│                                                                            │
│  LSTM:                                                                     │
│  ─────                                                                    │
│  c_{t-1} ────────[×f + ×i]────────► c_t                                   │
│                    ↑   ↑                                                   │
│                  gates (learned)                                           │
│                                                                            │
│  Cell state c flows through with only:                                     │
│  - ADDITIVE updates (not multiplicative = no vanishing!)                   │
│  - Gated modifications (network learns what to keep/forget)                │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ANALOGY: The Conveyor Belt                                                │
│  ──────────────────────────                                               │
│                                                                            │
│  Think of cell state as a conveyor belt in a factory:                      │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                                                                     │  │
│  │  ══════════════════════════════════════════════════════════════    │  │
│  │  ════════════ CELL STATE (conveyor belt) ═══════════════════════   │  │
│  │  ══════════════════════════════════════════════════════════════    │  │
│  │       ↑              ↑              ↑                              │  │
│  │    [remove]        [add]         [read]                            │  │
│  │    old items     new items      for output                         │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Items (information) travel on the belt with minimal change                │
│  At each station (time step), workers can:                                 │
│  - Remove some items (forget gate)                                         │
│  - Add new items (input gate)                                              │
│  - Read items for the current task (output gate)                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### LSTM Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         LSTM CELL ARCHITECTURE                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                        c_{t-1}                                             │
│                           │                                                │
│          ┌────────────────┼────────────────────────┐                      │
│          │                │                        │                      │
│          │     ┌──────────┼──────────┐             │                      │
│          │     │          │          │             │                      │
│          │     │    ┌─────┴─────┐    │             │                      │
│          │     │    │           │    │             │                      │
│          │   ┌─┴─┐  │  ┌───────┐│  ┌─┴─┐           │                      │
│          │   │ × │──┼──│   +   ││──│ × │           │                      │
│          │   └─┬─┘  │  └───┬───┘│  └─┬─┘           │                      │
│          │     │    │      │    │    │             │                      │
│          │     │    │      │    │    │             │ c_t                  │
│          │  ┌──┴──┐ │   ┌──┴──┐ │ ┌──┴──┐          │ │                    │
│          │  │  f  │ │   │  g  │ │ │  i  │          │ │                    │
│          │  │ gate│ │   │cand.│ │ │gate │          │ │                    │
│          │  └──┬──┘ │   └──┬──┘ │ └──┬──┘          │ │                    │
│          │     │    │      │    │    │             │ │                    │
│          │     σ    │    tanh   │    σ             │ │                    │
│          │     │    │      │    │    │             │ │                    │
│          │     └────┴──────┴────┴────┘             │ │                    │
│          │                │                        │ │                    │
│          │          ┌─────┴─────┐                  │ │                    │
│          │          │ [h_{t-1}; │                  │ │                    │
│          │          │   x_t]    │                  │ │                    │
│          │          └───────────┘                  │ │                    │
│          │                                         │ │                    │
│          │                              ┌──────────┘ │                    │
│          │                              │            │                    │
│          │                           ┌──┴──┐        │                    │
│          │                           │tanh │        │                    │
│          │                           └──┬──┘        │                    │
│          │                              │           │                    │
│          │                           ┌──┴──┐     ┌──┴──┐                 │
│          │                           │  ×  │─────│  o  │                 │
│          │                           └──┬──┘     │gate │                 │
│          │                              │        └──┬──┘                 │
│     h_{t-1}───────────────────────────────────────►│                     │
│                                         │          σ                     │
│                                         │          │                     │
│                                      h_t ◄─────────┘                     │
│                                                                            │
│  Legend:                                                                   │
│  × = element-wise multiplication                                          │
│  + = element-wise addition                                                 │
│  σ = sigmoid (values 0-1, acts as gate)                                   │
│  tanh = tanh activation (values -1 to 1)                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### LSTM Equations Step by Step

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         LSTM EQUATIONS                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  STEP 1: FORGET GATE - "What should we throw away?"                       │
│  ──────────────────────────────────────────────────                       │
│                                                                            │
│  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)                                      │
│                                                                            │
│  - σ (sigmoid) outputs values between 0 and 1                             │
│  - f_t ≈ 0: Forget this information                                       │
│  - f_t ≈ 1: Keep this information                                         │
│                                                                            │
│  Example:                                                                  │
│  - "He went to the store. She bought milk."                               │
│  - When processing "She", forget gate might forget "He" was subject       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  STEP 2: INPUT GATE - "What new information should we store?"             │
│  ─────────────────────────────────────────────────────────────            │
│                                                                            │
│  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        (How much to write)           │
│  g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)     (What to write: candidates)   │
│                                                                            │
│  - i_t: Gate controlling how much of g_t to add                           │
│  - g_t: Candidate values (what we might store)                            │
│                                                                            │
│  Example:                                                                  │
│  - "The cat sat": Store "cat" as subject                                  │
│  - "on the mat": Store "sat" as action, "mat" as location                 │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  STEP 3: UPDATE CELL STATE - "Forget old, add new"                        │
│  ──────────────────────────────────────────────────                       │
│                                                                            │
│  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                                         │
│        └────┬────┘   └────┬────┘                                          │
│        What we keep   What we add                                          │
│                                                                            │
│  This is the KEY equation!                                                 │
│  - c_t is updated ADDITIVELY, not multiplicatively                        │
│  - If f_t ≈ 1 and i_t ≈ 0: c_t ≈ c_{t-1} (information preserved!)        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  STEP 4: OUTPUT GATE - "What should we output?"                           │
│  ──────────────────────────────────────────────                           │
│                                                                            │
│  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        (How much to read)            │
│  h_t = o_t ⊙ tanh(c_t)                      (Filter cell state)           │
│                                                                            │
│  - o_t: Gate controlling how much of cell state to expose                 │
│  - h_t: Hidden state (used for predictions AND next step)                 │
│                                                                            │
│  Example:                                                                  │
│  - Cell might store subject gender, but output gate only exposes it       │
│    when predicting a gendered word                                        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SUMMARY OF OPERATIONS:                                                    │
│  ──────────────────────                                                   │
│                                                                            │
│  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      # Forget gate                   │
│  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)      # Input gate                    │
│  g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   # Candidate values              │
│  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      # Output gate                   │
│                                                                            │
│  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          # Update cell state            │
│  h_t = o_t ⊙ tanh(c_t)                    # Compute hidden state         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why LSTMs Solve Vanishing Gradients

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    WHY LSTMs SOLVE VANISHING GRADIENTS                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  GRADIENT THROUGH CELL STATE:                                              │
│  ────────────────────────────                                             │
│                                                                            │
│  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                                         │
│                                                                            │
│  ∂c_t/∂c_{t-1} = f_t  (element-wise)                                      │
│                                                                            │
│  If forget gate f_t ≈ 1:                                                  │
│  ∂c_t/∂c_{t-1} ≈ 1                                                        │
│                                                                            │
│  Gradient flows DIRECTLY through without decay!                            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  COMPARISON: RNN vs LSTM gradient flow over 100 time steps                │
│  ────────────────────────────────────────────────────────                 │
│                                                                            │
│  Standard RNN:                                                             │
│  gradient = 0.9 × 0.9 × 0.9 × ... = 0.9^100 ≈ 0.00003                     │
│                                                                            │
│  LSTM (with f ≈ 0.99):                                                    │
│  gradient = 0.99 × 0.99 × 0.99 × ... = 0.99^100 ≈ 0.37                    │
│                                                                            │
│  LSTM gradient is 10,000× larger!                                         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE LEARNED GRADIENT HIGHWAY:                                             │
│  ─────────────────────────────                                            │
│                                                                            │
│  The network LEARNS when to let gradients flow:                            │
│                                                                            │
│  "The cat [that I saw yesterday at the park] sat on the ___"              │
│                                                                            │
│  Forget gate pattern: [0.99, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.99, 0.99,...]│
│                        │     └──────────────────────────┘  │              │
│                       "cat"  forget relative clause       keep "sat"      │
│                       (keep)                               context        │
│                                                                            │
│  Cell state preserves "cat" through the relative clause!                   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  VISUALIZATION OF LSTM vs RNN GRADIENT MAGNITUDE:                          │
│  ─────────────────────────────────────────────────                        │
│                                                                            │
│  Gradient                                                                  │
│  magnitude                                                                 │
│     ↑                                                                      │
│  1.0│ ██████████████████████████████████████████ LSTM                     │
│     │ ██████████████████████████████████████████                          │
│     │                                                                      │
│  0.5│                                                                      │
│     │                                                                      │
│     │                      RNN                                             │
│     │ █████████████████                                                   │
│  0.0│─█─█─█─█─█─█─█─█─█─█─█─█─█─█─█─█─█─█─█─█─────► time step            │
│      100                 50                   1                            │
│                                                                            │
│  LSTM maintains gradient magnitude across entire sequence!                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### PyTorch LSTM Implementation

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """
    Manual LSTM cell implementation for educational purposes.
    In practice, use nn.LSTM which is optimized.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined weight matrices for efficiency
        # [h_{t-1}, x_t] is concatenated, so input dim = hidden_size + input_size
        combined_size = hidden_size + input_size

        # Four gates: forget, input, candidate, output
        self.W_f = nn.Linear(combined_size, hidden_size)  # Forget gate
        self.W_i = nn.Linear(combined_size, hidden_size)  # Input gate
        self.W_g = nn.Linear(combined_size, hidden_size)  # Candidate
        self.W_o = nn.Linear(combined_size, hidden_size)  # Output gate

    def forward(self, x_t, h_prev, c_prev):
        """
        One step of LSTM.

        Args:
            x_t: Input at time t, shape [batch, input_size]
            h_prev: Previous hidden state, shape [batch, hidden_size]
            c_prev: Previous cell state, shape [batch, hidden_size]

        Returns:
            h_t: New hidden state
            c_t: New cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([h_prev, x_t], dim=1)  # [batch, hidden + input]

        # Compute gates
        f_t = torch.sigmoid(self.W_f(combined))    # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))    # Input gate
        g_t = torch.tanh(self.W_g(combined))       # Candidate values
        o_t = torch.sigmoid(self.W_o(combined))    # Output gate

        # Update cell state: forget some, add some
        c_t = f_t * c_prev + i_t * g_t

        # Compute new hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# Using PyTorch's built-in LSTM (more efficient)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,      # Input shape: [batch, seq, features]
            dropout=dropout,        # Dropout between layers
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: Token IDs, shape [batch, seq_len]
            hidden: Tuple of (h_0, c_0) or None

        Returns:
            output: Predictions for each position
            hidden: Final (h_n, c_n)
        """
        # Embed tokens
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Run through LSTM
        output, hidden = self.lstm(embedded, hidden)
        # output: [batch, seq_len, hidden_dim]
        # hidden: tuple of (h_n, c_n), each [num_layers, batch, hidden_dim]

        # Predict next token at each position
        logits = self.fc(output)  # [batch, seq_len, vocab_size]

        return logits, hidden
```

---

## 4.5 Gated Recurrent Unit (GRU)

### GRU vs LSTM

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         GRU ARCHITECTURE                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  GRU simplifies LSTM by:                                                   │
│  - Combining forget and input gates into single UPDATE gate                │
│  - Removing separate cell state (just hidden state)                        │
│  - Using RESET gate instead of output gate                                 │
│                                                                            │
│  LSTM: 4 gates (f, i, g, o) + separate c and h                            │
│  GRU:  2 gates (z, r) + single h                                          │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  GRU EQUATIONS:                                                            │
│  ──────────────                                                           │
│                                                                            │
│  Update gate (combines forget + input):                                    │
│  z_t = σ(W_z · [h_{t-1}, x_t] + b_z)                                      │
│                                                                            │
│  Reset gate (how much past to forget for candidate):                       │
│  r_t = σ(W_r · [h_{t-1}, x_t] + b_r)                                      │
│                                                                            │
│  Candidate hidden state:                                                   │
│  h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)                            │
│              ↑                                                             │
│        Reset gate controls how much                                        │
│        of previous hidden to use                                           │
│                                                                            │
│  Final hidden state (interpolate between old and new):                     │
│  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t                                   │
│        └────┬────┘           └────┬────┘                                   │
│        keep old              add new                                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  GRU ARCHITECTURE DIAGRAM:                                                 │
│  ─────────────────────────                                                │
│                                                                            │
│                     h_{t-1}                                                │
│                        │                                                   │
│        ┌───────────────┼───────────────┐                                  │
│        │               │               │                                  │
│        │     ┌─────────┴─────────┐     │                                  │
│        │     │                   │     │                                  │
│        │  ┌──┴──┐             ┌──┴──┐  │                                  │
│        │  │  ×  │◄───[r_t]───►│  ×  │  │                                  │
│        │  └──┬──┘             └──┬──┘  │                                  │
│        │     │                   │     │                                  │
│        │     │    [h̃_t]         │     │                                  │
│        │     │      ↑           │     │                                  │
│        │     │   [tanh]         │     │                                  │
│        │     │      ↑           │     │                                  │
│        │     └──────┴───────────┘     │                                  │
│        │            │                 │                                  │
│        │         [concat]             │                                  │
│        │            │                 │                                  │
│        │            x_t               │                                  │
│        │                              │                                  │
│        │     ┌──────────────────┐     │                                  │
│        │     │   z_t (update)   │     │                                  │
│        │     └────────┬─────────┘     │                                  │
│        │              │               │                                  │
│     ┌──┴──┐     ┌─────┴─────┐     ┌──┴──┐                                │
│     │ × ◄─┼─────┤  (1-z_t)  │     │  ×  │                                │
│     │(1-z)│     └───────────┘     │  z  │                                │
│     └──┬──┘                       └──┬──┘                                │
│        │                             │                                   │
│        └─────────────┬───────────────┘                                   │
│                      │                                                    │
│                      + (add)                                              │
│                      │                                                    │
│                      ▼                                                    │
│                     h_t                                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### GRU vs LSTM Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         GRU vs LSTM COMPARISON                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                          LSTM                         GRU                  │
│  ─────────────────────────────────────────────────────────────────────────│
│  States            h_t and c_t (separate)        h_t only                  │
│  Gates             4 (f, i, g, o)                2 (z, r)                  │
│  Parameters        4 × (h×h + h×x + h)           3 × (h×h + h×x + h)      │
│  Cell state        Yes (additive)                No (built into h)         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PARAMETER COUNT (input_size=300, hidden_size=512):                       │
│  ──────────────────────────────────────────────────                       │
│                                                                            │
│  LSTM:                                                                     │
│  Each gate: (512 + 300) × 512 + 512 = 416,256                             │
│  4 gates: 4 × 416,256 = 1,665,024 parameters                              │
│                                                                            │
│  GRU:                                                                      │
│  Each gate: (512 + 300) × 512 + 512 = 416,256                             │
│  3 gates: 3 × 416,256 = 1,248,768 parameters                              │
│                                                                            │
│  GRU has ~25% fewer parameters!                                           │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHEN TO USE WHICH:                                                        │
│  ──────────────────                                                       │
│                                                                            │
│  Use LSTM when:                                                            │
│  ✓ You have enough data and compute                                       │
│  ✓ Very long sequences (LSTM's cell state helps)                          │
│  ✓ You need maximum performance                                           │
│  ✓ It's the default choice for most tasks                                 │
│                                                                            │
│  Use GRU when:                                                             │
│  ✓ Training speed is important                                            │
│  ✓ Limited compute/memory                                                 │
│  ✓ Shorter sequences                                                      │
│  ✓ Similar performance to LSTM on your task                               │
│                                                                            │
│  In practice: Both work well. Try LSTM first, switch to GRU if needed.    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  EMPIRICAL COMPARISON (approximate):                                       │
│  ───────────────────────────────────                                      │
│                                                                            │
│  Task                      LSTM       GRU        Notes                     │
│  ─────────────────────────────────────────────────────────────────────────│
│  Language modeling         ✓✓         ✓✓        Similar quality           │
│  Machine translation       ✓✓✓        ✓✓        LSTM slightly better      │
│  Speech recognition        ✓✓✓        ✓✓        LSTM more common          │
│  Sentiment analysis        ✓✓         ✓✓        Similar                   │
│  Training speed            ✓          ✓✓✓       GRU ~20% faster           │
│                                                                            │
│  Legend: ✓ = good, ✓✓ = better, ✓✓✓ = best                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.6 Sequence-to-Sequence (Seq2Seq)

### Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       SEQUENCE-TO-SEQUENCE MODEL                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Task: Machine Translation (English → French)                              │
│  Input:  "I love cats"                                                     │
│  Output: "J'aime les chats"                                                │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                                                                     │  │
│  │                          ENCODER                                    │  │
│  │                                                                     │  │
│  │   "I"        "love"      "cats"      <EOS>                         │  │
│  │    │           │           │           │                           │  │
│  │    ↓           ↓           ↓           ↓                           │  │
│  │ ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                          │  │
│  │ │LSTM │───►│LSTM │───►│LSTM │───►│LSTM │                          │  │
│  │ │     │    │     │    │     │    │     │                          │  │
│  │ └─────┘    └─────┘    └─────┘    └──┬──┘                          │  │
│  │                                     │                              │  │
│  │                                     │ context                      │  │
│  │                                     │ vector                       │  │
│  │                                     │ (h, c)                       │  │
│  └─────────────────────────────────────┼───────────────────────────────┘  │
│                                        │                                  │
│                                        │                                  │
│  ┌─────────────────────────────────────┼───────────────────────────────┐  │
│  │                                     │                               │  │
│  │                          DECODER    │                               │  │
│  │                                     ↓                               │  │
│  │                                 ┌─────┐                             │  │
│  │                       <SOS> ───►│LSTM │───► "J'"                   │  │
│  │                                 └──┬──┘                             │  │
│  │                                    │                                │  │
│  │                                    ↓                                │  │
│  │                                 ┌─────┐                             │  │
│  │                         "J'" ──►│LSTM │───► "aime"                 │  │
│  │                                 └──┬──┘                             │  │
│  │                                    │                                │  │
│  │                                    ↓                                │  │
│  │                                 ┌─────┐                             │  │
│  │                       "aime" ──►│LSTM │───► "les"                  │  │
│  │                                 └──┬──┘                             │  │
│  │                                    │                                │  │
│  │                                    ↓                                │  │
│  │                                 ┌─────┐                             │  │
│  │                        "les" ──►│LSTM │───► "chats"                │  │
│  │                                 └──┬──┘                             │  │
│  │                                    │                                │  │
│  │                                    ↓                                │  │
│  │                                 ┌─────┐                             │  │
│  │                      "chats" ──►│LSTM │───► <EOS>                  │  │
│  │                                 └─────┘                             │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Encoder: Reads input, compresses to context vector                       │
│  Decoder: Generates output from context, one token at a time              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### The Bottleneck Problem

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    THE CONTEXT VECTOR BOTTLENECK                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PROBLEM: All source information must fit in ONE fixed-size vector!       │
│  ────────────────────────────────────────────────────────────────        │
│                                                                            │
│  Short sentence: "I love cats" → context vector → "J'aime les chats"      │
│  ✓ Works well! 3 words → 512-dim vector → 4 words                        │
│                                                                            │
│  Long sentence:                                                            │
│  "The quick brown fox that jumped over the lazy dog which was sleeping    │
│   under the old oak tree in the garden behind the red brick house..."     │
│                                                                            │
│  40+ words ──────────────────────────────────────────────────────►        │
│                                                                            │
│      Must squeeze ALL this into one 512-dim vector!                       │
│                                                                            │
│              ┌───────────────────────────────────────────┐                │
│              │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← Too much info!│
│              │ ░░░░░░░░░ 512-dim vector ░░░░░░░░░░░░░░░░│                │
│              │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│                │
│              └───────────────────────────────────────────┘                │
│                                                                            │
│  Information is LOST because vector can't hold everything                  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  BLEU SCORE DEGRADATION WITH LENGTH:                                       │
│  ─────────────────────────────────────                                    │
│                                                                            │
│  BLEU                                                                      │
│   ↑                                                                        │
│  50│ ████                                                                  │
│    │ ████ ████                                                             │
│  40│ ████ ████ ████                                                        │
│    │ ████ ████ ████ ████                                                   │
│  30│ ████ ████ ████ ████ ████                                              │
│    │ ████ ████ ████ ████ ████ ████                                         │
│  20│ ████ ████ ████ ████ ████ ████ ████                                    │
│    │ ████ ████ ████ ████ ████ ████ ████ ████                               │
│  10│ ████ ████ ████ ████ ████ ████ ████ ████                               │
│    │                                                                       │
│   0└──────────────────────────────────────────────► sentence length       │
│       5   10   15   20   25   30   35   40                                 │
│                                                                            │
│  Quality degrades severely for long sentences!                             │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SOLUTION: ATTENTION MECHANISM (Module 5)                                  │
│  ─────────────────────────────────────────                                │
│                                                                            │
│  Instead of one context vector, let decoder LOOK BACK at all encoder      │
│  states and focus on relevant parts:                                       │
│                                                                            │
│  Decoder generating "chats":                                               │
│  ────────────────────────────                                             │
│                                                                            │
│  Encoder states: [h₁("I"), h₂("love"), h₃("cats"), h₄(<EOS>)]            │
│                                                                            │
│  Attention weights: [0.1, 0.2, 0.7, 0.0]  ← Focus on "cats"!             │
│                             ↑                                              │
│                        Most relevant                                       │
│                                                                            │
│  Context = 0.1×h₁ + 0.2×h₂ + 0.7×h₃ + 0.0×h₄                             │
│                                                                            │
│  Different context for each decoder step!                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Teacher Forcing

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          TEACHER FORCING                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  TRAINING TIME: Use ground truth as decoder input                         │
│  ────────────────────────────────────────────────                         │
│                                                                            │
│  Target: "J'aime les chats <EOS>"                                         │
│                                                                            │
│  <SOS> ───► [LSTM] ───► predict "J'"                                      │
│                ↑                                                           │
│           context from encoder                                             │
│                                                                            │
│  "J'" (ground truth) ───► [LSTM] ───► predict "aime"                      │
│                              ↑                                             │
│                         Even if model predicted "Je"                       │
│                         we still feed correct "J'"                         │
│                                                                            │
│  "aime" (ground truth) ───► [LSTM] ───► predict "les"                     │
│  "les" (ground truth) ───► [LSTM] ───► predict "chats"                    │
│  "chats" (ground truth) ───► [LSTM] ───► predict <EOS>                    │
│                                                                            │
│  Benefits: Faster convergence, more stable training                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INFERENCE TIME: Use model's own predictions                              │
│  ───────────────────────────────────────────                              │
│                                                                            │
│  <SOS> ───► [LSTM] ───► "J'" (predicted)                                  │
│                              │                                             │
│                              ↓                                             │
│           "J'" (predicted) ───► [LSTM] ───► "aime" (predicted)            │
│                                                  │                         │
│                                                  ↓                         │
│                         "aime" (predicted) ───► [LSTM] ───► "les"         │
│                                                               │            │
│                                                               ...          │
│                                                                            │
│  No ground truth available! Model must use its own outputs.               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  EXPOSURE BIAS PROBLEM:                                                    │
│  ──────────────────────                                                   │
│                                                                            │
│  Training: Model always sees correct input                                 │
│  Inference: Model sees its own (possibly wrong) predictions               │
│                                                                            │
│  If model makes ONE mistake at inference:                                  │
│                                                                            │
│  <SOS> ───► [LSTM] ───► "Je" (WRONG! should be "J'")                      │
│                              │                                             │
│                              ↓                                             │
│          "Je" (wrong) ───► [LSTM] ───► "???"                              │
│                                         │                                  │
│                               Model never saw "Je"                         │
│                               as input during training!                    │
│                               Errors compound.                             │
│                                                                            │
│  Solutions:                                                                │
│  - Scheduled sampling: Gradually shift from teacher forcing to own preds │
│  - Reinforcement learning: Train on actual outputs                        │
│  - Beam search at inference (consider multiple hypotheses)                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.7 Bidirectional RNNs

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       BIDIRECTIONAL RNNs                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  MOTIVATION:                                                               │
│  ───────────                                                              │
│  "The movie was ___ even though the acting was great"                     │
│                   ↑                                                        │
│            Fill in the blank                                               │
│                                                                            │
│  Forward-only RNN at "___" has seen:                                       │
│  "The movie was" → Predicts: ???                                          │
│                                                                            │
│  But the FUTURE context "even though" implies CONTRAST!                    │
│  → The answer is probably negative (e.g., "terrible")                     │
│                                                                            │
│  Forward-only RNN misses this crucial information.                         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  BIDIRECTIONAL RNN ARCHITECTURE:                                           │
│  ───────────────────────────────                                          │
│                                                                            │
│     x₁        x₂        x₃        x₄        x₅                            │
│      │         │         │         │         │                             │
│      ↓         ↓         ↓         ↓         ↓                             │
│   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                             │
│   │ →   │─►│ →   │─►│ →   │─►│ →   │─►│ →   │  Forward                    │
│   │ RNN │  │ RNN │  │ RNN │  │ RNN │  │ RNN │  LSTM/GRU                   │
│   └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                             │
│      │         │         │         │         │                             │
│   ┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐                            │
│   │concat│  │concat│  │concat│  │concat│  │concat│                         │
│   └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                             │
│      │         │         │         │         │                             │
│   ┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐                            │
│   │ ←   │◄─│ ←   │◄─│ ←   │◄─│ ←   │◄─│ ←   │  Backward                   │
│   │ RNN │  │ RNN │  │ RNN │  │ RNN │  │ RNN │  LSTM/GRU                   │
│   └─────┘  └─────┘  └─────┘  └─────┘  └─────┘                             │
│                                                                            │
│   Output at position t:                                                    │
│   h_t = [h_forward_t ; h_backward_t]  (concatenation)                     │
│                                                                            │
│   If forward/backward each have dim D → output has dim 2D                 │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INFORMATION AVAILABLE AT EACH POSITION:                                   │
│  ───────────────────────────────────────                                  │
│                                                                            │
│  Position 3 (middle of sequence):                                          │
│                                                                            │
│  Forward h₃:  Sees x₁, x₂, x₃     (past + current)                        │
│  Backward h₃: Sees x₅, x₄, x₃     (future + current)                      │
│                                                                            │
│  Combined:    Full context from BOTH directions!                           │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHEN TO USE / NOT USE:                                                    │
│  ──────────────────────                                                   │
│                                                                            │
│  ✓ USE for:                                                               │
│    - Sequence labeling (NER, POS tagging)                                  │
│    - Sentence classification                                               │
│    - Encoding for seq2seq (encoder side)                                   │
│    - Any task where full sequence is available                             │
│                                                                            │
│  ✗ DON'T use for:                                                         │
│    - Language modeling (can't see future when generating)                  │
│    - Text generation (decoder side of seq2seq)                             │
│    - Real-time processing (must wait for full sequence)                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.8 Multi-layer (Stacked) RNNs

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-LAYER RNNs                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                      Layer 3 (highest abstraction)                         │
│  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                     │
│  │LSTM │───►│LSTM │───►│LSTM │───►│LSTM │───►│LSTM │───► output           │
│  └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                     │
│     │          │          │          │          │                          │
│     ↑          ↑          ↑          ↑          ↑                          │
│                                                                            │
│                      Layer 2 (medium abstraction)                          │
│  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                     │
│  │LSTM │───►│LSTM │───►│LSTM │───►│LSTM │───►│LSTM │                     │
│  └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                     │
│     │          │          │          │          │                          │
│     ↑          ↑          ↑          ↑          ↑                          │
│                                                                            │
│                      Layer 1 (low-level features)                          │
│  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                     │
│  │LSTM │───►│LSTM │───►│LSTM │───►│LSTM │───►│LSTM │                     │
│  └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                     │
│     │          │          │          │          │                          │
│     ↑          ↑          ↑          ↑          ↑                          │
│                                                                            │
│    x₁         x₂         x₃         x₄         x₅  (input)                │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  TYPICAL CONFIGURATIONS:                                                   │
│  ───────────────────────                                                  │
│                                                                            │
│  Task                         Layers    Hidden Size    Bidirectional       │
│  ──────────────────────────────────────────────────────────────────────── │
│  Sentiment classification     2         256-512        Often yes           │
│  Language modeling            2-3       512-1024       No (forward only)   │
│  Machine translation          4-8       512-1024       Encoder: yes        │
│  Speech recognition           4-6       512-1024       Often yes           │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DROPOUT IN STACKED RNNs:                                                  │
│  ────────────────────────                                                 │
│                                                                            │
│  Apply dropout BETWEEN layers (not within recurrent connections):          │
│                                                                            │
│  Layer 2 ────► [DROPOUT] ────► Layer 3                                    │
│  Layer 1 ────► [DROPOUT] ────► Layer 2                                    │
│  Input   ────► [DROPOUT] ────► Layer 1                                    │
│                                                                            │
│  PyTorch nn.LSTM has dropout parameter that handles this automatically.   │
│  (Applies dropout between layers, not on output of last layer)            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.9 Practical Considerations

### Word Embeddings

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         WORD EMBEDDINGS                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ONE-HOT ENCODING (naive approach):                                        │
│  ──────────────────────────────────                                       │
│                                                                            │
│  Vocabulary: [cat, dog, bird, fish, ...]  (|V| = 50,000 words)            │
│                                                                            │
│  "cat" → [1, 0, 0, 0, 0, ..., 0]    (50,000-dimensional)                  │
│  "dog" → [0, 1, 0, 0, 0, ..., 0]                                          │
│  "bird"→ [0, 0, 1, 0, 0, ..., 0]                                          │
│                                                                            │
│  Problems:                                                                 │
│  ✗ Extremely sparse (only one 1, rest zeros)                              │
│  ✗ Very high dimensional (50,000 dims!)                                   │
│  ✗ No similarity information: cat·dog = 0, cat·fish = 0                   │
│    (orthogonal = no relationship)                                          │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  LEARNED EMBEDDINGS (the right approach):                                  │
│  ────────────────────────────────────────                                 │
│                                                                            │
│  "cat" → [0.21, -0.53, 0.82, 0.11, -0.34, ...]  (300-dimensional)        │
│  "dog" → [0.19, -0.48, 0.79, 0.15, -0.29, ...]  (similar to cat!)        │
│  "fish"→ [-0.52, 0.31, 0.22, -0.61, 0.14, ...]  (different)              │
│                                                                            │
│  Benefits:                                                                 │
│  ✓ Dense representation (300 dims vs 50,000)                              │
│  ✓ Captures semantic similarity: cosine(cat, dog) ≈ 0.8                   │
│  ✓ Learned from data (or use pre-trained)                                 │
│                                                                            │
│  Similarity visualization:                                                 │
│                                                                            │
│               man ───────── woman                                          │
│                │              │                                            │
│              king ─────── queen                                            │
│                                                                            │
│  vector(king) - vector(man) + vector(woman) ≈ vector(queen)               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  EMBEDDING LAYER IN PYTORCH:                                               │
│  ───────────────────────────                                              │
│                                                                            │
│  # Create embedding layer                                                  │
│  embed = nn.Embedding(                                                     │
│      num_embeddings=50000,    # Vocabulary size                           │
│      embedding_dim=300        # Embedding dimension                       │
│  )                                                                         │
│                                                                            │
│  # Forward pass                                                            │
│  token_ids = torch.tensor([42, 156, 3, 891])  # Word indices             │
│  embeddings = embed(token_ids)                 # [4, 300]                 │
│                                                                            │
│  Internally: Embedding layer is just a lookup table!                       │
│  embed.weight has shape [50000, 300]                                       │
│  embed(42) returns embed.weight[42]                                        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PRE-TRAINED EMBEDDINGS:                                                   │
│  ───────────────────────                                                  │
│                                                                            │
│  Word2Vec:  Trained on Google News (100B words)                           │
│             Skip-gram and CBOW architectures                               │
│                                                                            │
│  GloVe:     Trained on Wikipedia + Common Crawl                           │
│             Global word co-occurrence statistics                           │
│                                                                            │
│  FastText:  Handles subwords (useful for rare words)                       │
│             "unfriendliness" → "un" + "friend" + "li" + "ness"            │
│                                                                            │
│  Usage:                                                                    │
│  - Initialize embedding layer with pre-trained weights                     │
│  - Fine-tune during training or keep frozen                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Complete PyTorch Pattern

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """
    Complete LSTM classifier for sequence classification.

    Architecture:
        Embedding → Bidirectional LSTM → FC → Softmax
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # Assume 0 is <PAD> token
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout between layers
            bidirectional=bidirectional
        )

        # Dropout before final layer
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        # *2 if bidirectional (concat forward and backward)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Token IDs, shape [batch, seq_len]
            lengths: Actual lengths (before padding), shape [batch]

        Returns:
            logits: Class scores, shape [batch, num_classes]
        """
        batch_size = x.size(0)

        # Embed tokens: [batch, seq_len] → [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # Pack for efficient processing of variable-length sequences
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths, sort_idx = lengths.sort(descending=True)
            embedded = embedded[sort_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True
            )

            # Run through LSTM
            packed_output, (hidden, cell) = self.lstm(packed)

            # Unsort
            _, unsort_idx = sort_idx.sort()
            hidden = hidden[:, unsort_idx]
        else:
            # No packing, process directly
            output, (hidden, cell) = self.lstm(embedded)

        # Get final hidden states
        # hidden shape: [num_layers * num_directions, batch, hidden_dim]
        if self.bidirectional:
            # Concatenate final forward and backward hidden states
            # Forward: hidden[-2], Backward: hidden[-1]
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        # Classify: [batch, hidden_dim * num_directions] → [batch, num_classes]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)

        return logits


# Usage example
model = LSTMClassifier(
    vocab_size=50000,
    embed_dim=300,
    hidden_dim=256,
    num_classes=5,  # e.g., 5-star rating
    num_layers=2,
    dropout=0.3,
    bidirectional=True
)

# Example forward pass
batch_size = 32
seq_len = 100
x = torch.randint(0, 50000, (batch_size, seq_len))  # Random token IDs
lengths = torch.randint(50, 100, (batch_size,))     # Actual lengths

logits = model(x, lengths)
print(logits.shape)  # torch.Size([32, 5])
```

---

## 4.10 Summary

### Key Concepts

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       SEQUENCE MODELS: KEY TAKEAWAYS                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. RNNs process sequences by maintaining HIDDEN STATE                     │
│     - Same weights applied at every time step                              │
│     - Hidden state = "memory" of past inputs                               │
│                                                                            │
│  2. VANISHING GRADIENTS prevent learning long-range dependencies           │
│     - Gradient = product of many small numbers → 0                        │
│     - Can't learn that distant words are related                           │
│                                                                            │
│  3. LSTMs solve vanishing gradients with CELL STATE                        │
│     - Cell state flows through with additive updates                       │
│     - Forget/input/output gates control information flow                   │
│     - ∂c_t/∂c_{t-1} = f_t (can be close to 1!)                           │
│                                                                            │
│  4. GRUs are simpler alternative to LSTMs                                  │
│     - 2 gates instead of 4                                                 │
│     - Similar performance, faster training                                 │
│                                                                            │
│  5. Seq2Seq = Encoder-Decoder for variable-length I/O                      │
│     - Encoder compresses input to context vector                           │
│     - Decoder generates output from context                                │
│     - Bottleneck problem → Attention (Module 5)                           │
│                                                                            │
│  6. Bidirectional RNNs capture context from both directions                │
│     - Can't be used for generation (needs future)                          │
│     - Great for encoding, classification, labeling                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Glossary Terms

| Term | Definition |
|------|------------|
| **RNN** | Neural network that processes sequences via hidden state |
| **Hidden State** | Internal memory vector passed between time steps |
| **LSTM** | RNN with cell state and gates for long-range dependencies |
| **Cell State** | LSTM's "memory highway" with additive updates |
| **Forget Gate** | LSTM gate controlling what to remove from cell state |
| **Input Gate** | LSTM gate controlling what new info to store |
| **Output Gate** | LSTM gate controlling what to expose from cell state |
| **GRU** | Simplified LSTM with update and reset gates |
| **Seq2Seq** | Encoder-decoder architecture for sequence transduction |
| **Teacher Forcing** | Training technique using ground truth as decoder input |
| **Bidirectional** | RNN that processes sequence in both directions |
| **Embedding** | Dense vector representation of discrete tokens |
| **Perplexity** | Evaluation metric for language models (lower = better) |

### Limitations Leading to Attention

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    LIMITATIONS OF RNNs/LSTMs                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. SEQUENTIAL PROCESSING → SLOW                                          │
│     - Must process t-1 before t                                            │
│     - Cannot parallelize across time steps                                 │
│     - Training is slow on long sequences                                   │
│                                                                            │
│  2. LONG-RANGE DEPENDENCIES still difficult                               │
│     - LSTM helps but doesn't fully solve                                   │
│     - Information still must flow step-by-step                             │
│     - 100+ steps is still challenging                                      │
│                                                                            │
│  3. FIXED-SIZE BOTTLENECK in Seq2Seq                                      │
│     - All source info compressed to one vector                             │
│     - Decoder must do everything from that vector                          │
│                                                                            │
│  These motivate ATTENTION MECHANISMS (Module 5):                           │
│  - Direct connections between any positions                                │
│  - Parallelizable computation                                              │
│  - Dynamic, query-dependent context                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.11 Exercises

1. **RNN Hidden States**: An RNN has hidden size 128 and processes a sequence of length 50. How many hidden states are computed? What is the total number of multiplications in the recurrent weight matrix application?

2. **LSTM Gates**: If the forget gate outputs 0.9 for all dimensions, what happens to information in the cell state over 10 time steps?

3. **Parameter Count**: Calculate the total parameters in an LSTM cell with input size 256 and hidden size 512.

4. **Vanishing Gradient**: If each gradient multiplication factor is 0.8, what is the gradient magnitude after 50 time steps?

5. **Code Exercise**: Implement a character-level language model using LSTM that can generate text.

---

## References

### Foundational Papers

- **LSTM**: Hochreiter & Schmidhuber, ["Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf) (1997)
- **GRU**: Cho et al., ["Learning Phrase Representations using RNN Encoder-Decoder"](https://arxiv.org/abs/1406.1078) (2014)
- **Seq2Seq**: Sutskever et al., ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) (2014)
- **Bidirectional RNNs**: Schuster & Paliwal, ["Bidirectional Recurrent Neural Networks"](https://ieeexplore.ieee.org/document/650093) (1997)

### Courses and Lectures

- [CS231n: Recurrent Neural Networks](http://cs231n.stanford.edu/)
  - [Lecture 10: RNNs](http://cs231n.stanford.edu/slides/2023/lecture_10.pdf)
- [CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
  - [Lecture 5: Recurrent Neural Networks](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture05-rnnlm.pdf)
  - [Lecture 6: LSTMs and Seq2Seq](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture06-fancy-rnn.pdf)

### Books and Resources

- **Deep Learning** by Goodfellow, Bengio, Courville
  - [Chapter 10: Sequence Modeling](https://www.deeplearningbook.org/contents/rnn.html)
- **Understanding LSTM Networks** by Chris Olah
  - [Blog Post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **The Unreasonable Effectiveness of RNNs** by Andrej Karpathy
  - [Blog Post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Implementation References

- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [PyTorch Seq2Seq Translation](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

### Word Embeddings

- **Word2Vec**: Mikolov et al., ["Efficient Estimation of Word Representations"](https://arxiv.org/abs/1301.3781) (2013)
- **GloVe**: Pennington et al., ["GloVe: Global Vectors for Word Representation"](https://nlp.stanford.edu/pubs/glove.pdf) (2014)
- **FastText**: Bojanowski et al., ["Enriching Word Vectors with Subword Information"](https://arxiv.org/abs/1607.04606) (2016)

---

*Module 4 complete. Next: Module 5 covers Attention Mechanisms and Transformers.*
