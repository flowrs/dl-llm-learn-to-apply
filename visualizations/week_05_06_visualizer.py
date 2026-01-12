"""
Week 5-6 Visualizer: Self-Supervised Learning and RNNs
Interactive ASCII visualizations for SSL and sequence models.
Run: python week_05_06_visualizer.py
"""

import time
import os
import random

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause(msg="Press Enter to continue..."):
    input(f"\n{msg}")

def print_header(title):
    clear_screen()
    width = 60
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
    print()

# ============================================================
# VISUALIZATION 1: Self-Supervised Learning Overview
# ============================================================

def visualize_ssl_overview():
    print_header("VISUALIZATION 1: Self-Supervised Learning")

    print("""
The Data Labeling Problem
═══════════════════════════════════════════════════════════

Supervised Learning:
    ┌─────────┐     Human      ┌─────────┐
    │  Image  │ ──Labeling──▶ │  "cat"  │
    └─────────┘    (Expensive!) └─────────┘

    For ImageNet: 14 million images × $0.10/label = $1.4 million!

Self-Supervised Learning:
    ┌─────────┐    Automatic    ┌─────────┐
    │  Data   │ ──Processing──▶ │  Labels │
    └─────────┘      (Free!)    └─────────┘

    The data ITSELF provides the supervision signal!
    """)
    pause()

    print("""
SSL Paradigms
═══════════════════════════════════════════════════════════

1. PRETEXT TASKS: Create labels from data transformations

    Original Image    →    Task: Predict rotation
    ┌───────────┐          ┌───────────┐
    │    🐱     │          │    🐱     │  Label: 0°
    │           │          │   ↷90°   │  Label: 90°
    └───────────┘          │   ↷180°  │  Label: 180°
                           │   ↷270°  │  Label: 270°
                           └───────────┘

2. CONTRASTIVE LEARNING: Learn similar vs different

    Same image, different augmentations → SIMILAR
    Different images → DIFFERENT

    ┌────┐ ≈ ┌────┐     (same image, augmented)
    │ 🐱 │   │ 🐱 │
    └────┘   └────┘

    ┌────┐ ≠ ┌────┐     (different images)
    │ 🐱 │   │ 🐕 │
    └────┘   └────┘

3. MASKED PREDICTION: Predict hidden parts

    "The cat sat on the [MASK]" → predict "mat"
    """)
    pause()

# ============================================================
# VISUALIZATION 2: Contrastive Learning
# ============================================================

def visualize_contrastive():
    print_header("VISUALIZATION 2: Contrastive Learning (SimCLR)")

    print("""
SimCLR Pipeline
═══════════════════════════════════════════════════════════

Step 1: Take an image and create two augmented views

    Original Image
         │
    ┌────┴────┐
    ↓         ↓
┌───────┐ ┌───────┐
│ Aug 1 │ │ Aug 2 │     (random crop, flip, color jitter, blur)
│       │ │       │
│  x_i  │ │  x_j  │
└───────┘ └───────┘

These are "positive pairs" - they should have similar embeddings!
    """)
    pause()

    print("""
Step 2: Encode both views with the same network

    ┌───────┐         ┌───────┐
    │  x_i  │         │  x_j  │
    └───┬───┘         └───┬───┘
        │                 │
        ▼                 ▼
    ┌─────────────────────────┐
    │       Encoder f(·)      │  (ResNet backbone)
    └─────────────────────────┘
        │                 │
        ▼                 ▼
    ┌───────┐         ┌───────┐
    │  h_i  │         │  h_j  │   (representations)
    └───┬───┘         └───┬───┘
        │                 │
        ▼                 ▼
    ┌─────────────────────────┐
    │     Projection g(·)     │  (MLP)
    └─────────────────────────┘
        │                 │
        ▼                 ▼
    ┌───────┐         ┌───────┐
    │  z_i  │         │  z_j  │   (for contrastive loss)
    └───────┘         └───────┘
    """)
    pause()

    print("""
Step 3: Contrastive loss (NT-Xent)
═══════════════════════════════════════════════════════════

For a batch of N images, we have 2N augmented samples.
Each sample has 1 positive pair and (2N-2) negative pairs.

    In a batch of 4 images (8 augmented samples):

    ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
    │z_1a│ │z_1b│ │z_2a│ │z_2b│ │z_3a│ │z_3b│ │z_4a│ │z_4b│
    └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
       └───┬───┘     └───┬───┘     └───┬───┘     └───┬───┘
        positive      positive      positive      positive
          pair          pair          pair          pair

    For z_1a:
      Positive: z_1b          (same image, different aug)
      Negatives: z_2a, z_2b, z_3a, z_3b, z_4a, z_4b

    Loss = -log(similarity(z_1a, z_1b) / sum of all similarities)

    Pull positives TOGETHER, push negatives APART!
    """)
    pause()

    print("""
Embedding Space Visualization
═══════════════════════════════════════════════════════════

Before training:                After training:
(random embeddings)             (clustered by semantic)

        ×   ○                        ○○○
    △     ×                         ○○
       ○      △                   ×
    ×       ○                   ×××
         △                        ×
    ○    ×    ○                       △△
              △                    △△△

    × = cats                    Same class = nearby
    ○ = dogs                    Different class = far
    △ = birds

    The network learns WHAT makes images similar!
    """)
    pause()

# ============================================================
# VISUALIZATION 3: RNN Basics
# ============================================================

def visualize_rnn():
    print_header("VISUALIZATION 3: Recurrent Neural Networks")

    print("""
Why RNNs?
═══════════════════════════════════════════════════════════

Problem: Process SEQUENCES of variable length

    "The cat sat on the mat"
         ↓   ↓   ↓  ↓   ↓   ↓
        [6 words - sequence]

Feed-forward networks expect fixed-size input!

RNN Solution: Process one element at a time, maintain "memory"

    h_0 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3
           ↑            ↑            ↑
          x_1          x_2          x_3
         "The"        "cat"        "sat"
    """)
    pause()

    print("""
RNN Architecture (Unrolled)
═══════════════════════════════════════════════════════════

The same weights W are used at every time step!

                 Unrolled through time:

    h_0     h_1     h_2     h_3     h_4
     │       │       │       │       │
     ▼       ▼       ▼       ▼       ▼
    ┌─┐     ┌─┐     ┌─┐     ┌─┐     ┌─┐
    │ │────▶│ │────▶│ │────▶│ │────▶│ │
    │A│     │A│     │A│     │A│     │A│
    └─┘     └─┘     └─┘     └─┘     └─┘
     ▲       ▲       ▲       ▲       ▲
     │       │       │       │       │
    x_0     x_1     x_2     x_3     x_4

    All boxes labeled "A" share the SAME WEIGHTS

Equations:
    h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
    y_t = W_hy × h_t + b_y
    """)
    pause()

    # Animated RNN processing
    print_header("VISUALIZATION 3: RNN Processing (Animated)")

    sentence = ["The", "cat", "sat"]
    states = [
        "Initial state (zero)",
        "Processed 'The'",
        "Processed 'cat'",
        "Processed 'sat'"
    ]

    for i, state in enumerate(states):
        clear_screen()
        print("=" * 60)
        print(f"{'RNN PROCESSING ANIMATION':^60}")
        print("=" * 60)

        print(f"\nProcessing: {' '.join(sentence[:i])}")
        print(f"Current state: {state}\n")

        # Draw the unrolled RNN
        print("    ", end="")
        for j in range(4):
            if j <= i:
                print(f"  h_{j}  ", end="")
            else:
                print("        ", end="")
        print()

        print("    ", end="")
        for j in range(4):
            if j < i:
                print("───▶───", end="")
            elif j == i:
                print("  ●    ", end="")
            else:
                print("       ", end="")
        print()

        print("    ", end="")
        for j in range(4):
            if j <= i:
                print("  ▲    ", end="")
            else:
                print("       ", end="")
        print()

        print("    ", end="")
        for j in range(3):
            if j < i:
                word = f"'{sentence[j]}'"
                print(f"{word:^7}", end="")
            else:
                print("       ", end="")
        print()

        if i > 0:
            print(f"\n    Hidden state h_{i} now contains information about:")
            print(f"    [{' → '.join(sentence[:i])}]")

        time.sleep(1.5)

    pause()

# ============================================================
# VISUALIZATION 4: Vanishing Gradient Problem
# ============================================================

def visualize_vanishing_gradient():
    print_header("VISUALIZATION 4: Vanishing Gradient Problem")

    print("""
The Problem with Long Sequences
═══════════════════════════════════════════════════════════

During backpropagation through time (BPTT), gradients are
multiplied at each time step:

    ∂L/∂h_0 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0

    h_0 ─→ h_1 ─→ h_2 ─→ h_3 ─→ ... ─→ h_T ─→ Loss
     ←     ←     ←     ←           ←     ←
   ×W_hh ×W_hh ×W_hh ×W_hh       ×W_hh  gradient

If |W_hh| < 1: gradient → 0 (vanishing)
If |W_hh| > 1: gradient → ∞ (exploding)
    """)
    pause()

    print("""
Gradient Flow Visualization
═══════════════════════════════════════════════════════════

Forward pass (information flows right):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶

    h_0    h_1    h_2    h_3    h_4    h_5    Loss
    ┌─┐───▶┌─┐───▶┌─┐───▶┌─┐───▶┌─┐───▶┌─┐───▶│L│
    │ │    │ │    │ │    │ │    │ │    │ │    └─┘
    └─┘    └─┘    └─┘    └─┘    └─┘    └─┘

Backward pass (gradient flows left):
    ◀━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    h_0    h_1    h_2    h_3    h_4    h_5    Loss
    ┌─┐◀───┌─┐◀───┌─┐◀───┌─┐◀───┌─┐◀───┌─┐◀───│L│
    │ │    │ │    │ │    │ │    │ │    │ │    └─┘
    └─┘    └─┘    └─┘    └─┘    └─┘    └─┘
    0.01   0.05   0.13   0.33   0.57   0.82   1.0
    ↑                                         ↑
    Vanished!                           Original gradient

    Gradients shrink exponentially → early layers don't learn!
    """)
    pause()

    print("""
Why This Matters
═══════════════════════════════════════════════════════════

Example: "The man who wore the hat that was red went home"
          ↑                                        ↑
          Subject                                  Verb

For the network to learn "man" → "went" (singular agreement),
the gradient must flow through 10+ time steps!

    "The" → "man" → "who" → "wore" → ... → "went"
      ↑                                       ↑
      Need to connect these!

With vanishing gradients:
    By the time gradient reaches "man", it's nearly zero.
    The network can't learn long-range dependencies!

Solutions:
    1. LSTM (Long Short-Term Memory)
    2. GRU (Gated Recurrent Unit)
    3. Skip connections
    4. Attention mechanisms (Week 7!)
    """)
    pause()

# ============================================================
# VISUALIZATION 5: LSTM
# ============================================================

def visualize_lstm():
    print_header("VISUALIZATION 5: Long Short-Term Memory (LSTM)")

    print("""
LSTM: Solving the Vanishing Gradient Problem
═══════════════════════════════════════════════════════════

Key innovation: The CELL STATE (c_t)
A "highway" that information can flow through unchanged!

            ┌────────────────────────────────────────────┐
            │              Cell state (c_t)              │
            │  ─────────────────────────────────────────▶│
            │      ×         +                          │
            │      ↑         ↑                          │
            │   ┌──┴──┐   ┌──┴──┐                       │
            │   │ f_t │   │i_t×c̃│                       │
            └───┴─────┴───┴─────┴───────────────────────┘
                  ↑         ↑
    ┌─────────────┴─────────┴─────────────┐
    │           Hidden state (h_t)         │
    │    ┌───┐  ┌───┐  ┌───┐  ┌───┐       │
    │    │ f │  │ i │  │ c̃ │  │ o │       │
    │    └───┘  └───┘  └───┘  └───┘       │
    │      ↑      ↑      ↑      ↑         │
    └──────┴──────┴──────┴──────┴─────────┘
                     │
         [h_{t-1}, x_t] (concatenated input)

    f = forget gate
    i = input gate
    c̃ = candidate values
    o = output gate
    """)
    pause()

    print("""
The Three Gates
═══════════════════════════════════════════════════════════

1. FORGET GATE (f_t): What to throw away from cell state

    "Read the book that he gave me yesterday"
    When we see "yesterday", forget old context about "book"

    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # 0 = forget, 1 = keep

    c_{old} = [0.8, 0.3, 0.9, 0.2]
         ×
    f_t     = [0.1, 1.0, 0.0, 0.9]    "Forget positions 0 and 2"
         ↓
    result  = [0.08, 0.3, 0.0, 0.18]

2. INPUT GATE (i_t): What new info to store

    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # How much to write
    c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c) # What to write

3. OUTPUT GATE (o_t): What to output

    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    h_t = o_t × tanh(c_t)
    """)
    pause()

    print("""
LSTM Step-by-Step
═══════════════════════════════════════════════════════════

Input: "I love Paris" → predict next word

Step 1: "I"
    ┌──────────────────────────────────────┐
    │ c: [subject=I, verb=?, object=?]     │
    │ h: encoding of "I"                   │
    └──────────────────────────────────────┘

Step 2: "love"
    Forget: nothing yet
    Input: store "love" as verb
    ┌──────────────────────────────────────┐
    │ c: [subject=I, verb=love, object=?]  │
    │ h: encoding of "I love"              │
    └──────────────────────────────────────┘

Step 3: "Paris"
    Forget: nothing important
    Input: store "Paris" as object
    ┌──────────────────────────────────────┐
    │ c: [subject=I, verb=love, obj=Paris] │
    │ h: encoding of "I love Paris"        │
    └──────────────────────────────────────┘

    Output: Use h to predict next word
    """)
    pause()

    print("""
Why LSTM Solves Vanishing Gradients
═══════════════════════════════════════════════════════════

Cell state provides an "uninterrupted" gradient highway:

Regular RNN:              LSTM:
    h_0 → h_1 → h_2           c_0 ───────────────▶ c_T
     ×     ×     ×                  +     +
    W_hh  W_hh  W_hh               (additive)
    (multiplicative)

    Gradient = W^T            Gradient = Σ (additions)
    Vanishes exponentially!   Can flow unchanged!

    ┌─────────────────────────────────────────────────────┐
    │ The cell state acts like a "conveyor belt"          │
    │ Information can flow for 100+ time steps!           │
    │                                                     │
    │ Gates control what gets ON and OFF the belt.        │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 6: GRU
# ============================================================

def visualize_gru():
    print_header("VISUALIZATION 6: Gated Recurrent Unit (GRU)")

    print("""
GRU: Simplified LSTM
═══════════════════════════════════════════════════════════

GRU combines forget + input gates into a single "update" gate.
Fewer parameters, similar performance!

LSTM (4 gates):              GRU (2 gates):
    f_t = forget gate        z_t = update gate
    i_t = input gate         r_t = reset gate
    o_t = output gate
    c̃_t = candidate

    ┌────────────────┐       ┌────────────────┐
    │ f, i, o, c̃     │       │ z, r, h̃        │
    │ 4 × weight     │       │ 3 × weight     │
    │ matrices       │       │ matrices       │
    └────────────────┘       └────────────────┘
         LSTM                      GRU
    """)
    pause()

    print("""
GRU Equations
═══════════════════════════════════════════════════════════

    z_t = σ(W_z · [h_{t-1}, x_t])     # Update gate
    r_t = σ(W_r · [h_{t-1}, x_t])     # Reset gate
    h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])  # Candidate
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # New hidden state

Visual:
                    ┌─────────────────────────────┐
    h_{t-1} ───────┬┤         z_t                 ├──▶ h_t
                   ││    (update gate)            │
    x_t ──────────┬┤│                             │
                  │││  ┌───────────────────┐      │
                  ││└─▶│ (1-z)⊙h + z⊙h̃    │──────┤
                  ││   └───────────────────┘      │
                  ││         ↑                    │
                  │└────▶ r_t (reset) ──▶ h̃_t    │
                  └───────────────────────────────┘

Update gate z_t:
    z_t ≈ 1: Use new candidate h̃_t (update)
    z_t ≈ 0: Keep old h_{t-1} (remember)
    """)
    pause()

    print("""
LSTM vs GRU Comparison
═══════════════════════════════════════════════════════════

                    │  LSTM           │  GRU
    ────────────────┼─────────────────┼─────────────────
    Gates           │  3 + candidate  │  2 + candidate
    Parameters      │  4 × (hidden²)  │  3 × (hidden²)
    Separate memory │  Yes (c_t)      │  No (just h_t)
    Performance     │  ≈ same         │  ≈ same
    Training speed  │  Slower         │  Faster

When to use which?
    ┌─────────────────────────────────────────────────────┐
    │ GRU:  Smaller datasets, faster training needed      │
    │ LSTM: Very long sequences, more capacity needed     │
    │                                                     │
    │ In practice: Try both, see what works!              │
    │ Or: Use Transformers (Week 7)                       │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 7: Sequence-to-Sequence
# ============================================================

def visualize_seq2seq():
    print_header("VISUALIZATION 7: Sequence-to-Sequence Models")

    print("""
Seq2Seq: Map one sequence to another
═══════════════════════════════════════════════════════════

Applications:
    - Machine translation: "Hello" → "Bonjour"
    - Summarization: [long text] → [short summary]
    - Chatbots: [question] → [answer]

Architecture: Encoder-Decoder

    Input sequence          Output sequence
    "How are you"           "Comment allez-vous"
         │                        ▲
         ▼                        │
    ┌─────────┐              ┌─────────┐
    │ ENCODER │──▶ context ─▶│ DECODER │
    └─────────┘              └─────────┘
    """)
    pause()

    print("""
Encoder-Decoder Architecture
═══════════════════════════════════════════════════════════

                    ENCODER                 DECODER
    ┌───────────────────────────┐  ┌───────────────────────────┐
    │                           │  │                           │
    │  h_1 → h_2 → h_3 → [c]    │  │  [c] → s_1 → s_2 → s_3   │
    │   ↑     ↑     ↑           │  │         ↓     ↓     ↓    │
    │  x_1   x_2   x_3          │  │        y_1   y_2   y_3   │
    │ "How" "are" "you"         │  │      "Comme" "nt" "..."  │
    │                           │  │                           │
    └───────────────────────────┘  └───────────────────────────┘

    context vector [c] = final encoder hidden state
    Contains "meaning" of input sentence!

    Decoder generates output one token at a time,
    using previous output as next input.
    """)
    pause()

    print("""
The Bottleneck Problem
═══════════════════════════════════════════════════════════

Problem: ALL input information must fit in fixed-size context!

    "The quick brown fox jumps over the lazy dog"
                        ↓
                   ┌─────────┐
                   │ context │  (256 or 512 dimensions)
                   │ [c]     │
                   └─────────┘
                        ↓
    "Le renard brun rapide saute par-dessus le chien paresseux"

    For long sentences, context can't capture everything!

Solution: ATTENTION (Week 7!)
    Instead of single context vector,
    decoder can "look at" all encoder hidden states.

    ┌───────────────────────────────────────┐
    │   h_1   h_2   h_3   h_4   h_5   h_6  │
    │    ↑     ↑     ↑     ↑     ↑     ↑  │ ← Attention weights
    │   0.1   0.1   0.6   0.1   0.05  0.05│   (sum to 1)
    └───────────────────────────────────────┘
                        ↓
                   Weighted sum for each decoder step!
    """)
    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print_header("WEEK 5-6: SSL AND RNNs")
        print("""
Choose a visualization:

    [1] Self-Supervised Learning Overview
        - Pretext tasks, contrastive learning, masking

    [2] Contrastive Learning (SimCLR)
        - Positive pairs, negative pairs, loss function

    [3] RNN Basics
        - Recurrent architecture, unrolling, equations

    [4] Vanishing Gradient Problem
        - Why RNNs struggle with long sequences

    [5] LSTM
        - Gates, cell state, solving vanishing gradients

    [6] GRU
        - Simplified gating, comparison with LSTM

    [7] Sequence-to-Sequence
        - Encoder-decoder, the bottleneck problem

    [A] Run ALL visualizations

    [Q] Quit
        """)

        choice = input("Enter choice: ").strip().upper()

        if choice == '1':
            visualize_ssl_overview()
        elif choice == '2':
            visualize_contrastive()
        elif choice == '3':
            visualize_rnn()
        elif choice == '4':
            visualize_vanishing_gradient()
        elif choice == '5':
            visualize_lstm()
        elif choice == '6':
            visualize_gru()
        elif choice == '7':
            visualize_seq2seq()
        elif choice == 'A':
            visualize_ssl_overview()
            visualize_contrastive()
            visualize_rnn()
            visualize_vanishing_gradient()
            visualize_lstm()
            visualize_gru()
            visualize_seq2seq()
        elif choice == 'Q':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
