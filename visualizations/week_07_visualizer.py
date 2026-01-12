"""
Week 7 Visualizer: Attention and Transformers
Interactive ASCII visualizations for attention mechanisms.
Run: python week_07_visualizer.py
"""

import time
import os
import random
import math

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
# VISUALIZATION 1: Attention Intuition
# ============================================================

def visualize_attention_intuition():
    print_header("VISUALIZATION 1: Attention Intuition")

    print("""
What is Attention?
═══════════════════════════════════════════════════════════

Attention allows the model to FOCUS on relevant parts of input.

Example: Translation "The cat sat on the mat"

    When generating "le" (French "the"):
    ┌─────────────────────────────────────────────┐
    │ The   cat   sat   on   the   mat           │
    │ ████  ░░░   ░░░   ░░░  ███   ░░░           │
    │  ▲                      ▲                   │
    │  └──────── FOCUS ───────┘                  │
    └─────────────────────────────────────────────┘
    High attention on "The" and "the"!

    When generating "chat" (French "cat"):
    ┌─────────────────────────────────────────────┐
    │ The   cat   sat   on   the   mat           │
    │ ░░░   ████  ░░░   ░░░  ░░░   ░░░           │
    │        ▲                                    │
    │        └───── FOCUS                        │
    └─────────────────────────────────────────────┘
    High attention on "cat"!
    """)
    pause()

    print("""
Attention as Weighted Sum
═══════════════════════════════════════════════════════════

    Query (Q): "What am I looking for?"
    Keys (K):  "What information is available?"
    Values (V): "The actual information"

    Attention(Q, K, V) = softmax(Q × K^T / √d) × V

    Step by step:

    1. Compare query with all keys (dot product)
       ┌───────────────────────────────────┐
       │ Q · K_1 = 0.8   "very relevant"   │
       │ Q · K_2 = 0.2   "somewhat"        │
       │ Q · K_3 = 0.1   "not much"        │
       └───────────────────────────────────┘

    2. Normalize with softmax
       ┌───────────────────────────────────┐
       │ softmax([0.8, 0.2, 0.1])          │
       │ = [0.52, 0.28, 0.20]              │
       │   (weights sum to 1)              │
       └───────────────────────────────────┘

    3. Weighted sum of values
       ┌───────────────────────────────────┐
       │ output = 0.52×V_1 + 0.28×V_2 + 0.20×V_3 │
       └───────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 2: Self-Attention Step by Step
# ============================================================

def visualize_self_attention():
    print_header("VISUALIZATION 2: Self-Attention Step by Step")

    print("""
Self-Attention: Each position attends to ALL positions
═══════════════════════════════════════════════════════════

Input: "The cat sat"

Step 1: Create Q, K, V for each token

    Token embeddings:
    ┌─────┐  ┌─────┐  ┌─────┐
    │ The │  │ cat │  │ sat │
    │ x_1 │  │ x_2 │  │ x_3 │
    └──┬──┘  └──┬──┘  └──┬──┘
       │        │        │
       ▼        ▼        ▼
    ┌──────────────────────────┐
    │    Linear projections    │
    │    Q = x·W_Q             │
    │    K = x·W_K             │
    │    V = x·W_V             │
    └──────────────────────────┘
       │        │        │
       ▼        ▼        ▼
    Q_1,K_1,V_1  Q_2,K_2,V_2  Q_3,K_3,V_3
    """)
    pause()

    print("""
Step 2: Compute attention scores (Q × K^T)
═══════════════════════════════════════════════════════════

Each query attends to all keys:

              Keys
              K_1   K_2   K_3
            ┌─────┬─────┬─────┐
    Q_1 "The"│ 0.9 │ 0.3 │ 0.2 │  "The" mostly looks at itself
            ├─────┼─────┼─────┤
    Q_2 "cat"│ 0.4 │ 0.8 │ 0.5 │  "cat" mostly looks at itself
            ├─────┼─────┼─────┤
    Q_3 "sat"│ 0.2 │ 0.7 │ 0.9 │  "sat" looks at "cat" and itself
            └─────┴─────┴─────┘
    Queries

After softmax (rows sum to 1):
              K_1   K_2   K_3
            ┌─────┬─────┬─────┐
    Q_1     │ 0.50│ 0.28│ 0.22│
            ├─────┼─────┼─────┤
    Q_2     │ 0.24│ 0.48│ 0.28│
            ├─────┼─────┼─────┤
    Q_3     │ 0.18│ 0.38│ 0.44│
            └─────┴─────┴─────┘
    """)
    pause()

    print("""
Step 3: Compute output (attention × values)
═══════════════════════════════════════════════════════════

    Output_1 = 0.50×V_1 + 0.28×V_2 + 0.22×V_3
    Output_2 = 0.24×V_1 + 0.48×V_2 + 0.28×V_3
    Output_3 = 0.18×V_1 + 0.38×V_2 + 0.44×V_3

    ┌───────────────────────────────────────────────────┐
    │                                                   │
    │  Each output is a BLEND of all input positions   │
    │  weighted by how relevant they are!               │
    │                                                   │
    │  Output for "sat" contains info about:            │
    │    - "sat" itself (44%)                          │
    │    - "cat" (38%) ← subject-verb relationship!    │
    │    - "The" (18%)                                 │
    │                                                   │
    └───────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 3: Multi-Head Attention
# ============================================================

def visualize_multihead():
    print_header("VISUALIZATION 3: Multi-Head Attention")

    print("""
Why Multiple Heads?
═══════════════════════════════════════════════════════════

Different heads can learn different types of relationships:

    Head 1: Syntactic (subject-verb)
    ┌─────────────────────────────────────┐
    │ The   cat   that   I   saw   ran   │
    │  └────────────────────────────┘    │
    │       "cat" attends to "ran"        │
    └─────────────────────────────────────┘

    Head 2: Semantic (noun-noun)
    ┌─────────────────────────────────────┐
    │ The   cat   that   I   saw   ran   │
    │         └─────────┘                │
    │       "cat" attends to "I"          │
    └─────────────────────────────────────┘

    Head 3: Positional (nearby words)
    ┌─────────────────────────────────────┐
    │ The   cat   that   I   saw   ran   │
    │   └───┘                            │
    │       "cat" attends to "The"        │
    └─────────────────────────────────────┘
    """)
    pause()

    print("""
Multi-Head Architecture
═══════════════════════════════════════════════════════════

    Input
      │
      ├──────────────┬──────────────┬──────────────┐
      │              │              │              │
      ▼              ▼              ▼              ▼
   ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
   │Head 1│      │Head 2│      │Head 3│      │Head 4│
   │Q,K,V │      │Q,K,V │      │Q,K,V │      │Q,K,V │
   │Attn  │      │Attn  │      │Attn  │      │Attn  │
   └──┬───┘      └──┬───┘      └──┬───┘      └──┬───┘
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │ Concat   │
                    │ + Linear │
                    └────┬─────┘
                         │
                         ▼
                      Output

Each head uses smaller dimension (d_model / num_heads)
Then results are concatenated back to d_model
    """)
    pause()

    print("""
Multi-Head Computation
═══════════════════════════════════════════════════════════

    d_model = 512, num_heads = 8
    d_head = 512 / 8 = 64

    For each head h:
        Q_h = X × W_Q^h    (W_Q^h is 512×64)
        K_h = X × W_K^h
        V_h = X × W_V^h

        head_h = Attention(Q_h, K_h, V_h)  # output: 64 dims

    Combine:
        MultiHead = Concat(head_1, ..., head_8) × W_O
                  = [64, 64, 64, 64, 64, 64, 64, 64] × W_O
                  = [512 dims]

    ┌─────────────────────────────────────────────────────┐
    │ Same compute cost as single-head with full dims!    │
    │ But more expressive (different attention patterns)  │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 4: Transformer Architecture
# ============================================================

def visualize_transformer():
    print_header("VISUALIZATION 4: Transformer Architecture")

    print("""
The Transformer: "Attention Is All You Need"
═══════════════════════════════════════════════════════════

No recurrence, no convolution - ONLY attention!

                         TRANSFORMER
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  Input Tokens                  Output Tokens    │
    │       ↓                             ↑          │
    │  ┌─────────┐                  ┌─────────┐      │
    │  │Embedding│                  │ Linear  │      │
    │  │+ PositE │                  │+Softmax │      │
    │  └────┬────┘                  └────┬────┘      │
    │       │                            │          │
    │       ▼                            │          │
    │  ┌─────────┐                  ┌────┴────┐      │
    │  │ ENCODER │───────────────▶ │ DECODER │      │
    │  │   ×N    │  (cross-attn)   │   ×N    │      │
    │  └─────────┘                  └─────────┘      │
    │                                                 │
    └─────────────────────────────────────────────────┘
    """)
    pause()

    print("""
Encoder Block (×N layers)
═══════════════════════════════════════════════════════════

    Input
      │
      ▼
    ┌───────────────────────────────────┐
    │     Multi-Head Self-Attention     │
    └───────────────────┬───────────────┘
                        │
    ─────────────────Add & Norm──────────  ◄── Residual connection
                        │
    ┌───────────────────┴───────────────┐
    │        Feed-Forward Network       │
    │     (Linear → ReLU → Linear)      │
    └───────────────────┬───────────────┘
                        │
    ─────────────────Add & Norm──────────  ◄── Residual connection
                        │
                        ▼
                     Output

Residual connections: output = LayerNorm(x + Sublayer(x))
    - Helps gradient flow (like ResNet)
    - Allows stacking many layers
    """)
    pause()

    print("""
Decoder Block (×N layers)
═══════════════════════════════════════════════════════════

    Input (shifted right)
      │
      ▼
    ┌───────────────────────────────────┐
    │   MASKED Multi-Head Self-Attn     │  ◄── Can't see future!
    └───────────────────┬───────────────┘
                        │
    ─────────────────Add & Norm──────────
                        │
    ┌───────────────────┴───────────────┐
    │    Cross-Attention (Q from        │
    │    decoder, K,V from encoder)     │  ◄── Attend to encoder
    └───────────────────┬───────────────┘
                        │
    ─────────────────Add & Norm──────────
                        │
    ┌───────────────────┴───────────────┐
    │        Feed-Forward Network       │
    └───────────────────┬───────────────┘
                        │
    ─────────────────Add & Norm──────────
                        │
                        ▼
                     Output
    """)
    pause()

# ============================================================
# VISUALIZATION 5: Causal Masking
# ============================================================

def visualize_causal_mask():
    print_header("VISUALIZATION 5: Causal Masking (For Autoregressive)")

    print("""
Why Mask?
═══════════════════════════════════════════════════════════

During training, decoder sees entire target sequence.
But it must NOT peek at future tokens!

    Target: "I love Paris"

    When predicting "love":
        Can see: "I"
        Cannot see: "Paris"  ← would be cheating!

    We need a MASK to hide future positions.
    """)
    pause()

    print("""
Causal Attention Mask
═══════════════════════════════════════════════════════════

    Before softmax, add large negative to future positions:

              attend to position →
              0     1     2     3
            ┌─────┬─────┬─────┬─────┐
    pos 0   │  0  │ -∞  │ -∞  │ -∞  │  can only see itself
            ├─────┼─────┼─────┼─────┤
    pos 1   │  0  │  0  │ -∞  │ -∞  │  can see 0,1
            ├─────┼─────┼─────┼─────┤
    pos 2   │  0  │  0  │  0  │ -∞  │  can see 0,1,2
            ├─────┼─────┼─────┼─────┤
    pos 3   │  0  │  0  │  0  │  0  │  can see all
            └─────┴─────┴─────┴─────┘

    After softmax, -∞ becomes 0:

              0     1     2     3
            ┌─────┬─────┬─────┬─────┐
    pos 0   │ 1.0 │  0  │  0  │  0  │
            ├─────┼─────┼─────┼─────┤
    pos 1   │ 0.4 │ 0.6 │  0  │  0  │
            ├─────┼─────┼─────┼─────┤
    pos 2   │ 0.2 │ 0.3 │ 0.5 │  0  │
            ├─────┼─────┼─────┼─────┤
    pos 3   │ 0.1 │ 0.2 │ 0.3 │ 0.4 │
            └─────┴─────┴─────┴─────┘
    """)
    pause()

    # Animation of autoregressive generation
    print_header("VISUALIZATION 5: Autoregressive Generation")

    tokens = ["<start>", "I", "love", "Paris", "<end>"]

    for i in range(1, len(tokens)):
        clear_screen()
        print("=" * 60)
        print(f"{'AUTOREGRESSIVE GENERATION':^60}")
        print("=" * 60)

        print(f"\nStep {i}: Generating token {i}")
        print()

        # Show attention pattern
        visible = tokens[:i]
        generating = tokens[i]

        print("    Visible context:")
        print("    ┌" + "─" * (6 * len(visible) + 1) + "┐")
        print("    │", end="")
        for t in visible:
            print(f" {t:^5}", end="")
        print(" │")
        print("    └" + "─" * (6 * len(visible) + 1) + "┘")
        print()
        print("           ↓ attend")
        print()
        print(f"    Generating: [{generating}]")
        print()

        # Show attention weights (fake)
        print("    Attention distribution:")
        weights = [round(random.random(), 2) for _ in visible]
        total = sum(weights)
        weights = [round(w/total, 2) for w in weights]

        for j, (t, w) in enumerate(zip(visible, weights)):
            bar = "█" * int(w * 20)
            print(f"    {t:>8}: {bar} {w:.2f}")

        time.sleep(1.5)

    pause()

# ============================================================
# VISUALIZATION 6: Positional Encoding
# ============================================================

def visualize_positional_encoding():
    print_header("VISUALIZATION 6: Positional Encoding")

    print("""
The Position Problem
═══════════════════════════════════════════════════════════

Self-attention is PERMUTATION INVARIANT!

    Attention("cat sat the") = Attention("the cat sat")

    The order of inputs doesn't matter to pure attention.
    But order matters for language!

Solution: Add position information to embeddings.

    Final embedding = token embedding + position encoding
    """)
    pause()

    print("""
Sinusoidal Positional Encoding
═══════════════════════════════════════════════════════════

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Position 0:  [sin(0), cos(0), sin(0), cos(0), ...]
                 [  0   ,   1   ,   0   ,   1   , ...]

    Position 1:  [sin(1/1), cos(1/1), sin(1/10), cos(1/10), ...]
                 [ 0.84  ,  0.54  ,   0.10   ,   0.99   , ...]

    Visualization (for d=8):

    pos │ dim0  dim1  dim2  dim3  dim4  dim5  dim6  dim7
    ────┼────────────────────────────────────────────────
     0  │ ░░░░  ████  ░░░░  ████  ░░░░  ████  ░░░░  ████
     1  │ ████  ▒▒▒▒  ░▒▒░  ████  ░░░▒  ████  ░░░░  ████
     2  │ ████  ░░░░  ░▒▒▒  ███▒  ░░░▒  ████  ░░░░  ████
     3  │ ░▒▒▒  ░░░░  ░▒▒▒  ███░  ░░░▒  ████  ░░░░  ████
     4  │ ░░░░  ▒▒▒▒  ░░▒▒  ██▒▒  ░░░▒  ████  ░░░░  ████

    Different frequencies for different dimensions!
    Low dims: fast oscillation (nearby positions differ)
    High dims: slow oscillation (captures long-range)
    """)
    pause()

    print("""
Why Sinusoidal Works
═══════════════════════════════════════════════════════════

Property: PE(pos+k) can be expressed as linear function of PE(pos)

    This allows the model to easily learn RELATIVE positions!

    PE(pos+k) = f(PE(pos))  (learnable linear relationship)

Modern alternatives:

    1. Learned positional embeddings
       - Just learn a vector for each position
       - Works well, less inductive bias

    2. Rotary Position Embeddings (RoPE)
       - Encode position in rotation angle
       - Better for long sequences
       - Used in LLaMA, GPT-NeoX

    3. ALiBi (Attention with Linear Biases)
       - Add position-based bias to attention scores
       - No extra parameters
    """)
    pause()

# ============================================================
# VISUALIZATION 7: KV Cache for Inference
# ============================================================

def visualize_kv_cache():
    print_header("VISUALIZATION 7: KV Cache for Fast Inference")

    print("""
The Inference Problem
═══════════════════════════════════════════════════════════

Autoregressive generation:
    Token 1 → Token 2 → Token 3 → Token 4 → ...

At each step, we recompute attention over ALL previous tokens.

    Step 1: Compute Q,K,V for "The"
    Step 2: Compute Q,K,V for "The cat"
    Step 3: Compute Q,K,V for "The cat sat"
    ...

    We're recomputing K,V for "The" every single step!
    This is O(n²) redundant work!
    """)
    pause()

    print("""
KV Cache Solution
═══════════════════════════════════════════════════════════

CACHE the K and V values from previous steps!

    Step 1: Compute K_1, V_1 for "The"
            Cache: K=[K_1], V=[V_1]

    Step 2: Compute K_2, V_2 for "cat" ONLY
            Cache: K=[K_1,K_2], V=[V_1,V_2]
            Compute attention using cached K,V

    Step 3: Compute K_3, V_3 for "sat" ONLY
            Cache: K=[K_1,K_2,K_3], V=[V_1,V_2,V_3]
            Compute attention using cached K,V

    ┌────────────────────────────────────────────────────┐
    │  Without cache: O(n²) operations per sequence     │
    │  With cache:    O(n) operations per sequence      │
    │                                                    │
    │  For a 1000-token sequence: 1000x faster!         │
    └────────────────────────────────────────────────────┘
    """)
    pause()

    # Animation of KV cache
    print_header("VISUALIZATION 7: KV Cache Animation")

    tokens = ["The", "cat", "sat", "on"]

    for i, token in enumerate(tokens):
        clear_screen()
        print("=" * 60)
        print(f"{'KV CACHE ANIMATION':^60}")
        print("=" * 60)

        print(f"\nStep {i+1}: Processing '{token}'")
        print()

        # Show what's cached
        print("    KV Cache:")
        print("    ┌" + "─" * 40 + "┐")
        for j in range(i+1):
            status = "█ cached" if j < i else "▶ computing"
            print(f"    │ K_{j+1}, V_{j+1} ({tokens[j]}) {status:>20} │")
        print("    └" + "─" * 40 + "┘")
        print()

        # Show attention computation
        print("    Attention computation:")
        print(f"    Q_{i+1} (new) × [K_1...K_{i+1}] (cached) = weights")
        print(f"    weights × [V_1...V_{i+1}] (cached) = output")
        print()

        if i > 0:
            print(f"    Saved computation: {i} K,V pairs reused!")

        time.sleep(1.5)

    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print_header("WEEK 7: ATTENTION AND TRANSFORMERS")
        print("""
Choose a visualization:

    [1] Attention Intuition
        - Q, K, V and weighted sums

    [2] Self-Attention Step by Step
        - Computing attention for "The cat sat"

    [3] Multi-Head Attention
        - Why multiple heads, architecture

    [4] Transformer Architecture
        - Encoder, decoder, residual connections

    [5] Causal Masking
        - Preventing future peek, autoregressive generation

    [6] Positional Encoding
        - The position problem, sinusoidal encoding

    [7] KV Cache
        - Fast inference for autoregressive models

    [A] Run ALL visualizations

    [Q] Quit
        """)

        choice = input("Enter choice: ").strip().upper()

        if choice == '1':
            visualize_attention_intuition()
        elif choice == '2':
            visualize_self_attention()
        elif choice == '3':
            visualize_multihead()
        elif choice == '4':
            visualize_transformer()
        elif choice == '5':
            visualize_causal_mask()
        elif choice == '6':
            visualize_positional_encoding()
        elif choice == '7':
            visualize_kv_cache()
        elif choice == 'A':
            visualize_attention_intuition()
            visualize_self_attention()
            visualize_multihead()
            visualize_transformer()
            visualize_causal_mask()
            visualize_positional_encoding()
            visualize_kv_cache()
        elif choice == 'Q':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
