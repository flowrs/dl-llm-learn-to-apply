"""
Week 11 Visualizer: Large Language Models
Interactive ASCII visualizations for LLM concepts.
Run: python week_11_visualizer.py
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
# VISUALIZATION 1: Tokenization
# ============================================================

def visualize_tokenization():
    print_header("VISUALIZATION 1: Tokenization")

    print("""
Why Tokenization?
═══════════════════════════════════════════════════════════

Neural networks need NUMBERS, not text!

    "Hello, world!" → [15496, 11, 995, 0]

Three approaches:

    CHARACTER-LEVEL:
    "Hello" → ['H','e','l','l','o'] → [72,101,108,108,111]
    ✗ Very long sequences
    ✗ Hard to learn meaning

    WORD-LEVEL:
    "Hello world" → ['Hello', 'world'] → [1234, 5678]
    ✗ Huge vocabulary (100K+ words)
    ✗ Can't handle new words → [UNK]

    SUBWORD (BPE):  ← Best of both worlds!
    "unhappiness" → ['un', 'happiness'] → [592, 8234]
    ✓ Reasonable vocabulary (~50K)
    ✓ Can handle any word
    """)
    pause()

    # BPE animation
    print_header("VISUALIZATION 1: BPE Training Animation")

    print("Training BPE on: ['low', 'lower', 'lowest', 'new', 'newer']\n")

    steps = [
        ("Initial (character level)",
         "l o w | l o w e r | l o w e s t | n e w | n e w e r"),
        ("Merge 'e' + 'r' → 'er'",
         "l o w | l o w er | l o w e s t | n e w | n e w er"),
        ("Merge 'e' + 's' → 'es'",
         "l o w | l o w er | l o w es t | n e w | n e w er"),
        ("Merge 'es' + 't' → 'est'",
         "l o w | l o w er | l o w est | n e w | n e w er"),
        ("Merge 'l' + 'o' → 'lo'",
         "lo w | lo w er | lo w est | n e w | n e w er"),
        ("Merge 'lo' + 'w' → 'low'",
         "low | low er | low est | n e w | n e w er"),
        ("Merge 'n' + 'e' → 'ne'",
         "low | low er | low est | ne w | ne w er"),
        ("Merge 'ne' + 'w' → 'new'",
         "low | low er | low est | new | new er"),
    ]

    for i, (action, result) in enumerate(steps):
        clear_screen()
        print("=" * 60)
        print(f"{'BPE TRAINING':^60}")
        print("=" * 60)
        print(f"\nStep {i}: {action}\n")
        print(f"Tokens: {result}")
        print()

        vocab = set()
        for token in result.replace("|", "").split():
            vocab.add(token)
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Vocabulary: {sorted(vocab)}")

        time.sleep(1.2)

    print("""

Final vocabulary: ['low', 'er', 'est', 'new', ...]

Now tokenizing new words:
    "lowest"  → ['low', 'est']     ✓
    "newest"  → ['new', 'est']     ✓
    "lower"   → ['low', 'er']      ✓
    "newish"  → ['new', 'ish']     ✓ (handles unseen!)
    """)
    pause()

# ============================================================
# VISUALIZATION 2: Pre-training
# ============================================================

def visualize_pretraining():
    print_header("VISUALIZATION 2: Pre-training (Next Token Prediction)")

    print("""
The Core Task: Predict the Next Token
═══════════════════════════════════════════════════════════

Training data: Massive text corpus (books, web, code, etc.)

    Input:  "The cat sat on the"
    Target: "mat"

    Input:  "Paris is the capital of"
    Target: "France"

    Input:  "def fibonacci(n):"
    Target: "\\n"

    Loss = -log P(correct_token | previous_tokens)

    Repeat BILLIONS of times across TRILLIONS of tokens!
    """)
    pause()

    # Animated next token prediction
    print_header("VISUALIZATION 2: Next Token Prediction")

    prompt = "The quick brown fox"
    continuations = ["jumps", "over", "the", "lazy", "dog"]

    for i, next_token in enumerate(continuations):
        clear_screen()
        print("=" * 60)
        print(f"{'NEXT TOKEN PREDICTION':^60}")
        print("=" * 60)

        current = prompt + " " + " ".join(continuations[:i])
        print(f"\nInput: \"{current}\"")
        print()
        print("    Probability distribution over vocabulary:")
        print("    ┌" + "─" * 40 + "┐")

        # Show top predictions
        if next_token == "jumps":
            probs = [("jumps", 0.35), ("runs", 0.15), ("leaps", 0.12), ("walked", 0.08)]
        elif next_token == "over":
            probs = [("over", 0.65), ("across", 0.12), ("through", 0.08), ("into", 0.05)]
        elif next_token == "the":
            probs = [("the", 0.72), ("a", 0.15), ("his", 0.05), ("my", 0.03)]
        elif next_token == "lazy":
            probs = [("lazy", 0.45), ("sleeping", 0.18), ("old", 0.12), ("brown", 0.08)]
        else:
            probs = [("dog", 0.55), ("cat", 0.15), ("hound", 0.10), ("fox", 0.05)]

        for token, prob in probs:
            bar = "█" * int(prob * 30)
            marker = " ← CORRECT" if token == next_token else ""
            print(f"    │ {token:>10}: {bar} {prob:.2f}{marker}")

        print("    │      ...")
        print("    └" + "─" * 40 + "┘")
        print(f"\n    Selected: \"{next_token}\"")

        time.sleep(1.5)

    print(f"\nFinal: \"{prompt} {' '.join(continuations)}\"")
    pause()

# ============================================================
# VISUALIZATION 3: Scaling Laws
# ============================================================

def visualize_scaling():
    print_header("VISUALIZATION 3: Scaling Laws")

    print("""
The Key Discovery: Performance Scales Predictably
═══════════════════════════════════════════════════════════

    Loss ≈ C / N^0.076 + C / D^0.095

    N = Number of parameters
    D = Dataset size (tokens)

    ┌─────────────────────────────────────────────────────┐
    │  Loss                                               │
    │   │                                                 │
    │ 4 │●                                               │
    │   │ ●                                              │
    │ 3 │   ●                                            │
    │   │     ●●                                         │
    │ 2 │        ●●●                                     │
    │   │            ●●●●●                               │
    │ 1 │                  ●●●●●●●●●●●●●●●               │
    │   └───────────────────────────────────────────────  │
    │     1B      10B      100B     1T      Parameters    │
    └─────────────────────────────────────────────────────┘

    Each 10x increase in compute → predictable loss decrease
    """)
    pause()

    print("""
Chinchilla Scaling Laws (2022)
═══════════════════════════════════════════════════════════

Key finding: Most models were UNDERTRAINED!

    GPT-3:      175B params, 300B tokens
    Chinchilla:  70B params, 1.4T tokens  ← Same compute, better!

    Optimal ratio: ~20 tokens per parameter

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Before Chinchilla:    After Chinchilla:            │
    │                                                     │
    │  "Make the model      "Balance model size           │
    │   bigger!"             and data!"                   │
    │                                                     │
    │  ┌───────────┐        ┌─────┐                       │
    │  │  HUGE     │        │     │ + lots of data        │
    │  │  MODEL    │   →    │model│ ████████████          │
    │  │           │        │     │ ████████████          │
    │  └───────────┘        └─────┘                       │
    │   few data                                          │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

    print("""
Emergent Capabilities
═══════════════════════════════════════════════════════════

Some abilities appear SUDDENLY at certain scales:

    Capability      │ Appears at
    ────────────────┼─────────────
    Basic grammar   │    ~1B
    Coherent text   │    ~7B
    Basic math      │   ~13B
    Chain-of-thought│   ~60B
    Complex reasoning│  ~175B+

    ┌─────────────────────────────────────────────────────┐
    │  Accuracy                                           │
    │     │                         ┌─────────────────    │
    │ 100%│                         │                     │
    │     │                        ╱                      │
    │  50%│                       ╱                       │
    │     │                      ╱                        │
    │   0%│──────────────────────╯                        │
    │     └───────────────────────────────────────────    │
    │       1B    10B    100B  Parameters                 │
    │                                                     │
    │  Not gradual improvement - sudden jumps!            │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 4: LoRA Fine-tuning
# ============================================================

def visualize_lora():
    print_header("VISUALIZATION 4: LoRA (Low-Rank Adaptation)")

    print("""
The Fine-tuning Problem
═══════════════════════════════════════════════════════════

Full fine-tuning:
    - Update ALL parameters
    - 70B model = 280GB memory (fp32)
    - Need multiple high-end GPUs

    ┌───────────────────────────────────────────────────┐
    │  W: [4096 × 4096] = 16 million parameters        │
    │                                                   │
    │  ████████████████████████████████████████████    │
    │  ████████████████████████████████████████████    │
    │  ████████████████████████████████████████████    │
    │  ████████████████████████████████████████████    │
    │                                                   │
    │  ALL trainable (expensive!)                       │
    └───────────────────────────────────────────────────┘
    """)
    pause()

    print("""
LoRA: Train Only Low-Rank Updates
═══════════════════════════════════════════════════════════

Instead of updating W directly, add low-rank matrices A and B:

    W_new = W_frozen + A × B

    W: [4096 × 4096]  (FROZEN - 16M params)
    A: [4096 × 16]    (trainable - 65K params)
    B: [16 × 4096]    (trainable - 65K params)

    Total trainable: 130K vs 16M = 0.8%!

    ┌───────────────────────────────────────────────────┐
    │                                                   │
    │        ┌───────────────────────────────┐         │
    │   x ──▶│      W (frozen, 4096×4096)    │──┐      │
    │        └───────────────────────────────┘  │      │
    │                                           ├──▶ y │
    │        ┌──────┐         ┌──────┐          │      │
    │   x ──▶│ A    │────────▶│ B    │─────────┘      │
    │        │(4096 │         │(16×  │                 │
    │        │ ×16) │         │4096) │                 │
    │        └──────┘         └──────┘                 │
    │        (trainable)      (trainable)              │
    │                                                   │
    └───────────────────────────────────────────────────┘

    Output = W×x + A×B×x  (frozen + learned adjustment)
    """)
    pause()

    print("""
Why Low-Rank Works
═══════════════════════════════════════════════════════════

Hypothesis: Task-specific updates have low intrinsic dimension

    Pre-trained model knows "general language"
    Fine-tuning adds "task-specific knowledge"

    This delta (ΔW) is often LOW-RANK!

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Full fine-tuning updates (ΔW):                     │
    │  ┌───────────────────────────────────┐              │
    │  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  Looks       │
    │  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  random but  │
    │  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  actually... │
    │  └───────────────────────────────────┘              │
    │                                                     │
    │  SVD decomposition shows:                           │
    │  ┌────┐   ┌────────────────────────────┐           │
    │  │████│ × │▓▓░░░░░░░░░░░░░░░░░░░░░░░░░│           │
    │  │████│   └────────────────────────────┘           │
    │  │    │                                            │
    │  │    │   Most singular values ≈ 0!                │
    │  └────┘   Only a few dimensions matter.            │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 5: RLHF
# ============================================================

def visualize_rlhf():
    print_header("VISUALIZATION 5: RLHF (Reinforcement Learning from Human Feedback)")

    print("""
The Alignment Problem
═══════════════════════════════════════════════════════════

Pre-trained LLMs are HELPFUL but not ALIGNED:

    User: "How do I break into my neighbor's house?"

    Base model: "Here are the steps: 1. Check for unlocked
                 windows... 2. Use a credit card on the
                 door lock..."

    Aligned model: "I can't help with breaking into
                    someone's property. If you're locked
                    out of your OWN home, I can suggest..."

RLHF trains models to be helpful, harmless, and honest.
    """)
    pause()

    print("""
RLHF Pipeline: Three Steps
═══════════════════════════════════════════════════════════

Step 1: SUPERVISED FINE-TUNING (SFT)
    Train on human-written helpful responses

    [Prompt] → [Good Response written by human]
    [Prompt] → [Good Response written by human]
    ...

Step 2: REWARD MODEL TRAINING
    Humans rank model outputs, train reward model

    Prompt: "Explain quantum computing"
    Response A: [Technical, helpful explanation]
    Response B: [Vague, unhelpful response]

    Human says: A > B
    Reward model learns: R(A) > R(B)

Step 3: RL FINE-TUNING (PPO)
    Optimize model to maximize reward

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  ┌─────────┐    ┌───────────┐    ┌──────────────┐  │
    │  │ Prompt  │───▶│   Model   │───▶│   Response   │  │
    │  └─────────┘    └───────────┘    └──────┬───────┘  │
    │                       ▲                  │          │
    │                       │                  ▼          │
    │                  Update          ┌──────────────┐  │
    │                       │          │Reward Model  │  │
    │                       │          │   R(resp)    │  │
    │                       │          └──────┬───────┘  │
    │                       └──────────────────┘          │
    │                         Maximize reward             │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

    print("""
The KL Penalty Trick
═══════════════════════════════════════════════════════════

Problem: Model might "hack" the reward!

    Without constraint:
    "Amazing! Wonderful! You're the best! I love helping
     you! This is so exciting! Great question!!!!!"

    High reward (sycophantic) but not actually helpful.

Solution: Penalize divergence from original model

    Objective = Reward - β × KL(policy || reference)
                  ↑                    ↑
            Maximize this      Stay close to base model

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  The model should:                                  │
    │    ✓ Generate helpful responses (high reward)       │
    │    ✓ Sound like the original model (low KL)         │
    │                                                     │
    │  This prevents reward hacking while allowing        │
    │  beneficial changes in behavior.                    │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 6: Prompting Techniques
# ============================================================

def visualize_prompting():
    print_header("VISUALIZATION 6: Prompting Techniques")

    print("""
Zero-shot vs Few-shot
═══════════════════════════════════════════════════════════

ZERO-SHOT: Just ask directly
    ┌─────────────────────────────────────────────────────┐
    │ Classify the sentiment: "I love this product!"     │
    │                                                     │
    │ → positive                                          │
    └─────────────────────────────────────────────────────┘

FEW-SHOT: Provide examples first
    ┌─────────────────────────────────────────────────────┐
    │ Classify the sentiment:                             │
    │                                                     │
    │ "Great quality!" → positive                         │
    │ "Terrible waste of money" → negative                │
    │ "It works fine" → neutral                           │
    │                                                     │
    │ "I love this product!" →                            │
    │                                                     │
    │ → positive  (much more reliable!)                   │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

    print("""
Chain-of-Thought (CoT)
═══════════════════════════════════════════════════════════

Problem: LLMs often fail at reasoning

    Q: "A bat and ball cost $1.10. The bat costs $1 more
        than the ball. How much does the ball cost?"

    Without CoT: "$0.10"  ✗ WRONG!

    With CoT:
    ┌─────────────────────────────────────────────────────┐
    │ "Let me think step by step.                        │
    │                                                     │
    │  Let the ball cost x dollars.                       │
    │  The bat costs x + $1.                              │
    │  Total: x + (x + 1) = $1.10                         │
    │  2x + 1 = 1.10                                      │
    │  2x = 0.10                                          │
    │  x = $0.05                                          │
    │                                                     │
    │  The ball costs $0.05."  ✓ CORRECT!                │
    └─────────────────────────────────────────────────────┘

Magic phrase: "Let's think step by step"
    """)
    pause()

    print("""
Advanced Prompting Patterns
═══════════════════════════════════════════════════════════

SELF-CONSISTENCY: Multiple samples + majority vote

    Run same prompt 5 times:
    Response 1: "42"
    Response 2: "42"
    Response 3: "38"  ← outlier
    Response 4: "42"
    Response 5: "42"

    Final answer: 42 (4/5 votes)

TREE OF THOUGHTS: Explore multiple reasoning paths

              Problem
             /   |   \\
           A     B     C      ← Generate options
          /|    |     |\\
         A1 A2  B1    C1 C2   ← Explore branches
         ✗  ✓   ✗     ✓  ✗    ← Evaluate
            |         |
           ...       ...      ← Continue good paths

REFLECTION: Ask model to critique then improve

    Step 1: Generate answer
    Step 2: "What's wrong with this answer?"
    Step 3: "Improve based on critique"
    """)
    pause()

# ============================================================
# VISUALIZATION 7: RAG (Retrieval Augmented Generation)
# ============================================================

def visualize_rag():
    print_header("VISUALIZATION 7: RAG (Retrieval Augmented Generation)")

    print("""
The Knowledge Problem
═══════════════════════════════════════════════════════════

LLMs have limitations:
    ✗ Knowledge cutoff (training data date)
    ✗ Can't access your private documents
    ✗ May hallucinate facts

Solution: Retrieve relevant context, then generate

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  User Query: "What's our refund policy?"            │
    │        │                                            │
    │        ▼                                            │
    │  ┌──────────────┐   ┌─────────────────────────────┐│
    │  │  Embedding   │──▶│   Vector Database           ││
    │  │    Model     │   │ (your company documents)    ││
    │  └──────────────┘   └─────────────┬───────────────┘│
    │                                    │                │
    │                     Retrieved: "Refunds within 30  ││
    │                     days with receipt..."          ││
    │                                    │                │
    │                                    ▼                │
    │  ┌─────────────────────────────────────────────────┐│
    │  │ LLM receives: Query + Retrieved Context         ││
    │  │                                                 ││
    │  │ Generates: "Based on our policy, you can get   ││
    │  │ a refund within 30 days if you have your       ││
    │  │ receipt..."                                     ││
    │  └─────────────────────────────────────────────────┘│
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

    # RAG animation
    print_header("VISUALIZATION 7: RAG Pipeline Animation")

    steps = [
        ("Step 1: User Query", """
    User: "What are the side effects of aspirin?"

    ┌─────────────────────────────────────────────────────┐
    │  "What are the side effects of aspirin?"            │
    └─────────────────────────────────────────────────────┘
        """),
        ("Step 2: Embed Query", """
    Query → Embedding Model → [0.12, -0.34, 0.56, ...]

    ┌──────────────────┐     ┌───────────────────────────┐
    │ Query            │ ──▶ │ [0.12, -0.34, 0.56, ...]  │
    └──────────────────┘     └───────────────────────────┘
        """),
        ("Step 3: Search Vector Database", """
    Query embedding → Find similar document chunks

    ┌───────────────────────────────────────────────────┐
    │  Vector DB                                        │
    │  ┌────────────────────────────────────────────┐  │
    │  │ Doc 1: Aspirin overview...     (0.89 sim)  │◀─│─ Most similar!
    │  │ Doc 2: Aspirin side effects... (0.92 sim)  │◀─│─ Best match!
    │  │ Doc 3: Ibuprofen info...       (0.45 sim)  │  │
    │  │ Doc 4: Heart medication...     (0.32 sim)  │  │
    │  └────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────┘
        """),
        ("Step 4: Augment Prompt", """
    Combine query with retrieved context:

    ┌───────────────────────────────────────────────────┐
    │ CONTEXT:                                          │
    │ "Aspirin side effects include stomach upset,      │
    │  bleeding risk, allergic reactions. Common        │
    │  symptoms: nausea, heartburn. Serious: GI         │
    │  bleeding, Reye's syndrome in children..."        │
    │                                                   │
    │ QUERY: What are the side effects of aspirin?      │
    │                                                   │
    │ Answer based on the context above.                │
    └───────────────────────────────────────────────────┘
        """),
        ("Step 5: Generate Response", """
    LLM generates grounded response:

    ┌───────────────────────────────────────────────────┐
    │ "Based on medical information, aspirin can cause  │
    │  several side effects:                            │
    │                                                   │
    │  Common: stomach upset, nausea, heartburn         │
    │  Serious: GI bleeding, allergic reactions         │
    │  Warning: Risk of Reye's syndrome in children     │
    │                                                   │
    │  Consult your doctor if you experience..."        │
    └───────────────────────────────────────────────────┘
        """),
    ]

    for title, content in steps:
        clear_screen()
        print("=" * 60)
        print(f"{'RAG PIPELINE':^60}")
        print("=" * 60)
        print(f"\n{title}")
        print(content)
        time.sleep(2)

    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print_header("WEEK 11: LARGE LANGUAGE MODELS")
        print("""
Choose a visualization:

    [1] Tokenization
        - BPE training, subword tokens

    [2] Pre-training
        - Next token prediction animation

    [3] Scaling Laws
        - Parameter vs performance, emergence

    [4] LoRA Fine-tuning
        - Low-rank adaptation explained

    [5] RLHF
        - Reward model, PPO, alignment

    [6] Prompting Techniques
        - Zero-shot, few-shot, chain-of-thought

    [7] RAG
        - Retrieval augmented generation pipeline

    [A] Run ALL visualizations

    [Q] Quit
        """)

        choice = input("Enter choice: ").strip().upper()

        if choice == '1':
            visualize_tokenization()
        elif choice == '2':
            visualize_pretraining()
        elif choice == '3':
            visualize_scaling()
        elif choice == '4':
            visualize_lora()
        elif choice == '5':
            visualize_rlhf()
        elif choice == '6':
            visualize_prompting()
        elif choice == '7':
            visualize_rag()
        elif choice == 'A':
            visualize_tokenization()
            visualize_pretraining()
            visualize_scaling()
            visualize_lora()
            visualize_rlhf()
            visualize_prompting()
            visualize_rag()
        elif choice == 'Q':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
