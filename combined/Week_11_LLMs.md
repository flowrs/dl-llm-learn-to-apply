# Week 11: Large Language Models

> Combined guide merging CSE 493G1/CS231n course materials with Stanford CS231n notes

## Overview

This document covers:
- Tokenization and vocabulary
- Pre-training objectives (next token prediction)
- Scaling laws and emergent capabilities
- Fine-tuning (full, LoRA, QLoRA)
- RLHF and alignment
- Prompting techniques
- RAG and agents

---

## Part 1: Tokenization

### Why Tokenization?

```
    From Text to Numbers
    ════════════════════

    Neural networks need numbers, not text!

    "Hello world" → [15496, 995] → Embeddings → Transformer

    Options:
    1. Character-level: "Hello" → ['H','e','l','l','o'] → [72,101,108,108,111]
       - Tiny vocab (~256)
       - Very long sequences
       - Hard to learn meaning

    2. Word-level: "Hello" → ['Hello'] → [15496]
       - Huge vocab (100K+ words)
       - Unknown word problem
       - Can't handle typos

    3. Subword (BPE): "Hello" → ['Hel', 'lo'] → [15496, 995]
       - Medium vocab (30K-100K)
       - Handles unknown words
       - Best of both worlds ✓
```

### Byte Pair Encoding (BPE)

```
    BPE Algorithm
    ═════════════

    Start: character-level vocabulary

    Repeat until vocab size reached:
    1. Count all adjacent pair frequencies
    2. Merge most frequent pair into new token
    3. Add new token to vocabulary

    Example:
    Text: "low lower lowest"

    Initial: ['l','o','w',' ','l','o','w','e','r',' ',...]

    Step 1: Most frequent pair = ('l','o')
            Merge → 'lo'
            Text: ['lo','w',' ','lo','w','e','r',' ',...]

    Step 2: Most frequent pair = ('lo','w')
            Merge → 'low'
            Text: ['low',' ','low','e','r',' ',...]

    Continue until desired vocab size...
```

### Token Quirks

```
    Tokenization Gotchas
    ════════════════════

    Same word, different tokens based on context:

    " hello"  → [Token A]     (space + word)
    "hello"   → [Token B]     (word alone)
    "Hello"   → [Token C]     (capitalized)

    Numbers are especially tricky:

    "123"   → ['1', '23']  or ['12', '3']  (inconsistent!)
    " 1000" → [' 1000']
    "1000"  → ['100', '0']

    Implications for LLMs:
    - Math is hard (numbers split weirdly)
    - Spacing matters
    - Case sensitivity varies
```

---

## Part 2: Pre-training

### Next Token Prediction

```
    Autoregressive Language Modeling
    ════════════════════════════════

    Training objective: Predict next token

    Input:  "The cat sat on the"
    Target: "cat sat on the mat"

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Input:  [The] [cat] [sat] [on]  [the]             │
    │            │     │     │     │     │                │
    │            ▼     ▼     ▼     ▼     ▼                │
    │        ┌─────────────────────────────────┐          │
    │        │    Transformer (causal mask)    │          │
    │        └─────────────────────────────────┘          │
    │            │     │     │     │     │                │
    │            ▼     ▼     ▼     ▼     ▼                │
    │  Predict: [cat] [sat] [on]  [the] [mat]            │
    │                                                     │
    │  Loss = -log P(next_token | previous_tokens)        │
    │       = Cross-entropy over vocabulary               │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    This simple objective learns:
    - Grammar and syntax
    - World knowledge
    - Reasoning patterns
    - Many "emergent" capabilities
```

### Pre-training Data

```
    Data Scale
    ══════════

    Modern LLMs train on trillions of tokens:

    ┌────────────────────────────────────────────────────┐
    │ Source              │ Tokens    │ Quality          │
    ├────────────────────────────────────────────────────┤
    │ Web crawl (Common   │ Trillions │ Noisy, diverse   │
    │ Crawl, etc.)        │           │                  │
    │ Books               │ Billions  │ High quality     │
    │ Wikipedia           │ Billions  │ Factual          │
    │ Code (GitHub)       │ Billions  │ Structured       │
    │ Scientific papers   │ Billions  │ Technical        │
    │ Conversations       │ Billions  │ Dialogue         │
    └────────────────────────────────────────────────────┘

    Data quality > Data quantity
    (but you need both!)
```

### Training Infrastructure

```
    Distributed Training
    ════════════════════

    GPT-3 scale: 175B parameters

    Memory needed:
    - Parameters: 175B × 4 bytes = 700 GB
    - Optimizer states: 2× more = 1.4 TB
    - Activations: varies with batch

    Solution: Parallelism

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Data Parallelism:                                  │
    │  Same model on each GPU, different data batches     │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │
    │  │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│                   │
    │  │batch│ │batch│ │batch│ │batch│                   │
    │  │  0  │ │  1  │ │  2  │ │  3  │                   │
    │  └─────┘ └─────┘ └─────┘ └─────┘                   │
    │                                                     │
    │  Model/Tensor Parallelism:                          │
    │  Split model across GPUs                            │
    │  ┌─────────────────────────────────┐               │
    │  │ Layers 0-10 │ Layers 11-20 │ ...│               │
    │  │   GPU 0     │    GPU 1     │    │               │
    │  └─────────────────────────────────┘               │
    │                                                     │
    │  Pipeline Parallelism:                              │
    │  Different layers on different GPUs + micro-batches │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

---

## Part 3: Scaling Laws

### The Scaling Law

```
    Chinchilla Scaling Laws
    ═══════════════════════

    Performance predictably improves with scale:

    Loss ∝ (Parameters)^(-α) + (Tokens)^(-β) + constant

    Key finding: Compute-optimal training

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Given compute budget C:                            │
    │                                                     │
    │  Optimal params N ∝ C^0.5                           │
    │  Optimal tokens D ∝ C^0.5                           │
    │                                                     │
    │  Rule of thumb: ~20 tokens per parameter            │
    │                                                     │
    │  Model          │ Params │ Optimal Tokens          │
    │  ───────────────┼────────┼────────────────         │
    │  GPT-3          │ 175B   │ ~3.5T (undertrained)    │
    │  Chinchilla     │ 70B    │ ~1.4T (optimal!)        │
    │  LLaMA          │ 65B    │ ~1.4T                   │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    Implication: Train smaller models longer!
```

### Emergent Capabilities

```
    Emergence: Sudden Capability Jumps
    ══════════════════════════════════

    Some abilities appear "suddenly" at scale:

    Performance
        │
        │                      ╱──────  Large model
        │                    ╱
        │                  ●
        │                ╱
        │        ──────╱
        │       Medium
        │      ─────────────  Small model
        │
        └──────────────────────────▶ Scale

    Examples of emergent abilities:
    - Multi-step reasoning
    - Code generation
    - Mathematical problem solving
    - Following complex instructions
    - In-context learning

    Note: Whether these are "true emergence" or
    measurement artifacts is debated!
```

---

## Part 4: Fine-tuning

### Fine-tuning Approaches

```
    Fine-tuning Spectrum
    ════════════════════

    Full Fine-tuning:
    - Update ALL parameters
    - Most flexible, most expensive
    - Risk of catastrophic forgetting

    Parameter-Efficient Fine-tuning (PEFT):
    - Update only SOME parameters
    - Much cheaper
    - Better preservation of capabilities

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Full FT      PEFT              Prompting           │
    │  ◀────────────────────────────────────────────────▶ │
    │  All params   Few params        Zero params         │
    │  Expensive    Cheap             Free                │
    │  Flexible     Balanced          Limited             │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### LoRA (Low-Rank Adaptation)

```
    LoRA: Efficient Fine-tuning
    ═══════════════════════════

    Key insight: Weight updates are low-rank

    Instead of: W_new = W_old + ΔW        (full update)
    Use:        W_new = W_old + B × A     (low-rank update)

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Original weight W: [d × k]  (frozen)               │
    │                                                     │
    │  LoRA adapters:                                     │
    │  A: [r × k]    (trainable, r << d)                 │
    │  B: [d × r]    (trainable, r << d)                 │
    │                                                     │
    │  Output = W×x + B×A×x                              │
    │                                                     │
    │  Example: d=4096, k=4096, r=8                       │
    │  Full FT:  4096 × 4096 = 16M params                 │
    │  LoRA:     (4096×8 + 8×4096) = 65K params          │
    │  Reduction: 250× fewer trainable params!            │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### QLoRA

```
    QLoRA: Quantized LoRA
    ═════════════════════

    Combine quantization with LoRA:

    1. Quantize base model to 4-bit (NF4)
    2. Add LoRA adapters (16-bit)
    3. Train only adapters

    Memory comparison for 65B model:

    Method         │ Memory  │ Trainable
    ───────────────┼─────────┼──────────
    Full FT        │ 780 GB  │ 65B
    LoRA           │ 130 GB  │ ~100M
    QLoRA          │ 48 GB   │ ~100M   ← Fits on 1 GPU!

    QLoRA enables fine-tuning huge models on consumer hardware!
```

---

## Part 5: RLHF (Reinforcement Learning from Human Feedback)

### The Alignment Problem

```
    Why RLHF?
    ═════════

    Pre-training objective: Predict next token
    What we want: Helpful, harmless, honest responses

    Problem: These aren't the same!

    Pre-trained model might:
    - Complete harmful requests
    - Generate plausible but wrong info
    - Be verbose or off-topic

    RLHF aligns model behavior with human preferences
```

### RLHF Pipeline

```
    Three-Stage RLHF
    ════════════════

    Stage 1: Supervised Fine-tuning (SFT)
    ┌─────────────────────────────────────────────────────┐
    │  Human-written demonstrations                       │
    │  Prompt: "Explain quantum computing"                │
    │  Response: [High-quality human response]            │
    │                                                     │
    │  Fine-tune base model on these examples             │
    └─────────────────────────────────────────────────────┘
                           │
                           ▼
    Stage 2: Reward Model Training
    ┌─────────────────────────────────────────────────────┐
    │  Generate multiple responses per prompt             │
    │  Humans rank responses: A > B > C                   │
    │                                                     │
    │  Train reward model to predict human preferences    │
    │  R(prompt, response) → scalar score                │
    └─────────────────────────────────────────────────────┘
                           │
                           ▼
    Stage 3: RL Fine-tuning (PPO)
    ┌─────────────────────────────────────────────────────┐
    │  Policy = SFT model                                 │
    │  Reward = Reward model + KL penalty                 │
    │                                                     │
    │  Optimize: max E[R(x, y)] - β × KL(π || π_ref)     │
    │                                                     │
    │  KL penalty prevents forgetting/reward hacking      │
    └─────────────────────────────────────────────────────┘
```

### DPO (Direct Preference Optimization)

```
    DPO: Simpler Alternative to RLHF
    ════════════════════════════════

    Skip the reward model!

    RLHF: Train reward model → Train policy with RL
    DPO:  Directly optimize policy from preferences

    Loss:
    L = -log σ(β × (log π(y_w|x) - log π(y_l|x)
                  - log π_ref(y_w|x) + log π_ref(y_l|x)))

    Where:
    - y_w = preferred (winning) response
    - y_l = dispreferred (losing) response
    - π = current policy
    - π_ref = reference (SFT) policy

    Benefits:
    - Simpler pipeline
    - More stable
    - Often similar results to RLHF
```

---

## Part 6: Prompting

### Basic Prompting

```
    Prompting Techniques
    ════════════════════

    Zero-shot:
    "Translate to French: Hello world"
    → "Bonjour le monde"

    Few-shot (In-context learning):
    "Translate to French:
     Hello → Bonjour
     Goodbye → Au revoir
     World → "
    → "Monde"

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Few-shot prompting gives examples IN THE PROMPT    │
    │  No weight updates, just conditioning!              │
    │                                                     │
    │  Works because LLMs learn to learn during training  │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### Chain-of-Thought (CoT)

```
    Chain-of-Thought Prompting
    ══════════════════════════

    Standard:
    Q: "If I have 3 apples and buy 2 more, how many?"
    A: "5"

    Chain-of-Thought:
    Q: "If I have 3 apples and buy 2 more, how many?
        Let's think step by step."
    A: "I start with 3 apples.
        I buy 2 more apples.
        3 + 2 = 5.
        I have 5 apples."

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Why it works:                                      │
    │  - Breaks complex problems into steps               │
    │  - Intermediate results in context                  │
    │  - More "compute" via longer generation             │
    │                                                     │
    │  Variants:                                          │
    │  - Zero-shot CoT: "Let's think step by step"       │
    │  - Few-shot CoT: Provide reasoning examples         │
    │  - Self-consistency: Sample multiple, vote          │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### Advanced Prompting

```
    Advanced Techniques
    ═══════════════════

    Self-Consistency:
    1. Generate N different reasoning chains
    2. Take majority vote on final answer
    3. More robust than single chain

    Tree-of-Thought:
    1. Generate multiple reasoning branches
    2. Evaluate each branch
    3. Explore promising paths
    4. Backtrack if needed

    ReAct (Reasoning + Acting):
    1. Think: reasoning about what to do
    2. Act: take action (e.g., search)
    3. Observe: get result
    4. Repeat until done
```

---

## Part 7: RAG (Retrieval-Augmented Generation)

### Why RAG?

```
    RAG: Grounding LLMs with External Knowledge
    ═══════════════════════════════════════════

    Problems with vanilla LLMs:
    - Knowledge cutoff (don't know recent events)
    - Hallucinations (confident but wrong)
    - No source attribution
    - Can't access private data

    RAG solution:
    Query → Retrieve relevant docs → Generate with context

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  User: "What was Apple's revenue in Q4 2024?"      │
    │                                                     │
    │         ┌─────────────────────────────────┐        │
    │         │        Knowledge Base           │        │
    │         │  ┌──────┐ ┌──────┐ ┌──────┐    │        │
    │         │  │Doc 1 │ │Doc 2 │ │Doc 3 │    │        │
    │         │  └──────┘ └──────┘ └──────┘    │        │
    │         └───────────────┬─────────────────┘        │
    │                         │ retrieve                  │
    │                         ▼                           │
    │  ┌─────────────────────────────────────────────┐   │
    │  │  Context: "Apple reported Q4 2024 revenue   │   │
    │  │  of $94.9 billion..."                       │   │
    │  │                                             │   │
    │  │  Query: "What was Apple's revenue..."       │   │
    │  └─────────────────────────────────────────────┘   │
    │                         │                           │
    │                         ▼                           │
    │  Answer: "Apple's Q4 2024 revenue was $94.9B"      │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### RAG Pipeline

```
    RAG Components
    ══════════════

    1. Document Processing:
       - Chunk documents into passages
       - Typical size: 256-512 tokens

    2. Embedding:
       - Convert chunks to vectors
       - Models: text-embedding-ada-002, BGE, etc.

    3. Vector Store:
       - Index embeddings for fast search
       - Options: FAISS, Pinecone, Weaviate, etc.

    4. Retrieval:
       - Embed query
       - Find k nearest neighbors
       - Return top-k documents

    5. Generation:
       - Combine query + retrieved docs
       - Generate answer with LLM

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Indexing (offline):                               │
    │  Docs → Chunk → Embed → Store in Vector DB         │
    │                                                     │
    │  Query (online):                                    │
    │  Query → Embed → Search → Top-k docs → LLM → Answer │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

---

## Part 8: LLM Agents

### From LLMs to Agents

```
    Agent Architecture
    ══════════════════

    Agent = LLM + Tools + Memory + Planning

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  User: "Book me a flight from NYC to LA next week" │
    │                                                     │
    │         ┌─────────────────────────────────┐        │
    │         │           Agent Core            │        │
    │         │      (LLM + Planning)           │        │
    │         └───────────────┬─────────────────┘        │
    │                         │                           │
    │    ┌────────────────────┼────────────────────┐     │
    │    ▼                    ▼                    ▼     │
    │ ┌──────┐          ┌──────────┐          ┌──────┐  │
    │ │Search│          │ Calendar │          │ Book │  │
    │ │Flights│         │  Check   │          │Flight│  │
    │ └──────┘          └──────────┘          └──────┘  │
    │                                                    │
    │  Think → Act → Observe → Think → Act → Done       │
    │                                                    │
    └─────────────────────────────────────────────────────┘
```

### Tool Use

```
    Function Calling / Tool Use
    ═══════════════════════════

    LLM generates structured tool calls:

    User: "What's the weather in Paris?"

    LLM output:
    {
      "tool": "get_weather",
      "arguments": {
        "location": "Paris, France"
      }
    }

    System executes tool:
    → {"temperature": 18, "condition": "cloudy"}

    LLM generates response:
    "The weather in Paris is 18°C and cloudy."
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **Tokenization** | BPE subword, ~30-100K vocab |
| **Pre-training** | Next token prediction on trillions of tokens |
| **Scaling Laws** | Loss predictably decreases with scale |
| **Fine-tuning** | Full, LoRA (low-rank), QLoRA (quantized) |
| **RLHF** | SFT → Reward Model → PPO (or DPO) |
| **Prompting** | Zero/few-shot, Chain-of-Thought |
| **RAG** | Retrieve context from vector DB |
| **Agents** | LLM + Tools + Memory + Planning |

---

## References

**Foundational:**
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3), 2020
- Hoffmann et al., "Training Compute-Optimal LLMs" (Chinchilla), 2022
- Touvron et al., "LLaMA", 2023

**Alignment:**
- Ouyang et al., "Training language models to follow instructions" (InstructGPT), 2022
- Rafailov et al., "Direct Preference Optimization", 2023

**Prompting:**
- Wei et al., "Chain-of-Thought Prompting", 2022
- Wang et al., "Self-Consistency Improves CoT", 2023

**RAG:**
- Lewis et al., "Retrieval-Augmented Generation", 2020
