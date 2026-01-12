# Week 11: Large Language Models - From Theory to Application

## Overview

This week bridges the gap between understanding transformers (Week 7) and building real LLM applications. We cover the complete LLM stack: how they're trained, how to adapt them, and how to use them effectively.

**Prerequisites**: Attention mechanisms (Week 7), basic deep learning concepts (Weeks 1-6)

**Learning Objectives**:
- Understand tokenization and why it matters
- Grasp pre-training objectives and scaling laws
- Master fine-tuning techniques (full, LoRA, RLHF)
- Learn prompting strategies and in-context learning
- Understand inference optimization techniques
- Build practical LLM applications

---

## Table of Contents

1. [Tokenization: How LLMs See Text](#1-tokenization-how-llms-see-text)
2. [Pre-training: Teaching Models to Understand Language](#2-pre-training-teaching-models-to-understand-language)
3. [Scaling Laws: Bigger Models, Better Results?](#3-scaling-laws-bigger-models-better-results)
4. [Fine-tuning: Adapting Models to Your Needs](#4-fine-tuning-adapting-models-to-your-needs)
5. [RLHF: Aligning Models with Human Preferences](#5-rlhf-aligning-models-with-human-preferences)
6. [Prompting: The Art of Asking](#6-prompting-the-art-of-asking)
7. [In-Context Learning: Teaching Without Training](#7-in-context-learning-teaching-without-training)
8. [Inference Optimization: Making LLMs Fast](#8-inference-optimization-making-llms-fast)
9. [Building LLM Applications](#9-building-llm-applications)
10. [Coding Exercises](#10-coding-exercises)
11. [Further Reading](#11-further-reading)

---

## 1. Tokenization: How LLMs See Text

### The Problem: Text to Numbers

Neural networks operate on numbers, not text. Tokenization converts text into numerical tokens that models can process.

```
Input:  "Hello, world!"
Output: [15496, 11, 995, 0]  # Token IDs
```

### Naive Approaches and Their Problems

**Character-level tokenization**:
```
"Hello" → ['H', 'e', 'l', 'l', 'o'] → [72, 101, 108, 108, 111]

Pros: Small vocabulary (~256 characters)
Cons: Very long sequences, hard to capture meaning
      "artificial" = 10 tokens, each meaningless alone
```

**Word-level tokenization**:
```
"Hello world" → ['Hello', 'world'] → [1234, 5678]

Pros: Meaningful units
Cons: Huge vocabulary (100K+ words), can't handle new words
      "ChatGPT" → [UNK] (unknown token)
```

### Byte Pair Encoding (BPE): The Sweet Spot

BPE finds a middle ground by learning common subword patterns:

```
Training corpus: "low", "lower", "lowest", "new", "newer", "newest"

Initial: Character-level
Step 1: Merge most frequent pair 'e'+'s' → 'es'
Step 2: Merge 'es'+'t' → 'est'
Step 3: Merge 'l'+'o' → 'lo'
Step 4: Merge 'lo'+'w' → 'low'
Step 5: Merge 'n'+'e' → 'ne'
Step 6: Merge 'ne'+'w' → 'new'
...

Result vocabulary: ['low', 'er', 'est', 'new', ...]
```

**BPE in action**:
```
"lowest"  → ['low', 'est']      # 2 tokens
"newer"   → ['new', 'er']       # 2 tokens
"newish"  → ['new', 'ish']      # Can handle unseen words!
"ChatGPT" → ['Chat', 'G', 'PT'] # No UNK tokens
```

### Why Tokenization Matters

**Token count affects everything**:
```
Context window: 128K tokens
Your document:  50K characters ≈ 12K-15K tokens (typical ratio ~3.5 chars/token)

Longer documents? Tokenize efficiently.
```

**Tokenization quirks cause real problems**:
```python
# GPT tokenizers struggle with:
" hello"  → different token than "hello"
"hello\n" → different than "hello"

# This affects prompting:
prompt_v1 = "Count the letters in 'hello'"
prompt_v2 = "Count the letters in ' hello'"  # Different behavior!

# Numbers are tokenized unexpectedly:
"123456" → ['123', '456']  # Not ['1', '2', '3', '4', '5', '6']
# This is why LLMs struggle with arithmetic!
```

**Language affects tokenization**:
```
English: "Hello"     → 1 token
Chinese: "你好"       → 2 tokens (each character)
Korean:  "안녕하세요"  → 5+ tokens

# Same meaning, different costs!
# Non-English text uses more tokens = higher API costs
```

### Modern Tokenizers

| Tokenizer | Used By | Vocabulary Size | Notes |
|-----------|---------|-----------------|-------|
| BPE | GPT-2, GPT-3 | 50,257 | Original GPT tokenizer |
| Tiktoken | GPT-4, Claude | 100K+ | Optimized BPE |
| SentencePiece | LLaMA, T5 | Varies | Handles any language |
| WordPiece | BERT | 30,522 | Similar to BPE |

### Practical: Counting Tokens

```python
# Using tiktoken (OpenAI's tokenizer)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

text = "Hello, how are you today?"
tokens = enc.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded: {[enc.decode([t]) for t in tokens]}")

# Output:
# Text: Hello, how are you today?
# Tokens: [9906, 11, 1268, 527, 499, 3432, 30]
# Token count: 7
# Decoded: ['Hello', ',', ' how', ' are', ' you', ' today', '?']
```

---

## 2. Pre-training: Teaching Models to Understand Language

### The Core Idea: Next Token Prediction

LLMs learn by predicting the next token given previous tokens:

```
Input:  "The cat sat on the"
Target: "mat"

Input:  "The cat sat on the mat"
Target: "."

Repeat billions of times across massive datasets.
```

### Why This Works

**Implicit learning**: To predict the next word, the model must learn:
- Grammar ("The cat sat" → verb next unlikely)
- Facts ("Paris is the capital of" → "France")
- Reasoning ("2 + 2 =" → "4")
- Style ("Once upon a time" → fairy tale language)

```
Training objective: Minimize cross-entropy loss

Loss = -log P(correct_token | previous_tokens)

Lower loss = better predictions = more useful model
```

### Pre-training Data

**Scale matters enormously**:
```
GPT-2 (2019):  40GB of text
GPT-3 (2020):  570GB of text
GPT-4 (2023):  ~13T tokens (estimated)
LLaMA (2023):  1.4T tokens

Sources:
- Web crawls (Common Crawl)
- Books
- Wikipedia
- Code (GitHub)
- Scientific papers
- Curated datasets
```

**Data quality matters more**:
```
Raw web crawl: Lots of garbage, spam, duplicates
Filtered data: 10x smaller but much higher quality

The Pile (800GB) → Cleaned and deduplicated → Better than 8TB raw
```

### Training Dynamics

**Typical pre-training setup**:
```
Model size:     7B - 70B parameters
Sequence length: 2048 - 8192 tokens
Batch size:     1M - 4M tokens
Training steps: 100K - 1M steps
Hardware:       1000s of GPUs/TPUs
Duration:       Weeks to months
Cost:           $1M - $100M+
```

**Loss curves**:
```
Loss
│
4.0├──╮
│    ╰──╮
3.0├      ╰──╮
│          ╰──╮
2.0├            ╰────╮
│                  ╰────────────
1.0├                           (diminishing returns)
│
└────────────────────────────────────
  0     100K    200K    300K   Steps

Key observation: Loss drops quickly, then slows
                 But capabilities keep emerging!
```

### Emergent Capabilities

Some abilities only appear at scale:

```
Model Size    Capabilities
─────────────────────────────────────────────────────
1B params     Basic text completion, simple patterns
7B params     Coherent paragraphs, basic reasoning
13B params    Multi-step reasoning, code generation
70B params    Complex reasoning, following instructions
175B+ params  In-context learning, chain-of-thought
```

**Emergence is discontinuous**:
```
Arithmetic accuracy by model size:

      │    ┌───────────────
100%  │    │
      │    │
 50%  │   ╱
      │  ╱
  0%  ├─╯
      └────────────────────
        1B  10B  100B  Size

Not gradual improvement—sudden jumps!
```

---

## 3. Scaling Laws: Bigger Models, Better Results?

### The Scaling Laws Discovery

Researchers found predictable relationships between:
- **N**: Number of parameters
- **D**: Dataset size (tokens)
- **C**: Compute (FLOPs)
- **L**: Loss (performance)

```
Loss ∝ N^(-0.076) + D^(-0.095) + constant

Key insight: Performance improves predictably with scale
            You can estimate required resources!
```

### Chinchilla Scaling Laws (2022)

**The big finding**: Most models were undertrained!

```
GPT-3: 175B parameters, 300B tokens
       → Too many parameters for the data

Chinchilla: 70B parameters, 1.4T tokens
           → Same compute, better performance!

Optimal ratio: ~20 tokens per parameter
```

**Implications**:
```
Before Chinchilla: Make models bigger
After Chinchilla:  Balance model size and data

For 10B parameter model:
  Optimal training: ~200B tokens
  Not: 10B tokens (undertrained)
  Not: 2T tokens (diminishing returns)
```

### Compute-Optimal Training

```
Given fixed compute budget C:

Option A: Larger model, less data
Option B: Smaller model, more data  ← Usually better!

Why? Training is one-time cost
     Inference happens millions of times
     Smaller model = cheaper inference
```

### Practical Implications

**For practitioners**:
```
1. Don't assume bigger = better for your use case
   - 7B model fine-tuned > 70B model prompted (sometimes)

2. Check if model was trained compute-optimally
   - LLaMA: Yes (lots of data per parameter)
   - GPT-3: No (could have been smaller)

3. Consider inference costs
   - 70B model: ~4x slower than 13B
   - Monthly API costs scale linearly
```

**Scaling law limitations**:
```
What scaling laws predict:
  ✓ Average benchmark performance
  ✓ Perplexity on held-out data

What they don't predict:
  ✗ Specific capability emergence
  ✗ Reasoning breakthroughs
  ✗ Safety/alignment properties
```

---

## 4. Fine-tuning: Adapting Models to Your Needs

### Why Fine-tune?

Pre-trained models are general. Fine-tuning makes them specific:

```
Pre-trained GPT: Jack of all trades
Fine-tuned GPT:  Expert at your specific task

Examples:
- Customer support bot (trained on your tickets)
- Medical assistant (trained on clinical notes)
- Code assistant (trained on your codebase)
```

### Full Fine-tuning

Update all model parameters on your data:

```python
# Conceptual fine-tuning loop
for batch in your_dataset:
    outputs = model(batch.inputs)
    loss = compute_loss(outputs, batch.targets)
    loss.backward()  # Gradients for ALL parameters
    optimizer.step()
```

**Pros and cons**:
```
✓ Maximum flexibility
✓ Best performance (given enough data)

✗ Requires full model in memory (70B × 4 bytes = 280GB)
✗ Risk of catastrophic forgetting
✗ Need lots of data to avoid overfitting
```

### Parameter-Efficient Fine-tuning (PEFT)

**Key insight**: You don't need to update all parameters!

**LoRA (Low-Rank Adaptation)**:
```
Original weight matrix W: [4096 × 4096]
                         = 16M parameters

LoRA approach:
  - Freeze W
  - Add low-rank matrices A and B
  - A: [4096 × 16]  = 65K parameters
  - B: [16 × 4096]  = 65K parameters

Output = W·x + (A·B)·x

Total trainable: 130K vs 16M (0.8%!)
```

**Visual**:
```
        ┌───────────────┐
   x ───┤  W (frozen)   ├─── +  ──→ output
        └───────────────┘    │
              │              │
              │    ┌───┐  ┌───┐
              └────┤ A ├──┤ B ├──┘
                   └───┘  └───┘
                   (trainable)
```

**QLoRA**: LoRA + quantization
```
1. Quantize base model to 4-bit (70B → 35GB)
2. Add LoRA adapters in fp16
3. Train only the adapters

Result: Fine-tune 70B model on single 24GB GPU!
```

### Fine-tuning Best Practices

**Data quality over quantity**:
```
Good: 1,000 high-quality examples
Bad:  100,000 noisy examples

Quality checklist:
□ Correct outputs
□ Diverse inputs
□ Representative of real usage
□ No contradictions
```

**Hyperparameters**:
```python
# Typical fine-tuning config
learning_rate = 2e-5      # Much lower than pre-training
epochs = 3                # Don't overtrain!
batch_size = 8-32         # As large as memory allows
warmup_ratio = 0.1        # Gradual LR increase
weight_decay = 0.01       # Prevent overfitting
```

**Preventing catastrophic forgetting**:
```
Problem: Fine-tuning erases pre-trained knowledge

Solutions:
1. Low learning rate (2e-5 not 2e-3)
2. Few epochs (3 not 30)
3. Mix in pre-training data
4. Use LoRA (preserves original weights)
```

### When to Fine-tune vs. Prompt

```
Use Fine-tuning When:           Use Prompting When:
─────────────────────────       ──────────────────────
Need consistent format          Testing ideas quickly
Have lots of examples           Few examples available
Latency critical               Flexibility important
Domain very specific           Task is general
Budget for training            Minimize upfront cost
```

---

## 5. RLHF: Aligning Models with Human Preferences

### The Alignment Problem

Pre-trained models predict likely text, not helpful text:

```
User: "How do I hack into my neighbor's WiFi?"

Base model: [Provides detailed instructions]
           → Technically correct prediction!
           → But not what we want

Aligned model: "I can't help with that. Here's how to
               secure your own network instead..."
```

### RLHF Pipeline

**Step 1: Supervised Fine-tuning (SFT)**
```
Train on human-written helpful responses:

Input:  "What's the capital of France?"
Output: "The capital of France is Paris."

This gives the model the right "format" for responses.
```

**Step 2: Reward Model Training**
```
Collect human preferences:

Response A: "Paris is the capital of France. It's known
            for the Eiffel Tower..."

Response B: "The capital is Paris."

Human label: A > B (more helpful)

Train reward model to predict: R(A) > R(B)
```

**Step 3: RL Fine-tuning (PPO)**
```
                    ┌─────────────────┐
                    │  Reward Model   │
                    │   R(response)   │
                    └────────┬────────┘
                             │ reward signal
                             ▼
┌─────────┐    ┌─────────────────────┐    ┌─────────┐
│ Prompt  │───▶│    Policy Model     │───▶│Response │
│         │    │ (model being tuned) │    │         │
└─────────┘    └─────────────────────┘    └─────────┘
                             │
                             ▼
                    Update to maximize
                    reward while staying
                    close to original model
```

### Why RLHF Works

**Captures implicit preferences**:
```
Hard to specify in rules:
- "Be helpful but not harmful"
- "Be confident but acknowledge uncertainty"
- "Be concise but thorough"

Easy to demonstrate:
- "I prefer response A over B"
- Repeated thousands of times
- Reward model learns the pattern
```

**The KL penalty trick**:
```
Objective = Reward - β × KL(policy || reference)

Without KL penalty: Model finds reward hacks
  - Excessively verbose (humans prefer longer?)
  - Sycophantic ("Great question!")
  - Repetitive patterns

With KL penalty: Stays close to useful base model
```

### RLHF Alternatives

**DPO (Direct Preference Optimization)**:
```
Skip the reward model entirely!

Traditional RLHF: preference → reward model → RL → policy
DPO:              preference → directly → policy

Same result, simpler training, more stable.
```

**Constitutional AI (CAI)**:
```
Use AI to generate preferences:

1. Model generates response
2. Model critiques own response against principles
3. Model revises response
4. Use (original, revised) pairs for training

Scale: Can generate unlimited training data
```

### Practical Considerations

**RLHF is expensive**:
```
You need:
- Human labelers (expensive, slow)
- Training pipeline (RL is tricky)
- Multiple models (reward + policy)

Alternative for most practitioners:
- Use already-aligned models (Claude, GPT-4)
- Fine-tune with high-quality SFT data
- DPO if you have preference data
```

---

## 6. Prompting: The Art of Asking

### Why Prompting Matters

The same model gives vastly different outputs based on how you ask:

```
Bad prompt:  "Write about dogs"
→ Generic, unfocused output

Good prompt: "Write a 200-word blog post about the health
             benefits of daily walks for senior dogs,
             targeting pet owners over 50."
→ Specific, useful output
```

### Prompt Engineering Principles

**1. Be specific**:
```
Vague: "Summarize this"
Specific: "Summarize in 3 bullet points, focusing on
          action items for the engineering team"
```

**2. Provide context**:
```
Without context:
"Is this code good?"

With context:
"You are a senior Python developer reviewing code for
a production web service. Focus on security and performance.
Is this code good?"
```

**3. Specify format**:
```
"Return your analysis as JSON:
{
  'sentiment': 'positive' | 'negative' | 'neutral',
  'confidence': 0.0-1.0,
  'key_phrases': ['phrase1', 'phrase2']
}"
```

**4. Give examples** (few-shot):
```
"Classify the sentiment:

Text: 'Great product, fast shipping!'
Sentiment: positive

Text: 'Broke after one day'
Sentiment: negative

Text: 'Arrived on time'
Sentiment: neutral

Text: 'Not bad but expensive'
Sentiment: "
```

### Advanced Prompting Techniques

**Chain-of-Thought (CoT)**:
```
Without CoT:
Q: "A bat and ball cost $1.10. The bat costs $1 more
    than the ball. How much does the ball cost?"
A: "$0.10"  ← Wrong!

With CoT:
Q: [same question]
A: "Let me think step by step.
    Let ball = x
    Bat = x + $1
    Total: x + (x + 1) = $1.10
    2x = $0.10
    x = $0.05

    The ball costs $0.05"  ← Correct!
```

**Just add "Let's think step by step"**:
```
"[Your question]

Let's think step by step."

This simple addition improves reasoning significantly!
```

**Self-consistency**:
```
1. Generate multiple responses (temperature > 0)
2. Extract the final answer from each
3. Take majority vote

Response 1: "...therefore 42"
Response 2: "...the answer is 42"
Response 3: "...which gives us 38"
Response 4: "...equals 42"
Response 5: "...42"

Majority: 42 (4/5 votes)
```

**Tree of Thoughts**:
```
Problem: Solve complex planning task

        Start
       /  |  \
      A   B   C       ← Generate options
     /|   |   |\
    A1 A2 B1  C1 C2   ← Explore each
    ✗  ✓  ✗   ✓  ✗    ← Evaluate
       |      |
      ...    ...      ← Continue promising paths
```

### System Prompts

Set the overall behavior and persona:

```
System: "You are a helpful coding assistant. You write
        clean, well-documented Python code. When asked
        to write code, always include error handling
        and type hints. If a request is unclear, ask
        clarifying questions before writing code."

User: "Write a function to fetch data from an API"
```

---

## 7. In-Context Learning: Teaching Without Training

### What is In-Context Learning?

LLMs learn from examples in the prompt—no gradient updates required:

```
Traditional ML: Train on examples → Update weights → Apply
In-Context:     Show examples in prompt → Model adapts on the fly

No training! The model "learns" just by seeing examples in context.
```

### Zero-shot, One-shot, Few-shot

**Zero-shot**: No examples, just instructions
```
Prompt: "Translate 'Hello' to French"
Output: "Bonjour"
```

**One-shot**: One example
```
Prompt: "English to French:
        Hello -> Bonjour
        Goodbye -> "
Output: "Au revoir"
```

**Few-shot**: Multiple examples
```
Prompt: "Classify sentiment:
        'Great product!' -> positive
        'Terrible service' -> negative
        'It was okay' -> neutral
        'Loved it!' -> "
Output: "positive"
```

### Why Few-shot Works

**Pattern recognition at inference time**:
```
The model recognizes:
1. The format (input -> output)
2. The task (classification, translation, etc.)
3. The expected output type

Then applies this pattern to new inputs—all in the forward pass!
```

**Activation steering**:
```
Examples activate relevant "circuits" in the model

Without examples: Many possible interpretations
With examples:    Narrows to specific task

Like priming—the context sets up the computation.
```

### Few-shot Best Practices

**Example selection matters**:
```
Bad: All similar examples
  "I love it!" -> positive
  "Amazing!" -> positive
  "So good!" -> positive

Good: Diverse, balanced examples
  "I love it!" -> positive
  "Terrible waste of money" -> negative
  "It arrived on time" -> neutral
```

**Example order matters**:
```
Recency bias: Later examples often weighted more heavily

Strategy: Put most representative examples last
         Or randomize order across runs
```

**Format consistency**:
```
Bad:
  "Great!" is positive
  negative: "Terrible"
  okay -> neutral

Good:
  "Great!" -> positive
  "Terrible" -> negative
  "Okay" -> neutral
```

### Limitations

```
Context window limit: Can only fit so many examples
                     (but modern models have 128K+ tokens)

Not true learning: Model doesn't retain between sessions
                  Must include examples every time

Prompt sensitivity: Small changes can break performance
                   Requires careful engineering
```

---

## 8. Inference Optimization: Making LLMs Fast

### The Inference Challenge

LLMs are slow and expensive:
```
70B parameter model:
- Memory: ~140GB (fp16)
- Single token: ~1-2 TFLOPs
- Latency: Tens to hundreds of ms per token
- Cost: $0.01-0.10 per 1K tokens (API)

Optimization goal: Same quality, less cost/latency
```

### KV Cache (Refresher from Week 7)

Store computed attention keys and values:
```
Without KV cache:
  Token 1: Compute K,V for [1]
  Token 2: Compute K,V for [1,2]
  Token 3: Compute K,V for [1,2,3]
  ...
  Total: O(n³) attention computations

With KV cache:
  Token 1: Compute K,V for [1], store
  Token 2: Compute K,V for [2], reuse [1]
  Token 3: Compute K,V for [3], reuse [1,2]
  ...
  Total: O(n²) attention computations
```

### Quantization

Reduce precision to save memory and compute:
```
Full precision (fp32):  32 bits per parameter
Half precision (fp16):  16 bits per parameter
8-bit (int8):           8 bits per parameter
4-bit (int4):           4 bits per parameter

70B model memory:
  fp32: 280 GB
  fp16: 140 GB
  int8:  70 GB
  int4:  35 GB  ← Fits on single high-end GPU!
```

**Quantization methods**:
```
Post-training quantization (PTQ):
  - Quantize after training
  - Fast, easy
  - Some quality loss

Quantization-aware training (QAT):
  - Train with quantization in mind
  - Better quality
  - More expensive

GPTQ, AWQ, GGUF:
  - Modern PTQ methods
  - Minimal quality loss
  - Widely supported
```

### Batching and Throughput

**Static batching**: Wait for batch to fill
```
Request 1 arrives at t=0  ──┐
Request 2 arrives at t=50  ─┼─→ Process batch at t=100
Request 3 arrives at t=100 ─┘

Problem: Request 1 waits 100ms unnecessarily
```

**Continuous batching**: Dynamic insertion
```
Request 1 arrives, start immediately
Request 2 arrives, join ongoing batch
Request 3 arrives, join ongoing batch

Better latency for all requests!
```

**vLLM's PagedAttention**:
```
Problem: KV cache wastes memory on padding

Traditional:
  [Seq 1: 50 tokens][padding to 512]
  [Seq 2: 300 tokens][padding to 512]
  = 1024 slots allocated, 350 used

PagedAttention:
  [Block 1: 50 tokens][Block 2: 50 tokens]...
  Allocate only what's needed
  = 40-60% memory savings
```

### Speculative Decoding

Use a small model to propose, large model to verify:
```
         ┌─────────────────┐
         │  Draft Model    │──→ Propose: "The quick brown"
         │  (7B, fast)     │
         └─────────────────┘
                  ↓
         ┌─────────────────┐
         │  Target Model   │──→ Verify: "The quick brown"
         │  (70B, slow)    │    Accept 2/3, reject "brown"
         └─────────────────┘    Replace with "red"
                  ↓
         Output: "The quick red"

Result: Multiple tokens per forward pass of large model
        2-3x speedup possible
```

### Serving Frameworks

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| vLLM | PagedAttention, high throughput | Production serving |
| TensorRT-LLM | NVIDIA optimizations | Maximum speed on NVIDIA |
| llama.cpp | CPU inference, quantization | Local/edge deployment |
| Hugging Face TGI | Easy setup, broad support | Quick deployment |

---

## 9. Building LLM Applications

### Architecture Patterns

**Simple: Direct API Call**
```
User Input → Prompt Template → LLM API → Response

Good for: Chatbots, simple Q&A, writing assistance
```

**RAG (Retrieval Augmented Generation)**
```
User Query
    ↓
┌──────────────────┐
│ Embedding Model  │
└────────┬─────────┘
         ↓
┌──────────────────┐     ┌──────────────────┐
│ Vector Database  │────→│ Retrieved Docs   │
└──────────────────┘     └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ LLM with context │
                         └────────┬─────────┘
                                  ↓
                              Response

Good for: Knowledge-grounded answers, reducing hallucination
```

**Agents: LLM as Controller**
```
User Task
    ↓
┌──────────────────────────────────────┐
│              LLM Agent               │
│  ┌─────────────────────────────────┐ │
│  │ Think: What tool do I need?    │ │
│  │ Act: Call tool with parameters │ │
│  │ Observe: Process tool result   │ │
│  │ Repeat until task complete     │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
         ↓           ↓           ↓
    ┌────────┐  ┌────────┐  ┌────────┐
    │Search  │  │ Code   │  │  API   │
    │ Tool   │  │ Exec   │  │ Calls  │
    └────────┘  └────────┘  └────────┘

Good for: Complex multi-step tasks, tool use
```

### RAG Deep Dive

**Chunking strategies**:
```
Fixed-size chunks:
  Split every N tokens
  Simple but may break mid-sentence

Semantic chunks:
  Split at paragraph/section boundaries
  Preserves meaning better

Overlapping chunks:
  Chunks overlap by M tokens
  Helps with retrieval at boundaries
```

**Retrieval quality**:
```
Embedding models:
  - OpenAI ada-002: Good general purpose
  - Cohere embed: Multilingual
  - BGE, E5: Open source alternatives

Reranking:
  1. Retrieve top-100 with embeddings (fast)
  2. Rerank to top-5 with cross-encoder (accurate)

Hybrid search:
  - Combine vector similarity + keyword matching
  - Better for exact terms (names, codes)
```

### Agent Frameworks

**ReAct pattern**:
```
Thought: I need to find the current stock price of AAPL
Action: stock_price_tool(symbol="AAPL")
Observation: $178.50
Thought: Now I need to calculate the market cap
Action: calculator(expression="178.50 * 15.7e9")
Observation: 2,802,450,000,000
Thought: I have the answer
Answer: Apple's market cap is approximately $2.8 trillion
```

**Tool definition**:
```json
{
  "name": "web_search",
  "description": "Search the web for current information",
  "parameters": {
    "query": {
      "type": "string",
      "description": "The search query"
    }
  }
}
```

### Error Handling and Guardrails

**LLM outputs are unpredictable**:
```python
# Always validate LLM outputs
def parse_json_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Retry with different prompt
        # Or fall back to regex extraction
        return extract_with_regex(response)
```

**Guardrails**:
```
Input guardrails:
  - Filter harmful prompts
  - Detect prompt injection
  - Validate input format

Output guardrails:
  - Check for refusals
  - Validate against schema
  - Filter harmful content
  - Fact-check critical claims
```

### Cost Optimization

**Caching**:
```
Semantic cache:
  - Embed queries
  - Return cached responses for similar queries
  - Huge savings on repeated patterns

Prompt cache:
  - Reuse computed prefixes
  - Effective for long system prompts
```

**Model routing**:
```
Simple queries  → Small model (Claude Haiku, GPT-4o-mini)
Complex queries → Large model (Claude Opus, GPT-4)

Savings: 10-50x on API costs
Implementation: Classify query complexity first
```

**Batching requests**:
```
# Instead of individual calls
for item in items:
    result = llm.complete(item)  # N API calls

# Batch when possible
results = llm.complete_batch(items)  # 1 API call
```

---

## 10. Coding Exercises

### Exercise 1: Tokenization Explorer

Build a tokenizer visualization tool:
```python
"""
Create a program that:
1. Takes input text
2. Tokenizes it with different tokenizers
3. Shows token boundaries visually
4. Compares token counts across tokenizers
"""

# Example output:
# Text: "Hello, world! 你好"
#
# GPT-4 (tiktoken):
#   |Hello|,| world|!| 你|好|
#   Token count: 6
#
# LLaMA (sentencepiece):
#   |▁Hello|,|▁world|!|▁|你|好|
#   Token count: 7
```

### Exercise 2: Few-shot Classifier

Implement a few-shot text classifier:
```python
"""
Create a classifier that:
1. Takes example (text, label) pairs
2. Formats them as a prompt
3. Classifies new texts using an LLM API
4. Handles edge cases (uncertain, out-of-distribution)
"""

classifier = FewShotClassifier(
    examples=[
        ("Great product!", "positive"),
        ("Terrible experience", "negative"),
        ("It works", "neutral")
    ]
)

result = classifier.classify("I love this!")
# result: {"label": "positive", "confidence": 0.95}
```

### Exercise 3: Simple RAG System

Build a basic RAG pipeline:
```python
"""
Create a RAG system that:
1. Chunks documents
2. Embeds chunks
3. Stores in a vector database
4. Retrieves relevant chunks for queries
5. Generates answers with context
"""

rag = SimpleRAG()
rag.add_documents([
    "Python is a programming language...",
    "Machine learning uses algorithms..."
])

answer = rag.query("What is Python used for?")
# answer: "Python is used for programming..."
```

### Exercise 4: LoRA Implementation

Implement LoRA from scratch:
```python
"""
Create a LoRA layer that:
1. Wraps an existing linear layer
2. Adds low-rank A and B matrices
3. Computes output as W*x + (A*B)*x
4. Allows freezing original weights
"""

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8):
        # Your implementation here
        pass
```

### Exercise 5: Agent Loop

Build a simple agent:
```python
"""
Create an agent that:
1. Parses a task description
2. Decides which tool to use
3. Calls the tool
4. Processes the result
5. Repeats until task is complete
"""

tools = {
    "calculator": lambda expr: eval(expr),
    "search": lambda q: web_search(q),
    "write_file": lambda path, content: Path(path).write_text(content)
}

agent = SimpleAgent(tools)
result = agent.run("Calculate 15% tip on $45.80 and save to tips.txt")
```

---

## 11. Further Reading

### Papers

**Foundational**:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (GPT-3 paper, 2020)
- "Training Language Models to Follow Instructions" (InstructGPT, 2022)

**Scaling and Training**:
- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
- "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)

**Efficiency**:
- "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

**Alignment**:
- "Constitutional AI" (Anthropic, 2022)
- "Direct Preference Optimization" (2023)

**Applications**:
- "Retrieval-Augmented Generation" (RAG paper, 2020)
- "ReAct: Reasoning and Acting in Language Models" (2022)

### Books and Courses

- "Natural Language Processing with Transformers" (Hugging Face team)
- Stanford CS224N: NLP with Deep Learning
- DeepLearning.AI courses on LLMs

### Tools and Libraries

**Inference**:
- vLLM: High-throughput serving
- llama.cpp: CPU inference
- Hugging Face Transformers

**Fine-tuning**:
- Hugging Face PEFT
- Axolotl (LoRA training)
- OpenAI fine-tuning API

**Applications**:
- LangChain: Agent and RAG framework
- LlamaIndex: Data framework for LLMs
- Instructor: Structured outputs

---

## Summary

This week covered the complete LLM stack:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM APPLICATION                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Prompting, RAG, Agents, Tool Use                       ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    OPTIMIZATION                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ KV Cache, Quantization, Batching, Speculative Decode   ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    ADAPTATION                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Fine-tuning (Full, LoRA), RLHF, In-Context Learning    ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    FOUNDATION                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Tokenization, Pre-training, Scaling Laws               ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Key takeaways**:

1. **Tokenization** determines how LLMs see text—understanding it explains many quirks

2. **Pre-training** on next-token prediction creates general-purpose models

3. **Scaling laws** help predict resources needed for desired performance

4. **Fine-tuning** adapts models efficiently; LoRA makes it accessible

5. **RLHF** aligns models with human preferences; DPO simplifies it

6. **Prompting** is an art—chain-of-thought and few-shot dramatically improve results

7. **In-context learning** lets models adapt without training

8. **Optimization** (KV cache, quantization, batching) makes deployment practical

9. **Applications** combine these techniques: RAG for knowledge, agents for action

**What's Next**: Apply these concepts to build real applications. See the companion documents for case studies:
- `DL_AND_LLM_IN_PRODUCTION.md` - Hybrid DL+LLM architectures
- `DEEP_LEARNING_TO_LLM_APPLICATIONS.md` - When to use which approach
- `STYLEMATCH_PART_2_THE_LLM_ERA.md` - Strategic LLM adoption story