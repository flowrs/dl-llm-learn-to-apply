# Module 6: Large Language Models

## Learning Objectives

By the end of this module, you will understand:
- What makes a language model "large" and why scale matters
- Tokenization methods and vocabulary construction (BPE, WordPiece, SentencePiece)
- Pre-training objectives, data requirements, and scaling laws
- Fine-tuning techniques (Full, LoRA, QLoRA) and when to use each
- RLHF pipeline and alignment methods (PPO, DPO)
- Prompting strategies and in-context learning mechanics
- Inference optimization techniques for production deployment
- Hallucination causes and mitigation strategies

---

## 6.1 What Makes a Language Model "Large"?

### The Evolution of Language Models

Language models have existed for decades, but "Large Language Models" (LLMs) represent a
qualitative shift in capabilities that emerged from scaling up three key dimensions:

```
THE THREE PILLARS OF LLMs
═════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐              │
│   │                 │   │                 │   │                 │              │
│   │   PARAMETERS    │   │     DATA        │   │    COMPUTE      │              │
│   │                 │   │                 │   │                 │              │
│   │  Billions of    │   │  Trillions of   │   │  Thousands of   │              │
│   │  weights in     │   │  tokens from    │   │  GPUs training  │              │
│   │  the network    │   │  the internet   │   │  for months     │              │
│   │                 │   │                 │   │                 │              │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘              │
│            │                     │                     │                        │
│            └─────────────────────┼─────────────────────┘                        │
│                                  │                                              │
│                                  ▼                                              │
│                    ┌─────────────────────────┐                                  │
│                    │                         │                                  │
│                    │   EMERGENT CAPABILITIES │                                  │
│                    │                         │                                  │
│                    │   • Few-shot learning   │                                  │
│                    │   • Reasoning           │                                  │
│                    │   • Code generation     │                                  │
│                    │   • Following instruct. │                                  │
│                    │                         │                                  │
│                    └─────────────────────────┘                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The LLM Recipe

At its core, an LLM is simple: it's a Transformer decoder trained to predict the next token.
What makes it "large" is the scale at which this simple idea is executed:

```
LLM = Transformer Decoder + Massive Data + Massive Compute + Time
    = Simple Architecture  × Extreme Scale  × Careful Engineering
```

### Historical Scale Comparison

```
EVOLUTION OF MODEL SCALE
════════════════════════

                    Parameters        Training Tokens     Approx Cost
                    ──────────        ───────────────     ───────────
GPT-1 (2018)        117M              ~5B                 ~$10K
        │
        ▼
GPT-2 (2019)        1.5B              40B                 ~$50K
        │           (13× larger)      (8× more)
        ▼
GPT-3 (2020)        175B              300B                ~$5M
        │           (117× larger)     (7.5× more)
        ▼
Chinchilla (2022)   70B               1.4T                ~$3M
        │           (optimally       (4.7× more)
        │            scaled)
        ▼
Llama 2 (2023)      70B               2T                  ~$2M
        │                             (1.4× more)
        ▼
GPT-4 (2023)        ~1.8T (est.)      ~13T (est.)         ~$100M (est.)
                    (26× larger)      (6.5× more)

VISUAL SCALE COMPARISON (log scale):
────────────────────────────────────

Parameters:
GPT-1   ▓                                               117M
GPT-2   ▓▓                                              1.5B
GPT-3   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                  175B
Llama 2 ▓▓▓▓▓▓▓▓▓▓                                      70B
GPT-4   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      ~1.8T

Training Tokens:
GPT-1   ▓                                               ~5B
GPT-2   ▓▓▓                                             40B
GPT-3   ▓▓▓▓▓▓▓▓▓▓                                      300B
Llama 2 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      2T
GPT-4   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ~13T
```

### Emergent Capabilities

One of the most fascinating aspects of LLMs is **emergent capabilities** - abilities that
appear suddenly at certain scales and weren't explicitly programmed:

```
EMERGENT CAPABILITIES BY SCALE
══════════════════════════════

Capability              Emerges Around       What It Looks Like
───────────             ──────────────       ──────────────────

Basic Fluency           ~100M params         Grammatical sentences
                        │
Context Following       ~1B params           Maintains topic coherence
                        │
Few-Shot Learning       ~10B params          Learns from examples in prompt
                        │
Chain-of-Thought        ~100B params         Step-by-step reasoning
                        │
Complex Reasoning       ~500B+ params        Multi-step problem solving

VISUALIZATION:
─────────────

Performance on Complex Tasks
     │
   1 │                                              ┌────────
     │                                         ____/
     │                                    ____/
     │                               ____/
   0 │──────────────────────────────/
     └────────┬─────────┬──────────┬─────────┬────────→
             10M      100M        10B      100B     1T
                      Model Size (parameters)

                    ↑
                    │
              "Phase transition" - sudden emergence
              of capability at critical scale
```

### Why Scale Works: The Lottery Ticket Hypothesis Perspective

One theory for why scale helps: larger networks contain more "lottery ticket" subnetworks
that happen to solve the task well:

```
LOTTERY TICKET INTUITION
════════════════════════

Small Network (100M params):
┌──────────────────────────┐
│  ○ ○ ● ○ ○ ○ ○ ○ ○ ○    │   ● = "winning ticket" subnetwork
│  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○    │       for a specific task
│  ○ ○ ○ ○ ● ○ ○ ○ ○ ○    │
└──────────────────────────┘
Only 2 winning tickets for task A → Limited capability


Large Network (100B params):
┌──────────────────────────────────────────────────────┐
│  ○ ● ○ ● ○ ○ ○ ● ○ ○ ● ○ ○ ○ ● ○ ○ ○ ● ○ ○ ○ ● ○  │
│  ○ ○ ● ○ ○ ● ○ ○ ○ ● ○ ○ ● ○ ○ ○ ● ○ ○ ● ○ ○ ○ ●  │
│  ● ○ ○ ○ ● ○ ○ ○ ● ○ ○ ● ○ ○ ● ○ ○ ○ ● ○ ○ ○ ● ○  │
│  ○ ● ○ ○ ○ ● ○ ● ○ ○ ○ ● ○ ● ○ ○ ● ○ ○ ● ○ ○ ○ ●  │
│  ○ ○ ● ○ ○ ○ ● ○ ○ ● ○ ○ ○ ● ○ ○ ○ ● ○ ○ ○ ● ○ ○  │
└──────────────────────────────────────────────────────┘
Many winning tickets for tasks A, B, C, D, ... → General capability
```

---

## 6.2 Tokenization

### The Fundamental Problem

Neural networks operate on numbers, but language consists of text. Tokenization is the
bridge between these two worlds:

```
TOKENIZATION: TEXT TO NUMBERS
═════════════════════════════

"Hello, how are you?"
        │
        ▼
   [Tokenizer]
        │
        ▼
["Hello", ",", " how", " are", " you", "?"]    ← Tokens (strings)
        │
        ▼
   [15496,   11,   703,   389,   345,  30]     ← Token IDs (integers)
        │
        ▼
   [Embedding Layer]
        │
        ▼
   [[0.12, -0.34, ...],                        ← Embeddings (vectors)
    [0.56, 0.78, ...],                            d_model dimensions each
    [0.23, -0.12, ...],
    ...]
```

### Why Not Character-Level?

You might wonder: why not just use individual characters?

```
CHARACTER-LEVEL ISSUES
══════════════════════

Text: "The cat sat on the mat"

Character-level tokens: ['T', 'h', 'e', ' ', 'c', 'a', 't', ' ', 's', ...]
                        22 characters = 22 tokens

Word-level tokens:      ['The', 'cat', 'sat', 'on', 'the', 'mat']
                        6 tokens

Subword tokens:         ['The', ' cat', ' sat', ' on', ' the', ' mat']
                        6 tokens

PROBLEM WITH CHARACTERS:
────────────────────────
• Sequences become very long (more computation, harder to learn)
• Model must learn spelling from scratch
• Context window fills up quickly

Example: 4K context window
  - Character-level: ~4,000 characters = ~800 words
  - Subword-level:   ~4,000 tokens = ~3,000 words
                     (3.75× more content!)
```

### Why Not Word-Level?

Word-level tokenization has its own problems:

```
WORD-LEVEL ISSUES
═════════════════

Problem 1: Vocabulary Explosion
────────────────────────────────
English has ~170,000 words in current use
+ Technical terms, names, neologisms
+ Misspellings, slang, code
= Millions of potential tokens

Each token needs its own embedding vector (d_model × 1)
With d_model = 4096 and 1M vocabulary:
  Memory = 1M × 4096 × 4 bytes = 16 GB just for embeddings!


Problem 2: Out-of-Vocabulary (OOV)
──────────────────────────────────
Training vocabulary: ["cat", "dog", "house", ...]

New word encountered: "TikToker"

Word-level: Maps to <UNK> (unknown) → loses all meaning!
Subword:    ["Tik", "Tok", "er"] → preserves some meaning


Problem 3: No Morphological Sharing
───────────────────────────────────
"run", "runs", "running", "runner" → 4 separate tokens
Model must learn they're related from context alone

With subwords: ["run"], ["run", "s"], ["run", "ning"], ["run", "ner"]
               Shared "run" token captures common meaning
```

### Subword Tokenization: The Sweet Spot

Modern LLMs use **subword tokenization** which balances:
- Reasonable vocabulary size (~30K-100K tokens)
- Ability to represent any text (no OOV)
- Semantic chunking (common words = single tokens)

```
SUBWORD TOKENIZATION SPECTRUM
═════════════════════════════

             Characters    ←── Subwords ──→    Words
                │                                │
Vocabulary      26-256         30K-100K        100K-1M+
Size
                │                                │
Sequence        Very Long      Moderate         Short
Length
                │                                │
OOV             None           None             Common
Problems
                │                                │
Semantic        Poor           Good             Best
Chunking

                        ◄─────────────►
                          SWEET SPOT
                        BPE, WordPiece,
                         SentencePiece
```

### Byte-Pair Encoding (BPE)

BPE is the most common subword tokenization algorithm. It was originally a compression
algorithm, adapted for NLP by Sennrich et al. (2016).

```
BPE TRAINING ALGORITHM
══════════════════════

Starting corpus: "low lower lowest"

Step 0: Start with character vocabulary
────────────────────────────────────────
Vocabulary: {'l', 'o', 'w', 'e', 'r', 's', 't', ' '}

Corpus as characters:
"l o w   l o w e r   l o w e s t"


Step 1: Count all adjacent pairs
────────────────────────────────
('l', 'o'): 3    ← most frequent
('o', 'w'): 3
('w', ' '): 1
('w', 'e'): 2
('e', 'r'): 1
('e', 's'): 1
('s', 't'): 1


Step 2: Merge most frequent pair
────────────────────────────────
Merge ('l', 'o') → 'lo'

Vocabulary: {'lo', 'w', 'e', 'r', 's', 't', ' '}

Corpus: "lo w   lo w e r   lo w e s t"


Step 3: Repeat counting
───────────────────────
('lo', 'w'): 3   ← most frequent
('w', ' '): 1
('w', 'e'): 2
...


Step 4: Merge again
───────────────────
Merge ('lo', 'w') → 'low'

Vocabulary: {'low', 'e', 'r', 's', 't', ' '}

Corpus: "low   low e r   low e s t"


Continue until vocabulary size reached...
─────────────────────────────────────────

Final vocabulary might be:
{'low', 'lower', 'lowest', 'er', 'est', 'e', 'r', 's', 't', ' ', ...}
```

Visual representation of the BPE merge tree:

```
BPE MERGE TREE
══════════════

Starting characters:  l   o   w   e   r   s   t

Merge 1: (l,o)→lo     lo      w   e   r   s   t
                       \     /
Merge 2: (lo,w)→low     low     e   r   s   t
                         \     /
Merge 3: (low,e)→lowe     lowe     r   s   t
                           \      /
Merge 4: (lowe,r)→lower     lower    s   t

Merge 5: (low,e)→lowe       lowe       s   t
                             \        /
Merge 6: (lowe,s)→lowes       lowes     t
                               \       /
Merge 7: (lowes,t)→lowest       lowest

Final tokens: [low, lower, lowest, ...]
```

### BPE Tokenization (Inference)

Once trained, BPE tokenizes new text by greedily matching the longest tokens:

```
BPE TOKENIZATION EXAMPLE
════════════════════════

Vocabulary (trained): {"the", "th", "e", "un", "happi", "happiness",
                       "ness", "ly", "s", "i", "n", "h", "a", "p", ...}

Input: "unhappiness"

Step 1: Find longest match starting at position 0
        "unhappiness"
         ^^
         "un" matches! Add to output.

        Output: ["un"]
        Remaining: "happiness"

Step 2: Find longest match starting at position 0 of remainder
        "happiness"
         ^^^^^^^^^
         "happiness" matches! Add to output.

        Output: ["un", "happiness"]
        Remaining: ""

Done!

Alternative (if "happiness" not in vocabulary):
        "happiness"
         ^^^^^
         "happi" matches!

        "ness"
         ^^^^
         "ness" matches!

        Output: ["un", "happi", "ness"]
```

### WordPiece (Used by BERT)

WordPiece is similar to BPE but uses a different merge criterion:

```
BPE vs WORDPIECE MERGE CRITERION
════════════════════════════════

BPE:       Merge most frequent pair
           Score = count(pair)

WordPiece: Merge pair that maximizes likelihood
           Score = count(pair) / (count(token1) × count(token2))

Example:
─────────
Corpus frequencies:
  "un": 1000, "happi": 500, "unhappi": 200

BPE would merge based on raw count of "unhappi" (200)

WordPiece score = 200 / (1000 × 500) = 0.0004

If another pair "ly" (800) + "ing" (600) = "lying" (300):
WordPiece score = 300 / (800 × 600) = 0.000625

WordPiece prefers merges that aren't just combining frequent subwords,
but genuinely co-occur more than expected by chance.


WORDPIECE NOTATION
──────────────────
WordPiece marks continuation tokens with ##:

"unhappiness" → ["un", "##happi", "##ness"]
                      ^^
                These are "continuation" tokens
                (not starting a new word)

vs BPE which often uses space prefix:
"unhappiness" → ["un", "happiness"]
or              ["Ġun", "happiness"]  (GPT-style, Ġ = space)
```

### SentencePiece (Used by T5, LLaMA)

SentencePiece treats the input as raw bytes, avoiding the need for language-specific
pre-tokenization:

```
SENTENCEPIECE
═════════════

Traditional pipeline:
  "Hello world" → [Pre-tokenize by spaces] → ["Hello", "world"]
                                                    │
                                                    ▼
                                           [BPE/WordPiece]
                                                    │
                                                    ▼
                                           ["Hel", "lo", "wor", "ld"]

SentencePiece pipeline:
  "Hello world" → [Direct to subword tokenization] → ["▁Hello", "▁world"]
                  (no pre-tokenization needed)
                                                      ▁ = space marker

Advantages:
  • Language-agnostic (works for Chinese, Japanese without word boundaries)
  • Reversible (can reconstruct exact original text)
  • Treats space as a regular character (▁)
```

### Tokenization Quirks and Gotchas

Understanding tokenization quirks is crucial for working with LLMs effectively:

```
TOKENIZATION QUIRKS
═══════════════════

1. LEADING SPACE MATTERS
────────────────────────
" hello" ≠ "hello"

tokenize(" hello") → [" hello"] or ["Ġhello"]
tokenize("hello")  → ["hello"]

These are DIFFERENT tokens with different embeddings!

When concatenating: "Say" + "hello" → "Sayhello" (no space!)
                    "Say" + " hello" → "Say hello" ✓


2. NUMBER TOKENIZATION IS INCONSISTENT
──────────────────────────────────────
"1000" might tokenize as:
  • ["1000"]           - if common enough
  • ["10", "00"]       - split by pairs
  • ["1", "000"]       - different split
  • ["1", "0", "0", "0"] - character-level

This causes arithmetic problems:
  "1000 + 1 = ?"

  If "1000" = ["10", "00"], model might think about
  "10" and "00" separately, not "1000" as a number.


3. CASE SENSITIVITY
───────────────────
"Hello" and "hello" are typically different tokens
"THE" and "the" and "The" are three different tokens

This can cause issues with proper nouns and capitalization.


4. UNICODE AND SPECIAL CHARACTERS
─────────────────────────────────
"café" might tokenize as:
  • ["café"]           - single token (common)
  • ["caf", "é"]       - split at accent
  • ["caf", "Ã©"]      - encoding issues!

Emojis: "😀" → ["😀"] or ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]


5. CODE TOKENIZATION
────────────────────
"def function_name():" → ["def", " function", "_", "name", "():", ...]

Indentation:
"    x = 1" → ["    ", "x", " =", " 1"]
              ^^^^
              4 spaces as one token (or separate tokens?)

Different models handle code differently!
```

### Tokenization Implementation

```python
# Tokenization with Hugging Face Transformers
from transformers import AutoTokenizer

# Load tokenizer (each model has its own!)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Basic tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['Hello', ',', ' how', ' are', ' you', '?']

# Get token IDs
token_ids = tokenizer.encode(text)
print(token_ids)  # [15496, 11, 703, 389, 345, 30]

# Full encoding with attention mask
encoded = tokenizer(text, return_tensors="pt")
print(encoded)
# {'input_ids': tensor([[15496, 11, 703, 389, 345, 30]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(decoded)  # "Hello, how are you?"

# Batch tokenization with padding
texts = ["Hello!", "How are you doing today?"]
encoded = tokenizer(texts, padding=True, return_tensors="pt")
print(encoded['input_ids'])
# tensor([[15496,  0,  0,  0,  0,  0],    <- padded
#         [ 2437, 389, 345, 1804, 1909, 30]])


# Exploring vocabulary
print(f"Vocabulary size: {tokenizer.vocab_size}")  # 50257 for GPT-2

# Special tokens
print(tokenizer.special_tokens_map)
# {'bos_token': '<|endoftext|>',
#  'eos_token': '<|endoftext|>',
#  'unk_token': '<|endoftext|>'}

# Token → ID lookup
print(tokenizer.convert_tokens_to_ids(['hello', 'Hello', ' hello']))
# [31373, 15496, 23748]  <- All different!
```

### Tokenizer Comparison

```
TOKENIZER COMPARISON TABLE
══════════════════════════

Tokenizer    Used By       Vocab Size    Features
─────────    ───────       ──────────    ────────
BPE          GPT-2/3/4     50,257        Space prefix (Ġ)
                                         Byte-level fallback

WordPiece    BERT          30,522        ## continuation
                                         [CLS], [SEP] tokens

SentencePiece T5, LLaMA    32,000        ▁ space marker
                                         Language-agnostic

Tiktoken     GPT-3.5/4     100,000+      Optimized BPE
                                         Fast Rust impl.


TOKENS PER WORD (AVERAGE)
─────────────────────────

Language        GPT-2 BPE    LLaMA SP
────────        ─────────    ────────
English         1.3          1.2
Spanish         1.5          1.3
German          1.6          1.4
Chinese         2.5          2.0      ← Each character often = 2+ tokens
Japanese        2.8          2.2
Code (Python)   1.8          1.5
```

---

## 6.3 Pre-training

### The Core Objective: Next Token Prediction

Pre-training is deceptively simple: predict the next token given all previous tokens.
This single objective, at sufficient scale, produces remarkable capabilities.

```
NEXT TOKEN PREDICTION
═════════════════════

Input sequence:  "The cat sat on the"
                  ─────────────────
                         │
                         ▼
                 ┌───────────────┐
                 │               │
                 │  Transformer  │
                 │   Decoder     │
                 │               │
                 └───────┬───────┘
                         │
                         ▼
              Probability distribution
              over vocabulary (50,257)

              P("mat")    = 0.15   ← highest
              P("floor")  = 0.12
              P("table")  = 0.08
              P("couch")  = 0.05
              P("the")    = 0.02
              ...
              P("xyzzy")  = 0.0001

Target: Maximize P(actual_next_token)

Loss = -log P("mat" | "The cat sat on the")
     = -log(0.15)
     = 1.89  (lower is better)
```

### Causal Language Modeling

The key constraint is **causality**: each position can only attend to previous positions.
This is enforced through masked self-attention:

```
CAUSAL MASKING IN PRE-TRAINING
══════════════════════════════

Sequence: "The cat sat on the mat"

Position:   0     1     2     3     4     5
Token:     "The" "cat" "sat" "on" "the" "mat"

At each position, model predicts next token using only previous context:

Position 0: [The] → predict "cat"
            Uses: "The"

Position 1: [The][cat] → predict "sat"
            Uses: "The", "cat"

Position 2: [The][cat][sat] → predict "on"
            Uses: "The", "cat", "sat"

...and so on

ATTENTION MASK:
───────────────
           Query position
           0   1   2   3   4   5
        ┌───┬───┬───┬───┬───┬───┐
      0 │ ✓ │ ✗ │ ✗ │ ✗ │ ✗ │ ✗ │
        ├───┼───┼───┼───┼───┼───┤
Key   1 │ ✓ │ ✓ │ ✗ │ ✗ │ ✗ │ ✗ │
pos   2 │ ✓ │ ✓ │ ✓ │ ✗ │ ✗ │ ✗ │
      3 │ ✓ │ ✓ │ ✓ │ ✓ │ ✗ │ ✗ │
      4 │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✗ │
      5 │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
        └───┴───┴───┴───┴───┴───┘

✓ = Can attend    ✗ = Masked (set to -∞)
```

### Pre-training Data

The quality and diversity of pre-training data significantly impacts model capabilities:

```
PRE-TRAINING DATA SOURCES
═════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                         COMMON DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Web Crawl (Common Crawl)          ████████████████████████  ~60-70%   │
│  • Massive scale (petabytes)                                            │
│  • Highly diverse                                                       │
│  • Requires heavy filtering                                             │
│                                                                         │
│  Books (Books3, Gutenberg)         ████████                  ~15%      │
│  • High quality prose                                                   │
│  • Long-form coherent text                                              │
│  • Copyright concerns                                                   │
│                                                                         │
│  Wikipedia                         ████                      ~5%       │
│  • Factual, well-structured                                             │
│  • Neutral tone                                                         │
│  • Regularly updated                                                    │
│                                                                         │
│  Code (GitHub)                     ████                      ~5-10%    │
│  • Programming languages                                                │
│  • Documentation, comments                                              │
│  • Enables code generation                                              │
│                                                                         │
│  Scientific Papers (arXiv, etc.)   ██                        ~3%       │
│  • Technical knowledge                                                  │
│  • Mathematical notation                                                │
│                                                                         │
│  Other (news, forums, etc.)        ██                        ~5%       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


DATA PROCESSING PIPELINE
════════════════════════

Raw Web Crawl
     │
     ▼
┌─────────────┐
│  Language   │  Filter to target language(s)
│  Detection  │  Remove non-text content
└─────┬───────┘
      │
      ▼
┌─────────────┐
│   Quality   │  Remove low-quality pages
│   Filters   │  • Perplexity filtering (too high = gibberish)
└─────┬───────┘  • Length filtering (too short = low value)
      │          • Repetition filtering (too repetitive = spam)
      ▼
┌─────────────┐
│Deduplication│  Remove exact duplicates
│             │  Remove near-duplicates (MinHash, SimHash)
└─────┬───────┘  Critical! Training on duplicates causes memorization
      │
      ▼
┌─────────────┐
│   Content   │  Remove PII (names, emails, phones)
│   Filtering │  Remove toxic/harmful content
└─────┬───────┘  Remove benchmark contamination
      │
      ▼
┌─────────────┐
│ Tokenization│  Convert to token sequences
│  & Packing  │  Pack multiple documents to max_length
└─────┬───────┘
      │
      ▼
  Training Data
  (~2T tokens)
```

### Training Configuration

```
TYPICAL PRE-TRAINING SETUP (LLaMA 2 70B scale)
═══════════════════════════════════════════════

Model Architecture:
───────────────────
• Parameters: 70B
• Layers: 80
• Hidden dim: 8192
• Attention heads: 64
• Context length: 4096 tokens

Training:
─────────
• Batch size: 4M tokens (1024 sequences × 4096 tokens)
• Training tokens: 2T total
• Steps: 500,000
• Learning rate: 3e-4 (with warmup and decay)
• Optimizer: AdamW (β1=0.9, β2=0.95)
• Weight decay: 0.1

Hardware:
─────────
• GPUs: 2,048 × A100-80GB
• Training time: ~3 months
• Estimated cost: $2-5M

Data:
─────
• Sources: Web, books, code, Wikipedia, papers
• Ratio: 67% web, 15% books, 8% code, 4.5% Wikipedia, 5.5% other
```

### Scaling Laws

One of the most important discoveries in LLM research is that performance scales
predictably with compute, data, and model size:

```
SCALING LAWS (Kaplan et al. / Chinchilla)
════════════════════════════════════════

Loss scales as power law with three factors:

L(N, D, C) = A/N^α + B/D^β + E

Where:
  L = Cross-entropy loss (lower = better)
  N = Number of parameters
  D = Dataset size (tokens)
  C = Compute (FLOPs)
  α ≈ 0.076, β ≈ 0.095 (empirically measured)
  A, B, E = constants


VISUALIZATION:
──────────────

Loss
  │
  │ ╲
  │   ╲
  │     ╲╲
  │       ╲╲╲
  │          ╲╲╲╲
  │              ╲╲╲╲╲
  │                   ╲╲╲╲╲╲╲╲
  │                            ╲╲╲╲──────
  └──────────────────────────────────────→
                                      Scale (log)

Key observation: Smooth, predictable improvement!
No diminishing returns yet (as of current scale)


CHINCHILLA OPTIMAL SCALING
══════════════════════════

Chinchilla (Hoffmann et al., 2022) found:

For fixed compute budget C:
  • Optimal parameters N ∝ C^0.5
  • Optimal tokens D ∝ C^0.5
  • Ratio: D/N ≈ 20 tokens per parameter

Example:
────────
Budget: 10^24 FLOPs

                    Pre-Chinchilla    Chinchilla-Optimal
                    ──────────────    ──────────────────
Parameters:         280B              70B
Tokens:             300B              1.4T
D/N ratio:          1.1               20
Final loss:         2.0               1.7  ← Lower!

Implication: Many early LLMs were under-trained!
GPT-3 (175B params, 300B tokens) should have seen ~3.5T tokens


COMPUTE-OPTIMAL FRONTIER
════════════════════════

Parameters (N)
     │
     │                                          ★ Overparam'd
     │                                         /  (GPT-3)
     │                                        /
     │                              Optimal /
     │                              ratio  /
     │                                    / ★ Chinchilla
     │                                   /
     │                                  /
     │                                 /
     │                    ★ Under-  /
     │                    param'd /
     │                          /
     │                         /
     │________________________/________________→
                                         Data (D)

For any compute budget, there's an optimal (N, D) pair
that lies on the diagonal line.
```

### Pre-training Loss Landscape

```
LOSS DURING PRE-TRAINING
════════════════════════

Loss
  │
4 │ ╲
  │  ╲
  │   ╲
3 │    ╲
  │     ╲
  │      ╲
2 │       ╲__
  │          ╲___
  │              ╲____
1 │                   ╲_________
  │                             ╲______________
  │
  └──────────────────────────────────────────→
                                        Tokens seen
  │◄─────────────►│◄────────────────────────►│
    Rapid initial      Slow, steady improvement
    improvement

Phases:
───────
1. Random initialization (~4.0 loss)
2. Learning basic patterns (→ ~2.5)
3. Grammar and structure (→ ~2.0)
4. World knowledge (→ ~1.8)
5. Nuanced understanding (→ ~1.5)

The loss keeps improving even at 2T tokens!
```

### Pre-training Implementation (Simplified)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class PreTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Optimizer (AdamW is standard for LLMs)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,      # 3e-4
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        # Learning rate scheduler (cosine with warmup)
        self.scheduler = self.get_scheduler()

    def get_scheduler(self):
        """Cosine decay with linear warmup"""
        def lr_lambda(step):
            # Linear warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            # Cosine decay
            progress = (step - self.config.warmup_steps) / (
                self.config.total_steps - self.config.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def compute_loss(self, input_ids):
        """
        Standard causal LM loss: predict next token at each position
        """
        # Forward pass (model outputs logits for each position)
        # Shape: [batch, seq_len, vocab_size]
        logits = self.model(input_ids)

        # Shift for next-token prediction
        # logits[i] should predict input_ids[i+1]
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()   # [batch, seq-1]

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # [batch*seq, vocab]
            shift_labels.view(-1),                          # [batch*seq]
            ignore_index=self.tokenizer.pad_token_id
        )

        return loss

    def train_step(self, batch):
        """Single training step with gradient accumulation"""
        self.model.train()

        # Accumulate gradients over multiple micro-batches
        total_loss = 0
        for micro_batch in batch.split(self.config.micro_batch_size):
            loss = self.compute_loss(micro_batch)
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm  # typically 1.0
        )

        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return total_loss

    def train(self, dataloader):
        """Main training loop"""
        for step, batch in enumerate(dataloader):
            loss = self.train_step(batch)

            # Logging
            if step % self.config.log_every == 0:
                ppl = math.exp(loss)  # Perplexity
                lr = self.scheduler.get_last_lr()[0]
                print(f"Step {step}: loss={loss:.4f}, ppl={ppl:.2f}, lr={lr:.2e}")

            # Checkpointing
            if step % self.config.save_every == 0:
                self.save_checkpoint(step)

            # Evaluation
            if step % self.config.eval_every == 0:
                self.evaluate()
```

### Perplexity: Measuring Language Model Quality

Perplexity is the standard metric for evaluating language models:

```
PERPLEXITY EXPLAINED
════════════════════

Perplexity = exp(average cross-entropy loss)
           = exp(H(p, q))
           = 2^(bits per token)

Intuition: "How surprised is the model by the test data?"

Example:
────────
Loss = 2.0 → Perplexity = e^2.0 = 7.4
Loss = 1.5 → Perplexity = e^1.5 = 4.5
Loss = 1.0 → Perplexity = e^1.0 = 2.7

Lower perplexity = Better model

Another interpretation:
───────────────────────
Perplexity ≈ "effective vocabulary size" the model is choosing from

PPL = 7.4 means: On average, the model is as uncertain as if
                 uniformly choosing among ~7 equally likely tokens

PPL = 2.7 means: On average, choosing among ~3 equally likely tokens
                 (much more confident!)


BENCHMARK PERPLEXITIES
══════════════════════

Model           WikiText-103 PPL
─────           ────────────────
LSTM (2016)     48.7
GPT-2 (2019)    19.9
GPT-3 (2020)    ~15
Chinchilla      ~10
GPT-4           ~8 (est.)
```

---

## 6.4 Fine-tuning

### Why Fine-tune?

Pre-trained LLMs are general-purpose next-token predictors. Fine-tuning adapts them for
specific use cases:

```
PRE-TRAINED VS FINE-TUNED
═════════════════════════

Pre-trained base model (e.g., Llama-2-70B):
┌─────────────────────────────────────────────────────────────────────┐
│ Prompt: "What is the capital of France?"                            │
│                                                                     │
│ Completion: "What is the capital of Germany? What is the capital   │
│             of Spain? What is the capital..."                       │
│                                                                     │
│ (Continues in the style of training data - more questions!)        │
└─────────────────────────────────────────────────────────────────────┘

Fine-tuned instruction model (e.g., Llama-2-70B-Chat):
┌─────────────────────────────────────────────────────────────────────┐
│ Prompt: "What is the capital of France?"                            │
│                                                                     │
│ Completion: "The capital of France is Paris. Paris is also the     │
│             largest city in France and serves as the country's     │
│             political, economic, and cultural center."              │
│                                                                     │
│ (Answers the question helpfully!)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Types of Fine-tuning

```
FINE-TUNING SPECTRUM
════════════════════

                     Parameters       Memory          Performance
                     Updated          Required        vs Full FT
                     ────────         ────────        ──────────
Full Fine-tuning     100%             100%+           Baseline
                     (70B)            (140GB+)

LoRA                 0.1-1%           ~20%            95-99%
                     (70-700M)        (~30GB)

QLoRA                0.1-1%           ~10%            90-98%
                     (70-700M)        (~15GB)         (with 4-bit base)

Prefix Tuning        <0.1%            ~15%            85-95%
                     (10-100M)        (~20GB)

Prompt Tuning        <0.01%           ~15%            80-90%
                     (1-10M)          (~20GB)

                     ▲                                ▲
                     │                                │
              Less parameters          Trade-off      More capability
              More efficient                          Better quality
```

### Full Fine-tuning

Update all model parameters on task-specific data:

```
FULL FINE-TUNING
════════════════

Pre-trained weights: W_pretrained (all 70B parameters)
                           │
                           ▼
              ┌─────────────────────────┐
              │   Fine-tuning on        │
              │   task-specific data    │
              │   (e.g., instructions)  │
              └─────────────────────────┘
                           │
                           ▼
Fine-tuned weights: W_finetuned (all 70B parameters modified)


Memory Requirements:
────────────────────
• Model weights (fp16):  70B × 2 bytes = 140GB
• Gradients (fp16):      70B × 2 bytes = 140GB
• Optimizer states:      70B × 8 bytes = 560GB  (Adam has 2 states)
• Activations:           Variable (~50-200GB)
───────────────────────────────────────────────
Total:                   ~900GB+ GPU memory!

This requires multi-GPU setups with model parallelism.
```

### LoRA: Low-Rank Adaptation

LoRA makes fine-tuning efficient by only updating small "adapter" matrices:

```
LORA CORE IDEA
══════════════

Key insight: Weight updates during fine-tuning are low-rank!
            (Can be approximated by small matrices)

Instead of updating W directly:
  W' = W + ΔW     where ΔW is [d × d] (huge!)

Factor ΔW as product of small matrices:
  W' = W + A × B  where A is [d × r], B is [r × d], r << d


VISUALIZATION:
──────────────

Original weight matrix W:              Low-rank update ΔW = A × B:
┌──────────────────────────┐          ┌──────┐   ┌──────────────────────┐
│                          │          │      │   │                      │
│                          │          │      │   │                      │
│         [4096 × 4096]    │    ≈     │  A   │ × │         B            │
│         16M parameters   │          │      │   │                      │
│         (frozen)         │          │      │   │                      │
│                          │          │[4096 │   │     [8 × 4096]       │
│                          │          │  × 8]│   │      32K params      │
└──────────────────────────┘          └──────┘   └──────────────────────┘
                                       32K
                                       params

Total trainable: 32K + 32K = 64K params (instead of 16M!)
                            0.4% of original


TYPICAL RANKS:
──────────────
Rank (r)    Params per layer    % of original    Quality
────────    ────────────────    ─────────────    ───────
8           64K                 0.4%             Good
16          128K                0.8%             Better
32          256K                1.6%             Best
64          512K                3.2%             Diminishing returns
```

### LoRA Architecture in Detail

```
LORA IN A TRANSFORMER LAYER
═══════════════════════════

                Input x
                   │
                   ▼
         ┌─────────────────────┐
         │                     │
         │   W_q  W_k  W_v     │ ← Original attention weights (frozen)
         │    ↓    ↓    ↓      │
         │   +     +     +     │ ← Add LoRA adapters
         │    │    │    │      │
         │   A_q  A_k  A_v     │ ← LoRA A matrices [d × r]
         │   ↓    ↓    ↓      │
         │   B_q  B_k  B_v     │ ← LoRA B matrices [r × d]
         │                     │
         │   [Attention]       │
         │         ↓           │
         │      W_o            │ ← Output projection (frozen or LoRA)
         │         ↓           │
         │   Layer Norm        │
         │         ↓           │
         │      W_ffn          │ ← FFN weights (frozen or LoRA)
         │                     │
         └─────────────────────┘
                   │
                   ▼
               Output


WHICH LAYERS TO APPLY LORA:
───────────────────────────

Option 1: Attention only (default)
  Apply to: W_q, W_v (most important)
  Skip:     W_k, W_o, FFN
  Efficiency: Best

Option 2: All attention
  Apply to: W_q, W_k, W_v, W_o
  Skip:     FFN
  Efficiency: Good

Option 3: All layers
  Apply to: W_q, W_k, W_v, W_o, W_ffn1, W_ffn2
  Skip:     Nothing
  Efficiency: Lower, but best quality
```

### LoRA Implementation

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    LoRA adapter for a linear layer.
    Computes: y = Wx + (A @ B)x = Wx + ABx
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,  # Scaling factor
        dropout: float = 0.1
    ):
        super().__init__()

        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # How much to scale the LoRA contribution

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA matrices
        # A: down-projection [in_features → rank]
        # B: up-projection [rank → out_features]
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features)  # Initialize to zero!
        )

        self.dropout = nn.Dropout(dropout)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        # Original forward pass
        original_output = self.original(x)

        # LoRA forward pass: x @ A @ B * scaling
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return original_output + lora_output


def add_lora_to_model(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """
    Add LoRA adapters to specified modules in a model.
    """
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALinear(module, rank, alpha))

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# Example usage
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = add_lora_to_model(model, rank=16)
# Output: Trainable: 4,194,304 / 6,738,415,616 (0.06%)
```

### QLoRA: Quantized LoRA

QLoRA combines LoRA with 4-bit quantization for even more efficiency:

```
QLORA ARCHITECTURE
══════════════════

Standard LoRA:
┌─────────────────────────────────────────────────────────────────────┐
│  Base model weights: float16 (2 bytes/param)                        │
│  70B params × 2 bytes = 140GB                                       │
│                                                                     │
│  LoRA adapters: float16                                             │
│  ~100M params × 2 bytes = 200MB                                     │
│                                                                     │
│  Total: ~140GB                                                      │
└─────────────────────────────────────────────────────────────────────┘


QLoRA:
┌─────────────────────────────────────────────────────────────────────┐
│  Base model weights: NF4 (4-bit) with double quantization           │
│  70B params × 0.5 bytes = 35GB                                      │
│                                                                     │
│  LoRA adapters: BFloat16 (computed in higher precision)             │
│  ~100M params × 2 bytes = 200MB                                     │
│                                                                     │
│  Total: ~35GB (fits on single 40GB GPU!)                            │
└─────────────────────────────────────────────────────────────────────┘


QLORA INNOVATIONS:
──────────────────

1. 4-bit NormalFloat (NF4):
   • Optimized quantization for normally distributed weights
   • Better than uniform INT4

2. Double Quantization:
   • Quantize the quantization constants too
   • Saves additional memory

3. Paged Optimizers:
   • Use CPU RAM for optimizer states
   • Only move to GPU when needed
```

### QLoRA Implementation

```python
# Using the bitsandbytes library
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    bnb_4bit_use_double_quant=True,      # Double quantization
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",  # Automatic GPU allocation
)

# Add LoRA on top
lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling
    target_modules=[               # Which layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 167,772,160 || all params: 70,000,000,000
#         || trainable%: 0.24%
```

### Supervised Fine-Tuning (SFT) Data

Fine-tuning requires high-quality instruction-response pairs:

```
SFT DATA FORMAT
═══════════════

Standard format:
{
    "instruction": "Write a haiku about programming",
    "input": "",  # Optional additional context
    "output": "Code flows like water\nBugs emerge then fade away\nStack overflow saves"
}

Chat format (for multi-turn):
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What's its population?"},
        {"role": "assistant", "content": "Paris has about 2.1 million..."}
    ]
}


DATA QUALITY MATTERS MORE THAN QUANTITY:
────────────────────────────────────────

Dataset Size     Quality        Result
────────────     ───────        ──────
1M examples      Low quality    Mediocre model (learns noise)
50K examples     High quality   Good model
10K examples     Very high      Often sufficient!
1K examples      Expert-level   Can be competitive (LIMA paper)


LIMA PAPER INSIGHT (2023):
──────────────────────────
"A lot of alignment with users comes from pre-training,
and fine-tuning is mostly about learning style/format."

→ Small amounts of high-quality data can be very effective!
```

---

## 6.5 RLHF (Reinforcement Learning from Human Feedback)

### Why RLHF?

SFT teaches the model to follow instructions, but doesn't optimize for what humans
actually prefer. RLHF closes this gap:

```
SFT vs RLHF
═══════════

SFT (Supervised Fine-Tuning):
  Model learns to: Imitate the training examples
  Limitation: Can only be as good as the examples

RLHF (Reinforcement Learning from Human Feedback):
  Model learns to: Maximize human preference scores
  Advantage: Can potentially exceed demonstration quality


EXAMPLE:
────────

Question: "Explain quantum entanglement"

SFT response (imitating examples):
  "Quantum entanglement is when two particles become correlated..."
  [Technically correct but dry]

RLHF response (optimizing for preference):
  "Imagine you have two magic coins that always land the same way,
   no matter how far apart they are. That's quantum entanglement!
   More precisely, when particles become entangled, measuring one
   instantly affects the other, even across vast distances..."
  [Engaging, uses analogy, then precise definition]

Humans consistently prefer the RLHF response!
```

### The RLHF Pipeline

```
RLHF THREE-STAGE PIPELINE
═════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Supervised Fine-Tuning (SFT)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Pre-trained     Demonstration        SFT                              │
│  Base Model  +   Data             →   Model                            │
│  (Llama-2)       (instruction-        (follows instructions)           │
│                   response pairs)                                       │
│                                                                         │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Reward Model Training                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Human Preference Data:                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ Prompt: "Explain photosynthesis"        │                           │
│  │                                         │                           │
│  │ Response A: "Plants use sunlight..."    │  ← Preferred              │
│  │ Response B: "Photosynthesis is..."      │                           │
│  │                                         │                           │
│  │ Human label: A > B                      │                           │
│  └─────────────────────────────────────────┘                           │
│                                                                         │
│  Train reward model R(prompt, response) → scalar score                  │
│                                                                         │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: RL Optimization (PPO)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    ┌──────────────┐                                     │
│   Prompt ────────► │  Policy      │ ────────► Response                  │
│                    │  (SFT Model) │                                     │
│                    └──────────────┘                                     │
│                           │                                             │
│                           ▼                                             │
│                    ┌──────────────┐                                     │
│                    │   Reward     │ ────────► Score                     │
│                    │    Model     │                                     │
│                    └──────────────┘                                     │
│                           │                                             │
│                           ▼                                             │
│                    ┌──────────────┐                                     │
│                    │     PPO      │ ────────► Update Policy             │
│                    │  Algorithm   │                                     │
│                    └──────────────┘                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reward Model Training

```
REWARD MODEL DETAILS
════════════════════

Architecture: Same as base LLM, but with a linear head for scalar output

     Prompt + Response
            │
            ▼
    ┌───────────────┐
    │               │
    │  Transformer  │  (initialized from SFT model)
    │   Encoder     │
    │               │
    └───────┬───────┘
            │
            ▼
    [CLS/last token embedding]
            │
            ▼
    ┌───────────────┐
    │  Linear Head  │  [d_model → 1]
    └───────┬───────┘
            │
            ▼
       Scalar Score


TRAINING OBJECTIVE:
───────────────────

Given preference pair (prompt, response_chosen, response_rejected):

Loss = -log(σ(R(prompt, chosen) - R(prompt, rejected)))

Where σ is sigmoid function.

Intuition: Maximize probability that chosen scores higher than rejected.


PREFERENCE DATA COLLECTION:
───────────────────────────

Option 1: Binary comparison
  "Which response is better? A or B"
  Easier for annotators, more data volume

Option 2: Likert scale ratings
  "Rate this response 1-5"
  More nuanced, but noisier

Option 3: Ranking multiple responses
  "Rank these 4 responses best to worst"
  Most information per annotation


CHALLENGES:
───────────

1. Annotator disagreement: ~30% of pairs have no clear winner
2. Preference gaming: Model finds "reward hacks"
3. Distribution shift: RM trained on SFT outputs, but evaluated on RL outputs
```

### PPO (Proximal Policy Optimization)

```
PPO FOR RLHF
════════════

Goal: Update policy π to maximize reward while staying close to SFT policy

Objective:
  max E[R(x, y)] - β × KL(π || π_SFT)
   π

Where:
  - R(x, y) = reward model score
  - KL(π || π_SFT) = how far π has drifted from SFT
  - β = coefficient controlling the constraint


WHY THE KL PENALTY?
───────────────────

Without KL constraint:
  • Model finds "reward hacks" (ways to get high scores that aren't helpful)
  • Outputs become degenerate (very long, repetitive, etc.)
  • Model forgets pre-training knowledge

With KL constraint:
  • Model stays "close" to SFT (which is already good)
  • Prevents catastrophic forgetting
  • Limits reward hacking


PPO UPDATE RULE (simplified):
─────────────────────────────

For each batch of prompts:

1. Generate responses with current policy
2. Score with reward model
3. Compute advantage = R - baseline
4. Update policy to increase probability of high-advantage actions
5. Clip updates to prevent too-large changes

ratio = π(action|state) / π_old(action|state)
clip_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio × advantage, clip_ratio × advantage)
```

### PPO Implementation (Simplified)

```python
import torch
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, config):
        self.policy = policy_model  # Model being trained
        self.ref = ref_model        # SFT model (frozen, for KL)
        self.reward = reward_model  # Reward model (frozen)
        self.config = config

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )

    def compute_rewards(self, prompts, responses):
        """Get reward scores and KL penalties"""
        # Reward from reward model
        rewards = self.reward(prompts, responses)

        # KL divergence from reference model
        with torch.no_grad():
            ref_logprobs = self.ref.get_logprobs(prompts, responses)
        policy_logprobs = self.policy.get_logprobs(prompts, responses)

        kl = policy_logprobs - ref_logprobs  # Per-token KL

        # Final reward = RM score - β × KL
        final_rewards = rewards - self.config.kl_coef * kl.sum(dim=-1)

        return final_rewards, kl

    def compute_advantages(self, rewards, values):
        """GAE (Generalized Advantage Estimation)"""
        advantages = []
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t+1] - values[t]
            advantage = delta + self.config.gamma * self.config.lam * last_advantage
            advantages.insert(0, advantage)
            last_advantage = advantage

        return torch.stack(advantages)

    def ppo_step(self, prompts, old_responses, old_logprobs, old_values):
        """Single PPO update step"""

        # Generate new responses (or use cached)
        responses = old_responses

        # Compute current policy probabilities
        logprobs = self.policy.get_logprobs(prompts, responses)

        # Compute rewards
        rewards, kl = self.compute_rewards(prompts, responses)

        # Compute advantages
        advantages = self.compute_advantages(rewards, old_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped objective
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        # Value function loss
        values = self.policy.get_values(prompts, responses)
        value_loss = F.mse_loss(values, rewards)

        # Total loss
        loss = policy_loss + self.config.value_coef * value_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'kl': kl.mean().item()
        }
```

### DPO: Direct Preference Optimization

DPO simplifies RLHF by eliminating the reward model entirely:

```
DPO vs RLHF
═══════════

RLHF Pipeline:
  Preferences → Train Reward Model → RL with PPO → Aligned Model
                      ↓                  ↓
                 Separate model      Complex training
                 needed              (RL is finicky)


DPO Pipeline:
  Preferences → Direct optimization → Aligned Model
                      ↓
                 Single training loop
                 (much simpler!)


DPO KEY INSIGHT:
────────────────

The optimal policy under RLHF has a closed-form solution:

π*(y|x) = π_ref(y|x) × exp(R(x,y) / β) / Z(x)

Rearranging:

R(x,y) = β × log(π*(y|x) / π_ref(y|x)) + β × log(Z(x))

This means we can express the reward in terms of the policy!

→ We don't need to train a separate reward model


DPO LOSS FUNCTION:
──────────────────

L_DPO = -log σ(β × (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

Where:
  - y_w = preferred (winning) response
  - y_l = rejected (losing) response
  - π_ref = reference (SFT) model
  - β = temperature parameter

Intuition: Increase probability of preferred, decrease probability of rejected,
           relative to the reference model.
```

### DPO Implementation

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log π(y_w|x)
    policy_rejected_logps: torch.Tensor,  # log π(y_l|x)
    reference_chosen_logps: torch.Tensor, # log π_ref(y_w|x)
    reference_rejected_logps: torch.Tensor,  # log π_ref(y_l|x)
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute DPO loss.

    Args:
        policy_*_logps: Log probabilities from the model being trained
        reference_*_logps: Log probabilities from the reference (SFT) model
        beta: Temperature parameter (higher = more conservative updates)

    Returns:
        Scalar loss value
    """
    # Compute log ratios
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps

    # DPO loss
    losses = -F.logsigmoid(beta * (policy_ratio - reference_ratio))

    return losses.mean()


class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, config):
        self.model = model
        self.ref_model = ref_model  # Frozen
        self.tokenizer = tokenizer
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def get_log_probs(self, model, input_ids, labels):
        """Compute log probabilities of labels given input"""
        outputs = model(input_ids)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding
        mask = (shift_labels != self.tokenizer.pad_token_id)
        return (selected_log_probs * mask).sum(dim=-1)

    def train_step(self, batch):
        """Single DPO training step"""
        # batch contains: prompt, chosen_response, rejected_response

        # Tokenize
        chosen = self.tokenizer(
            [p + c for p, c in zip(batch['prompt'], batch['chosen'])],
            return_tensors='pt', padding=True
        )
        rejected = self.tokenizer(
            [p + r for p, r in zip(batch['prompt'], batch['rejected'])],
            return_tensors='pt', padding=True
        )

        # Get log probs from policy
        policy_chosen_logps = self.get_log_probs(
            self.model, chosen['input_ids'], chosen['input_ids']
        )
        policy_rejected_logps = self.get_log_probs(
            self.model, rejected['input_ids'], rejected['input_ids']
        )

        # Get log probs from reference (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.ref_model, chosen['input_ids'], chosen['input_ids']
            )
            ref_rejected_logps = self.get_log_probs(
                self.ref_model, rejected['input_ids'], rejected['input_ids']
            )

        # Compute loss
        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.config.beta
        )

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### RLHF vs DPO Comparison

```
RLHF vs DPO COMPARISON
══════════════════════

                        RLHF (PPO)              DPO
                        ──────────              ───
Complexity              High                    Low
Training stability      Tricky                  Stable
Memory (3 models)       RM + Policy + Ref       Policy + Ref
Hyperparameters         Many (PPO tuning)       Few (just β)
Online data generation  Yes (can sample)        No (fixed dataset)
Performance             Slightly better         Very close
Industry adoption       ChatGPT, Claude         Llama 2, Zephyr

WHEN TO USE WHICH:
──────────────────

Use RLHF when:
  • Maximum performance needed
  • Resources for complex training
  • Want online data collection
  • Team has RL expertise

Use DPO when:
  • Simpler is better
  • Limited compute resources
  • Quick iteration needed
  • Preference data already collected
```

---

## 6.6 Prompting Techniques

### Why Prompting Matters

The way you phrase a request to an LLM dramatically affects its output quality.
Good prompting can unlock capabilities that seem absent with naive prompts:

```
PROMPTING IMPACT EXAMPLE
════════════════════════

Task: "What is 17 × 24?"

Naive prompt:
  "What is 17 × 24?"
  → "17 × 24 = 408"  ✓ (but often wrong for harder problems)

Better prompt:
  "Calculate 17 × 24 step by step."
  → "Step 1: 17 × 20 = 340
     Step 2: 17 × 4 = 68
     Step 3: 340 + 68 = 408
     Therefore, 17 × 24 = 408"  ✓ (much more reliable)

The model has the capability; prompting unlocks it!
```

### Zero-Shot Prompting

Ask directly without examples:

```
ZERO-SHOT PROMPTING
═══════════════════

Template:
┌─────────────────────────────────────────────────────────────────────┐
│ [Task description]                                                   │
│ [Input]                                                             │
│ [Output format instruction]                                          │
└─────────────────────────────────────────────────────────────────────┘

Example:
┌─────────────────────────────────────────────────────────────────────┐
│ Classify the sentiment of the following review as positive,         │
│ negative, or neutral.                                               │
│                                                                     │
│ Review: "The food was okay, nothing special but not bad either."    │
│                                                                     │
│ Sentiment:                                                          │
└─────────────────────────────────────────────────────────────────────┘

Output: "neutral"


WHEN TO USE:
────────────
• Simple, well-defined tasks
• Model has seen similar tasks in pre-training
• Don't have good examples available
```

### Few-Shot Prompting

Provide examples to demonstrate the task:

```
FEW-SHOT PROMPTING
══════════════════

Template:
┌─────────────────────────────────────────────────────────────────────┐
│ [Task description]                                                   │
│                                                                     │
│ [Example 1 input]                                                   │
│ [Example 1 output]                                                  │
│                                                                     │
│ [Example 2 input]                                                   │
│ [Example 2 output]                                                  │
│                                                                     │
│ [Example 3 input]                                                   │
│ [Example 3 output]                                                  │
│                                                                     │
│ [Actual input]                                                      │
│ [Model completes...]                                                │
└─────────────────────────────────────────────────────────────────────┘


Example (Sentiment Classification):
┌─────────────────────────────────────────────────────────────────────┐
│ Classify the sentiment as positive, negative, or neutral.           │
│                                                                     │
│ Review: "This product exceeded my expectations!"                    │
│ Sentiment: positive                                                 │
│                                                                     │
│ Review: "Terrible quality, broke after one day."                    │
│ Sentiment: negative                                                 │
│                                                                     │
│ Review: "It works as described, nothing more."                      │
│ Sentiment: neutral                                                  │
│                                                                     │
│ Review: "Absolutely love it, best purchase ever!"                   │
│ Sentiment:                                                          │
└─────────────────────────────────────────────────────────────────────┘

Output: "positive"


NUMBER OF EXAMPLES:
───────────────────

Examples    Context Used    Quality     Recommendation
────────    ────────────    ───────     ──────────────
0 (zero)    Minimal         Varies      Simple tasks
1-2         ~500 tokens     Good        Default starting point
3-5         ~1000 tokens    Better      Complex tasks
5-10        ~2000 tokens    Best        Very complex / specific
10+         >3000 tokens    Diminishing Edge cases only


EXAMPLE SELECTION MATTERS:
──────────────────────────

Good examples:
• Representative of the task distribution
• Cover edge cases
• Diverse (not all similar)
• Correct (errors will be learned!)

Order matters (usually):
• More recent examples have stronger influence
• Put most relevant example last
```

### Chain-of-Thought (CoT) Prompting

Encourage step-by-step reasoning:

```
CHAIN-OF-THOUGHT PROMPTING
══════════════════════════

Key insight: Models perform better on reasoning tasks when they
             "think out loud" before answering.


Standard prompting:
┌─────────────────────────────────────────────────────────────────────┐
│ Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.   │
│    Each can has 3 tennis balls. How many tennis balls does he       │
│    have now?                                                        │
│                                                                     │
│ A: 11                                                               │
└─────────────────────────────────────────────────────────────────────┘
(Model just outputs answer - often wrong for complex problems)


Chain-of-Thought prompting:
┌─────────────────────────────────────────────────────────────────────┐
│ Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.   │
│    Each can has 3 tennis balls. How many tennis balls does he       │
│    have now?                                                        │
│                                                                     │
│ A: Roger started with 5 balls.                                      │
│    2 cans × 3 balls per can = 6 balls.                              │
│    5 + 6 = 11 balls.                                                │
│    The answer is 11.                                                │
└─────────────────────────────────────────────────────────────────────┘
(Model shows reasoning - much more reliable)


ZERO-SHOT COT:
──────────────
Simply add "Let's think step by step" to the prompt!

┌─────────────────────────────────────────────────────────────────────┐
│ Q: [Complex problem]                                                │
│                                                                     │
│ A: Let's think step by step.                                        │
│    [Model generates reasoning chain]                                │
│    Therefore, the answer is [X].                                    │
└─────────────────────────────────────────────────────────────────────┘


COT VARIATIONS:
───────────────

1. Zero-shot CoT: "Let's think step by step"
2. Few-shot CoT: Provide examples with reasoning
3. Self-consistency: Generate multiple chains, vote
4. Tree of Thought: Branch and explore multiple paths
```

### Self-Consistency

Generate multiple answers and take majority vote:

```
SELF-CONSISTENCY
════════════════

Insight: Different reasoning paths may lead to the same answer.
         If multiple paths agree, we're more confident.


Process:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                        Same prompt                                  │
│                           │                                         │
│         ┌─────────────────┼─────────────────┐                       │
│         │                 │                 │                       │
│         ▼                 ▼                 ▼                       │
│    ┌─────────┐       ┌─────────┐       ┌─────────┐                  │
│    │ Path 1  │       │ Path 2  │       │ Path 3  │                  │
│    │         │       │         │       │         │                  │
│    │ Step 1  │       │ Step 1  │       │ Step 1  │                  │
│    │ Step 2  │       │ Step 2  │       │ Step 2  │                  │
│    │ Step 3  │       │ Step 3  │       │ Step 3  │                  │
│    │         │       │         │       │         │                  │
│    │ Ans: 42 │       │ Ans: 42 │       │ Ans: 38 │                  │
│    └─────────┘       └─────────┘       └─────────┘                  │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           │                                         │
│                           ▼                                         │
│                    Majority vote                                    │
│                           │                                         │
│                           ▼                                         │
│                     Answer: 42                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


IMPLEMENTATION:
───────────────
1. Sample N responses with temperature > 0
2. Extract final answer from each
3. Return most common answer

N=5 is often sufficient, N=10-40 for harder problems


WHEN TO USE:
────────────
• Problems with discrete answers (numbers, multiple choice)
• When accuracy is more important than cost
• Complex reasoning tasks
```

### System Prompts and Personas

Set context and behavior at the start of conversation:

```
SYSTEM PROMPTS
══════════════

Structure:
┌─────────────────────────────────────────────────────────────────────┐
│ SYSTEM: [Persona and behavior instructions]                         │
│                                                                     │
│ USER: [First user message]                                          │
│                                                                     │
│ ASSISTANT: [Model response]                                         │
└─────────────────────────────────────────────────────────────────────┘


Example:
┌─────────────────────────────────────────────────────────────────────┐
│ SYSTEM: You are a helpful Python programming assistant. You:        │
│ - Write clean, well-documented code                                 │
│ - Explain your reasoning                                            │
│ - Suggest improvements when appropriate                             │
│ - Use type hints in all function signatures                         │
│ - Follow PEP 8 style guidelines                                     │
│                                                                     │
│ USER: Write a function to find prime numbers up to n.               │
│                                                                     │
│ ASSISTANT: [Responds as instructed Python expert]                   │
└─────────────────────────────────────────────────────────────────────┘


EFFECTIVE SYSTEM PROMPTS:
─────────────────────────

1. Define the role clearly
   "You are a senior software engineer specializing in..."

2. List specific behaviors
   "Always include error handling"
   "Never use deprecated APIs"

3. Set constraints
   "Keep responses under 500 words"
   "Only answer questions about Python"

4. Provide format instructions
   "Format code with ```python``` blocks"
   "Use bullet points for lists"
```

### Structured Output Prompting

Get outputs in specific formats:

```
STRUCTURED OUTPUT
═════════════════

JSON Output:
┌─────────────────────────────────────────────────────────────────────┐
│ Extract the following information from the text as JSON:            │
│ - name (string)                                                     │
│ - age (integer)                                                     │
│ - occupation (string)                                               │
│                                                                     │
│ Text: "John Smith, a 35-year-old software engineer from Seattle..." │
│                                                                     │
│ JSON:                                                               │
│ ```json                                                             │
│ {                                                                   │
│   "name": "John Smith",                                             │
│   "age": 35,                                                        │
│   "occupation": "software engineer"                                 │
│ }                                                                   │
│ ```                                                                 │
└─────────────────────────────────────────────────────────────────────┘


Markdown Tables:
┌─────────────────────────────────────────────────────────────────────┐
│ Compare Python and JavaScript. Format as a markdown table with      │
│ columns: Feature, Python, JavaScript                                │
│                                                                     │
│ | Feature | Python | JavaScript |                                   │
│ |---------|--------|------------|                                   │
│ | Typing | Dynamic, optional hints | Dynamic |                      │
│ | ...                                                               │
└─────────────────────────────────────────────────────────────────────┘


XML/Tags for Parsing:
┌─────────────────────────────────────────────────────────────────────┐
│ Analyze the sentiment and extract key themes.                       │
│ Format your response as:                                            │
│                                                                     │
│ <analysis>                                                          │
│   <sentiment>positive/negative/neutral</sentiment>                  │
│   <themes>                                                          │
│     <theme>theme 1</theme>                                          │
│     <theme>theme 2</theme>                                          │
│   </themes>                                                         │
│   <summary>Brief summary</summary>                                  │
│ </analysis>                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Prompting Best Practices

```
PROMPTING BEST PRACTICES
════════════════════════

1. BE SPECIFIC AND CLEAR
   Bad:  "Summarize this"
   Good: "Summarize this article in 3 bullet points, focusing on
         the main findings and their implications for healthcare"


2. PROVIDE CONTEXT
   Bad:  "Fix this code"
   Good: "Fix this Python code that's supposed to sort a list.
         It's giving an IndexError on line 5. Here's the input
         data format..."


3. USE DELIMITERS FOR INPUTS
   Good: """
   Translate the text between triple quotes to French:
   '''The quick brown fox jumps over the lazy dog'''
   """


4. SPECIFY OUTPUT FORMAT
   Good: "List the top 5 causes, numbered 1-5, with one
         sentence explanation each"


5. GIVE THE MODEL AN "OUT"
   Good: "If you're not sure, say 'I'm not certain' rather
         than guessing"


6. ITERATE AND REFINE
   • Start simple
   • Add constraints as needed
   • Test on edge cases
   • Version control your prompts!


COMMON MISTAKES:
────────────────

❌ Vague instructions ("make it better")
❌ Assuming context the model doesn't have
❌ Not specifying format
❌ Too many instructions at once
❌ Not providing examples for complex tasks
```

---

## 6.7 Inference Optimization

### Why Optimization Matters

LLM inference is expensive! Optimization is crucial for production deployment:

```
INFERENCE COST BREAKDOWN
════════════════════════

Without optimization (70B model, 1000 requests/day):
┌─────────────────────────────────────────────────────────────────────┐
│  • GPU needed: 2× A100-80GB (140GB for model)                       │
│  • GPU rental: ~$4/hour × 24 hours = $96/day                        │
│  • Monthly cost: ~$2,900                                            │
│  • Latency: 500ms - 2s per token                                    │
└─────────────────────────────────────────────────────────────────────┘

With optimization (quantization, batching, KV cache):
┌─────────────────────────────────────────────────────────────────────┐
│  • GPU needed: 1× A100-40GB (4-bit = 35GB)                          │
│  • GPU rental: ~$2/hour × 24 hours = $48/day                        │
│  • Monthly cost: ~$1,450                                            │
│  • Latency: 50ms - 200ms per token                                  │
│  • Throughput: 10× higher                                           │
└─────────────────────────────────────────────────────────────────────┘

That's 50% cost reduction + 5-10× speed improvement!
```

### Temperature and Sampling

Control the randomness of generation:

```
TEMPERATURE EXPLAINED
═════════════════════

Softmax with temperature:

P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)

Where T is temperature.


EFFECT OF TEMPERATURE:
──────────────────────

Original logits: [2.0, 1.0, 0.5, 0.3]

T = 0.1 (Low - deterministic):
  P = [0.9999, 0.0001, 0.0000, 0.0000]
  → Almost always picks highest probability

T = 1.0 (Default):
  P = [0.51, 0.19, 0.12, 0.10]
  → Balanced sampling

T = 2.0 (High - random):
  P = [0.36, 0.25, 0.20, 0.18]
  → More uniform, more variety


VISUALIZATION:
──────────────

                     Probability Distribution
                     ────────────────────────

T = 0.1   █████████████████████████████████████████████▏  token_1
(sharp)   █                                                token_2
          ▏                                                token_3
          ▏                                                token_4

T = 1.0   ████████████████████████▏                        token_1
(balanced)█████████▏                                       token_2
          █████▏                                           token_3
          ████▏                                            token_4

T = 2.0   ██████████████▏                                  token_1
(uniform) ██████████▏                                      token_2
          ████████▏                                        token_3
          ███████▏                                         token_4


USE CASES:
──────────

T = 0      Greedy/deterministic (factual Q&A, code)
T = 0.3-0.7 Focused but varied (most tasks)
T = 0.7-1.0 Creative writing, brainstorming
T > 1.0    Very creative, may be incoherent
```

### Top-k and Top-p (Nucleus) Sampling

Filter the probability distribution before sampling:

```
TOP-K SAMPLING
══════════════

Only consider the k most probable tokens.

Example with k=3:

Original distribution:
  "the"  : 0.40
  "a"    : 0.25
  "an"   : 0.15
  "some" : 0.10
  "my"   : 0.05
  ...

After top-k (k=3):
  "the"  : 0.40 → 0.50 (renormalized)
  "a"    : 0.25 → 0.31
  "an"   : 0.15 → 0.19
  [others set to 0]

Problem: k is fixed regardless of distribution shape.
  - If distribution is peaked, k=50 might include garbage
  - If distribution is flat, k=5 might be too restrictive


TOP-P (NUCLEUS) SAMPLING
════════════════════════

Keep smallest set of tokens whose cumulative probability ≥ p.

Example with p=0.9:

Original distribution:
  "the"  : 0.40  │ cumsum: 0.40
  "a"    : 0.25  │ cumsum: 0.65
  "an"   : 0.15  │ cumsum: 0.80
  "some" : 0.10  │ cumsum: 0.90  ← stop here!
  "my"   : 0.05
  ...

After top-p (p=0.9):
  "the"  : 0.44 (renormalized)
  "a"    : 0.28
  "an"   : 0.17
  "some" : 0.11

Advantage: Adapts to distribution shape!
  - Peaked dist → fewer tokens
  - Flat dist → more tokens


COMBINING TEMPERATURE, TOP-K, TOP-P:
────────────────────────────────────

Typical pipeline:

logits → [Temperature] → [Top-k] → [Top-p] → [Sample]
           (scale)       (filter)  (filter)   (random)

Common settings:
  • Factual: T=0.3, top_p=0.9
  • Conversational: T=0.7, top_p=0.95
  • Creative: T=1.0, top_p=0.98
  • Code: T=0.2, top_p=0.9 (or greedy T=0)
```

### KV Cache

The most important inference optimization for autoregressive models:

```
KV CACHE EXPLAINED
══════════════════

Problem: Each new token requires attending to ALL previous tokens.

Without KV Cache:
─────────────────

Generating token 5:
  Input: [tok1, tok2, tok3, tok4]
  Compute Q, K, V for ALL tokens (4× work)

Generating token 6:
  Input: [tok1, tok2, tok3, tok4, tok5]
  Compute Q, K, V for ALL tokens (5× work)  ← Recomputing 1-4!

Generating token 7:
  Input: [tok1, tok2, tok3, tok4, tok5, tok6]
  Compute Q, K, V for ALL tokens (6× work)  ← Recomputing 1-5!

Total work: 4 + 5 + 6 = 15 (O(n²) growth!)


With KV Cache:
──────────────

Generating token 5:
  Compute Q, K, V for token 4 only
  Use cached K, V for tokens 1-3
  Store K4, V4 in cache

Generating token 6:
  Compute Q, K, V for token 5 only
  Use cached K, V for tokens 1-4
  Store K5, V5 in cache

Total work: 1 + 1 + 1 = 3 (O(n) growth!)


VISUAL COMPARISON:
──────────────────

Without cache:               With cache:
Attention matrix             Attention matrix

Token 1: ■                   Token 1: ■        (from cache)
Token 2: ■ ■                 Token 2: ■ ■      (from cache)
Token 3: ■ ■ ■               Token 3: ■ ■ ■    (from cache)
Token 4: ■ ■ ■ ■             Token 4: ■ ■ ■ ■  (compute once)
Token 5: ■ ■ ■ ■ ■           Token 5: □ □ □ □ ■ (compute only ■)
Token 6: ■ ■ ■ ■ ■ ■         Token 6: □ □ □ □ □ ■ (compute only ■)

■ = computed  □ = from cache


KV CACHE MEMORY:
────────────────

Cache size per layer:
  K: [batch, num_heads, seq_len, head_dim]
  V: [batch, num_heads, seq_len, head_dim]

For Llama-2 70B at 4096 sequence length:
  • 80 layers × 2 (K+V) × 4096 × 8192 × 2 bytes
  • ≈ 10GB per sequence in batch!

This is why batch size is limited in LLM inference.
```

### Quantization

Reduce precision to save memory and increase speed:

```
QUANTIZATION LEVELS
═══════════════════

Precision    Bits/param    Memory (70B)    Quality      Speed
─────────    ──────────    ────────────    ───────      ─────
FP32         32            280 GB          Perfect      1×
FP16/BF16    16            140 GB          ~Perfect     2×
INT8         8             70 GB           Very Good    3×
INT4         4             35 GB           Good         4×
NF4          4             35 GB           Better       4×


HOW QUANTIZATION WORKS (INT8):
──────────────────────────────

Original weight (FP16): -0.237, 0.456, -0.892, 0.123, ...

Step 1: Find scale factor
  max_val = max(|weights|) = 0.892
  scale = 127 / 0.892 = 142.4

Step 2: Quantize
  -0.237 × 142.4 = -34 → clamp to int8: -34
  0.456 × 142.4 = 65 → int8: 65
  -0.892 × 142.4 = -127 → int8: -127

Step 3: Store quantized weights + scale
  [−34, 65, −127, 18, ...] + scale = 142.4

Step 4: Dequantize during inference
  -34 / 142.4 = -0.239 (small error!)


QUANTIZATION METHODS:
─────────────────────

1. Absmax (simple): Use absolute maximum for scale
   Error: Higher near zero

2. Zero-point: Add offset for asymmetric distributions
   q = round(w × scale + zero_point)

3. Per-channel: Different scale per output channel
   Better accuracy, more metadata

4. GPTQ: Find optimal quantization via reconstruction error
   Best accuracy, slow to quantize

5. AWQ (Activation-aware): Weight importance from activations
   Good accuracy/speed tradeoff
```

### Speculative Decoding

Use a small model to draft tokens, large model to verify:

```
SPECULATIVE DECODING
════════════════════

Insight: Small models are fast but less accurate.
         Large models are slow but can verify quickly.

Process:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Step 1: Draft with small model (γ tokens)                          │
│  ────────────────────────────────────────                           │
│                                                                     │
│  Small model (7B, fast):                                            │
│    "The quick" → "brown" → "fox" → "jumps" → "over"                │
│                    γ = 4 draft tokens                               │
│                                                                     │
│  Step 2: Verify with large model (parallel!)                        │
│  ───────────────────────────────────────────                        │
│                                                                     │
│  Large model (70B) in ONE forward pass:                             │
│    Check: P("brown"|"The quick") = 0.8    ✓ Accept                 │
│    Check: P("fox"|"...brown") = 0.9       ✓ Accept                 │
│    Check: P("jumps"|"...fox") = 0.7       ✓ Accept                 │
│    Check: P("over"|"...jumps") = 0.3      ✗ Reject! Sample "and"   │
│                                                                     │
│  Step 3: Accept verified tokens, continue from rejection            │
│  ───────────────────────────────────────────────────────            │
│                                                                     │
│  Output: "The quick brown fox jumps and" (4 tokens in ~1 forward)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


ACCEPTANCE RATE:
────────────────

If small model is well-aligned with large model:
  • Most drafts are accepted (3-4 out of 5)
  • Speedup: 2-3× typical

If models disagree:
  • Most drafts rejected (1-2 out of 5)
  • Speedup: Minimal


WHEN IT WORKS:
──────────────

✓ Same tokenizer (required)
✓ Similar training data
✓ Similar temperature settings
✓ Draft model is reasonably good

✗ Very different model families
✗ Complex reasoning (draft model fails)
```

### Batching Strategies

Process multiple requests together for efficiency:

```
BATCHING FOR THROUGHPUT
═══════════════════════

Without batching (sequential):
┌──────────────────────────────────────────────────────────────────┐
│  Request 1: ████████████████████████████  (2s)                   │
│  Request 2: ────────────────────────────████████████████████████ │
│  Request 3: ────────────────────────────────────────────────████│
│                                                                  │
│  Total time for 3 requests: 6 seconds                            │
│  GPU utilization: ~30%                                           │
└──────────────────────────────────────────────────────────────────┘


With static batching:
┌──────────────────────────────────────────────────────────────────┐
│  Request 1: ████████████████████████████                         │
│  Request 2: ████████████████████████████  (all together)         │
│  Request 3: ████████████████████████████                         │
│                                                                  │
│  Total time for 3 requests: 2.5 seconds                          │
│  GPU utilization: ~90%                                           │
└──────────────────────────────────────────────────────────────────┘


Continuous/Dynamic batching:
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Time →                                                          │
│  ───────────────────────────────────────────────────────────     │
│                                                                  │
│  Batch: [R1, R2, R3]     [R1, R2, R4]   [R2, R4, R5]             │
│           ↓                  ↓              ↓                     │
│         R3 done           R1 done        R2 done                 │
│         Add R4            Add R5         Add R6                  │
│                                                                  │
│  Requests come and go, batch stays full!                         │
│  GPU always ~95% utilized                                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Inference Optimization Implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load with optimizations
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,          # FP16 precision
    device_map="auto",                   # Automatic device placement
    load_in_4bit=True,                   # 4-bit quantization
)

# Generation with sampling parameters
def generate_optimized(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    use_cache=True,  # KV cache enabled
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            use_cache=use_cache,           # Enable KV cache
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Batched inference
def generate_batch(model, tokenizer, prompts, **kwargs):
    """Process multiple prompts in a batch"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

---

## 6.8 Hallucination

### The Hallucination Problem

LLMs confidently generate false information:

```
HALLUCINATION EXAMPLES
══════════════════════

Factual hallucination:
┌─────────────────────────────────────────────────────────────────────┐
│ User: "When did Einstein win the Nobel Prize for relativity?"       │
│                                                                     │
│ LLM: "Einstein won the Nobel Prize in Physics in 1921 for his      │
│       theory of relativity."                                        │
│                                                                     │
│ WRONG! He won it for the photoelectric effect, not relativity.     │
└─────────────────────────────────────────────────────────────────────┘

Citation hallucination:
┌─────────────────────────────────────────────────────────────────────┐
│ User: "Give me a citation for research on X."                       │
│                                                                     │
│ LLM: "Smith, J. (2019). 'Effects of X on Y.' Journal of Made-Up    │
│       Science, 42(3), 156-178. DOI: 10.1000/fake.12345"            │
│                                                                     │
│ FABRICATED! Paper, authors, journal may not exist.                  │
└─────────────────────────────────────────────────────────────────────┘

Reasoning hallucination:
┌─────────────────────────────────────────────────────────────────────┐
│ User: "If all A are B, and all B are C, are all A also C?"          │
│                                                                     │
│ LLM: "Let me think step by step...                                  │
│       All A are B ✓                                                 │
│       All B are C ✓                                                 │
│       Therefore, some A are C."                                     │
│                                                                     │
│ WRONG! Should be "ALL A are C" (valid syllogism).                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Why LLMs Hallucinate

```
CAUSES OF HALLUCINATION
═══════════════════════

1. TRAINING DATA ISSUES
   ─────────────────────
   • Internet contains errors and misinformation
   • Training data has knowledge cutoff
   • Facts change over time
   • Rare topics have sparse coverage

   Example: Model trained on 2022 data doesn't know 2024 events


2. OPTIMIZATION FOR FLUENCY
   ────────────────────────
   • Loss function rewards plausible text, not truthful text
   • Confident-sounding text often scores well
   • No explicit "I don't know" in training

   Loss = -log P(next_token | context)
        ≠ -log P(true_token | context)


3. NO GROUNDING
   ─────────────
   • Model has no connection to external world
   • Can't verify claims against reality
   • No distinction between "I read this" and "this is true"


4. COMPRESSION AND GENERALIZATION
   ───────────────────────────────
   • Billions of facts compressed into finite parameters
   • Rare facts more likely to be lossy
   • Model generalizes patterns (sometimes incorrectly)


HALLUCINATION VS CAPABILITIES TRADEOFF:
───────────────────────────────────────

                    │
     Capability     │                     ★ GPT-4
                    │               ★ Claude
                    │         ★ Llama-2
                    │    ★ GPT-3
                    │
                    │ More capable models
                    │ can hallucinate more
                    │ convincingly!
                    │
                    └─────────────────────────────→
                                        Hallucination risk
```

### Hallucination Mitigations

```
MITIGATION STRATEGIES
═════════════════════

1. RETRIEVAL-AUGMENTED GENERATION (RAG)
   ─────────────────────────────────────
   Ground responses in retrieved documents.

   Query → [Retrieve docs] → [Generate with docs as context]

   "What's the GDP of France?"
       ↓
   Retrieve: "France GDP 2023: $3.05 trillion (IMF)"
       ↓
   Generate: "According to IMF data, France's GDP is $3.05T"


2. FINE-TUNING FOR FACTUALITY
   ───────────────────────────
   Train on factual Q&A with "I don't know" examples.

   Data: {
     "question": "What color is the President's cat?",
     "answer": "I don't have reliable information about this."
   }


3. CALIBRATED CONFIDENCE
   ──────────────────────
   Teach model to express uncertainty.

   High confidence: "Paris is the capital of France."
   Low confidence: "I believe the population is around 2 million,
                    but you should verify this."


4. SELF-CONSISTENCY CHECKING
   ─────────────────────────
   Generate multiple responses, flag disagreements.

   Response 1: "Einstein was born in 1879"
   Response 2: "Einstein was born in 1879"
   Response 3: "Einstein was born in 1879"
   → High agreement, likely correct

   Response 1: "The author is John Smith"
   Response 2: "The author is Jane Doe"
   → Disagreement! Flag for review


5. CHAIN-OF-THOUGHT + VERIFICATION
   ────────────────────────────────
   Make reasoning explicit, then verify each step.

   "Let me verify each claim:
   - Claim 1: Paris is in France ✓ (verifiable)
   - Claim 2: Eiffel Tower built 1889 ✓ (verifiable)
   - Claim 3: It has 1,063 steps ⚠️ (I should double-check)"


6. TOOL USE FOR VERIFICATION
   ─────────────────────────
   Call external tools to verify facts.

   LLM: "I think X, let me verify..."
       → [Call search API]
       → "Actually, according to Wikipedia, Y is correct."
```

---

## 6.9 In-Context Learning

### The Remarkable Phenomenon

LLMs can learn new tasks from examples in the prompt without any weight updates:

```
IN-CONTEXT LEARNING EXAMPLE
═══════════════════════════

Zero gradient updates, just examples in context!

┌─────────────────────────────────────────────────────────────────────┐
│ Prompt:                                                             │
│                                                                     │
│ Translate English to French:                                        │
│                                                                     │
│ English: "The cat sleeps on the mat."                               │
│ French: "Le chat dort sur le tapis."                                │
│                                                                     │
│ English: "I love programming."                                      │
│ French: "J'aime la programmation."                                  │
│                                                                     │
│ English: "Where is the library?"                                    │
│ French:                                                             │
└─────────────────────────────────────────────────────────────────────┘

Model output: "Où est la bibliothèque?"

The model "learned" translation from 2 examples!
No fine-tuning, no gradient descent, just attention.
```

### How In-Context Learning Works

The exact mechanism is still debated, but here are leading theories:

```
THEORIES OF IN-CONTEXT LEARNING
═══════════════════════════════

Theory 1: TASK RECOGNITION
──────────────────────────
The model recognizes the task from examples and applies
pre-learned capabilities.

Pre-training creates "task templates":
  • Q&A template
  • Translation template
  • Sentiment template
  • ...

Examples in prompt → activate relevant template
No actual "learning", just task identification


Theory 2: IMPLICIT GRADIENT DESCENT
───────────────────────────────────
Attention mechanism implements something like gradient descent
within the forward pass.

Examples provide "training data"
Attention computes implicit updates
Output reflects "trained" model

Controversial! But some evidence from:
  - Linear attention ≈ gradient descent mathematically
  - Transformers can implement learning algorithms


Theory 3: INDUCTION HEADS
─────────────────────────
Specific attention patterns (induction heads) that:
  1. Look for patterns in context
  2. Copy from previous occurrences

"A B ... A" → predict "B"

Examples create patterns:
  "cat → chat, dog → chien, bird → "
  Model sees "cat → chat" pattern
  Applies to "bird → " → predicts "oiseau"


LIKELY REALITY:
───────────────
Combination of all three!
  • Task recognition for format
  • Induction heads for simple patterns
  • Something gradient-like for complex adaptation
```

### Factors Affecting In-Context Learning

```
WHAT AFFECTS ICL PERFORMANCE
════════════════════════════

1. NUMBER OF EXAMPLES
   ──────────────────
   Performance vs Examples:
   │
   │                    ┌────────
   │              _____/
   │        _____/
   │  _____/
   │/
   └────────────────────────────→
     0   1   4   8   16  32
          Number of examples

   Usually diminishing returns after 8-16 examples


2. EXAMPLE QUALITY
   ───────────────
   Good examples:                Bad examples:
   ✓ Representative              ✗ Biased sample
   ✓ Correct labels              ✗ Mislabeled
   ✓ Diverse coverage            ✗ All similar
   ✓ Clear formatting            ✗ Ambiguous format


3. EXAMPLE ORDER
   ─────────────
   Order matters! More recent examples have more influence.

   Examples: [A, B, C, D, E] → Query

   E has strongest influence (recency bias)
   Put most relevant/representative examples last


4. FORMAT CONSISTENCY
   ──────────────────
   Bad:
     Input: "text1" Label: positive
     text2 is negative
     "text3" → positive

   Good:
     Text: "text1" | Sentiment: positive
     Text: "text2" | Sentiment: negative
     Text: "text3" | Sentiment:


5. MODEL SIZE
   ──────────
   Larger models = better ICL

   ICL ability vs Model size:
   │
   │                          ★ 175B
   │                    ★
   │              ★
   │        ★
   │   ★
   │ ★
   └─────────────────────────────→
     1B   7B   13B  30B  65B  175B

   Small models (<7B) have weak ICL
```

### In-Context Learning vs Fine-Tuning

```
ICL vs FINE-TUNING COMPARISON
═════════════════════════════

                    In-Context Learning    Fine-Tuning
                    ────────────────────   ────────────
Gradient updates    No                     Yes
Training data       Few examples (~32)     Many examples (~10K+)
Latency             Adds to prompt         No inference overhead
Flexibility         Change task instantly  Need to retrain
Performance ceiling Lower                  Higher
Cost                Per-request tokens     Upfront training
Best for            Rapid prototyping      Production systems


DECISION FLOW:
──────────────

Do you have many labeled examples?
  │
  ├─ No → Use ICL (few-shot prompting)
  │
  └─ Yes → Is this a production system?
              │
              ├─ No → Start with ICL, evaluate
              │
              └─ Yes → Fine-tune for best performance
                       (but prototype with ICL first)
```

---

## 6.10 Summary

### Key Concepts Recap

```
LLM PIPELINE OVERVIEW
═════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. TOKENIZATION                                                    │
│     ─────────────                                                   │
│     Text → Tokens (BPE, WordPiece, SentencePiece)                  │
│     Vocabulary ~30K-100K tokens                                     │
│                                                                     │
│                              ▼                                      │
│                                                                     │
│  2. PRE-TRAINING                                                    │
│     ─────────────                                                   │
│     Next-token prediction on trillions of tokens                    │
│     Creates base model with general capabilities                    │
│     Cost: Millions of dollars                                       │
│                                                                     │
│                              ▼                                      │
│                                                                     │
│  3. FINE-TUNING                                                     │
│     ────────────                                                    │
│     SFT: Learn to follow instructions                               │
│     Methods: Full fine-tuning, LoRA, QLoRA                         │
│     Data: High-quality instruction-response pairs                   │
│                                                                     │
│                              ▼                                      │
│                                                                     │
│  4. ALIGNMENT                                                       │
│     ──────────                                                      │
│     RLHF/DPO: Align with human preferences                          │
│     Makes model helpful, harmless, honest                           │
│                                                                     │
│                              ▼                                      │
│                                                                     │
│  5. DEPLOYMENT                                                      │
│     ───────────                                                     │
│     Quantization: Reduce memory (4-bit)                             │
│     KV Cache: Efficient generation                                  │
│     Batching: High throughput                                       │
│                                                                     │
│                              ▼                                      │
│                                                                     │
│  6. APPLICATION                                                     │
│     ────────────                                                    │
│     Prompting: Zero-shot, few-shot, CoT                            │
│     RAG: Ground in external knowledge                               │
│     Agents: Tool use and planning                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Glossary Terms Covered

```
GLOSSARY
════════

LLM Fundamentals:
─────────────────
• Large Language Model (LLM) - Transformer decoder trained at scale
• Tokenization - Converting text to token IDs
• BPE (Byte-Pair Encoding) - Subword tokenization algorithm
• WordPiece - Alternative tokenization (BERT)
• SentencePiece - Language-agnostic tokenization
• Vocabulary - Set of all tokens the model knows

Training:
─────────
• Pre-training - Initial training on large corpus
• Next-token prediction - The fundamental LLM objective
• Scaling laws - Predictable performance vs scale
• Chinchilla-optimal - ~20 tokens per parameter
• Perplexity - Evaluation metric (lower is better)

Fine-tuning:
────────────
• Full fine-tuning - Update all parameters
• LoRA - Low-rank adapter matrices
• QLoRA - LoRA + 4-bit quantization
• PEFT - Parameter-efficient fine-tuning
• SFT - Supervised fine-tuning on instructions

Alignment:
──────────
• RLHF - RL from Human Feedback
• Reward model - Scores response quality
• PPO - RL algorithm for policy optimization
• DPO - Direct preference optimization (simpler than RLHF)
• KL penalty - Keeps model close to reference

Prompting:
──────────
• Zero-shot - No examples in prompt
• Few-shot - Examples in prompt
• Chain-of-Thought (CoT) - Step-by-step reasoning
• Self-consistency - Multiple samples, majority vote
• In-context learning - Learning from prompt examples

Inference:
──────────
• Temperature - Controls randomness
• Top-k/Top-p - Sampling strategies
• KV Cache - Cached keys/values for efficiency
• Quantization - Reduced precision (4-bit, 8-bit)
• Speculative decoding - Draft with small model, verify with large

Issues:
───────
• Hallucination - Generating false information
• Knowledge cutoff - Training data date limit
• Prompt injection - Malicious prompt manipulation
```

### What's Next

Module 7 covers **Generative Models** beyond text: Variational Autoencoders (VAE),
Generative Adversarial Networks (GAN), and Diffusion models for image generation.

---

## References

### Key Papers

**LLM Foundations:**
- Radford et al., ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
- Brown et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (GPT-3, 2020)
- Hoffmann et al., ["Training Compute-Optimal Large Language Models"](https://arxiv.org/abs/2203.15556) (Chinchilla, 2022)
- Touvron et al., ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971) (2023)

**Tokenization:**
- Sennrich et al., ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) (BPE, 2016)
- Wu et al., ["Google's Neural Machine Translation System"](https://arxiv.org/abs/1609.08144) (WordPiece, 2016)
- Kudo & Richardson, ["SentencePiece"](https://arxiv.org/abs/1808.06226) (2018)

**Fine-tuning:**
- Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (2021)
- Dettmers et al., ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314) (2023)
- Zhou et al., ["LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206) (2023)

**Alignment:**
- Ouyang et al., ["Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- Rafailov et al., ["Direct Preference Optimization"](https://arxiv.org/abs/2305.18290) (DPO, 2023)
- Schulman et al., ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) (PPO, 2017)

**Prompting:**
- Wei et al., ["Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903) (2022)
- Wang et al., ["Self-Consistency Improves Chain of Thought Reasoning"](https://arxiv.org/abs/2203.11171) (2022)
- Kojima et al., ["Large Language Models are Zero-Shot Reasoners"](https://arxiv.org/abs/2205.11916) (2022)

**Inference:**
- Leviathan et al., ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) (2022)
- Frantar et al., ["GPTQ: Accurate Post-Training Quantization"](https://arxiv.org/abs/2210.17323) (2022)
- Lin et al., ["AWQ: Activation-aware Weight Quantization"](https://arxiv.org/abs/2306.00978) (2023)

### Courses and Tutorials
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar)
- [Lilian Weng's LLM Blog Posts](https://lilianweng.github.io/)

### Tools and Libraries
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/)
- [vLLM](https://vllm.readthedocs.io/) - Fast LLM serving
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU/GPU inference
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/) - RLHF/DPO training

