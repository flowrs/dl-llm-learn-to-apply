# Module 6: Large Language Models

## Learning Objectives

By the end of this module, you will understand:
- Tokenization and vocabulary construction
- Pre-training objectives and scaling laws
- Fine-tuning techniques (Full, LoRA, QLoRA)
- RLHF and alignment methods
- Prompting strategies and in-context learning
- Inference optimization techniques

---

## 6.1 What Makes a Language Model "Large"?

### The LLM Recipe

```
LLM = Transformer decoder + Massive data + Massive compute + Scaling
```

**Scale comparison**:
| Model | Parameters | Training Tokens |
|-------|-----------|-----------------|
| GPT-2 (2019) | 1.5B | 40B |
| GPT-3 (2020) | 175B | 300B |
| GPT-4 (2023) | ~1.8T (est.) | ~13T (est.) |
| Llama 2 (2023) | 70B | 2T |

### Key Insight: Emergent Capabilities

At sufficient scale, LLMs exhibit abilities not explicitly trained:
- Few-shot learning
- Chain-of-thought reasoning
- Code generation
- Following complex instructions

---

## 6.2 Tokenization

### Why Tokenize?

Neural networks need numbers, not text. Tokenization converts text to token IDs.

### Subword Tokenization (BPE)

**Byte-Pair Encoding** training algorithm:
1. Start with character vocabulary
2. Count all adjacent pairs in corpus
3. Merge most frequent pair into new token
4. Repeat until desired vocabulary size

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, how are you?"
tokens = tokenizer.encode(text)  # [15496, 11, 703, 389, 345, 30]
```

### Token Quirks

- Spaces matter: `" hello"` ≠ `"hello"` (different tokens!)
- Numbers tokenize inconsistently: `"1000"` might be `['10', '00']`
- Common words = 1 token, rare words = multiple tokens

---

## 6.3 Pre-training

### The Objective: Next Token Prediction

```
Input:  "The cat sat on the"
Target: "mat"

Loss = -log P(mat | The cat sat on the)
```

### Scaling Laws

Performance scales predictably:
```
Loss ∝ 1/N^α × 1/D^β × 1/C^γ
```

**Chinchilla insight**: Optimal is ~20 tokens per parameter.

---

## 6.4 Fine-tuning

### Full Fine-tuning

Update all parameters. Problem: 70B model = 140GB+ memory.

### LoRA (Low-Rank Adaptation)

Add small trainable matrices:
```
W' = W + ΔW where ΔW = A × B
A: [d × r], B: [r × d], r << d
```

Train 0.1-1% of parameters, similar performance!

### QLoRA

LoRA + 4-bit quantization → Fine-tune 65B on single GPU.

---

## 6.5 RLHF (Reinforcement Learning from Human Feedback)

### The Pipeline

```
1. SFT: Train on demonstrations
2. Reward Model: Learn to score responses
3. PPO: Optimize policy using rewards
```

### DPO (Direct Preference Optimization)

Skip reward model, optimize preferences directly:
```python
loss = -log_sigmoid(β * (log_ratio_chosen - log_ratio_rejected))
```

---

## 6.6 Prompting Techniques

### Zero-Shot
Ask directly without examples.

### Few-Shot
Provide examples in prompt.

### Chain-of-Thought
"Let's think step by step" → Model shows reasoning.

### Self-Consistency
Generate multiple answers, take majority vote.

---

## 6.7 Inference Optimization

### Temperature and Sampling

**Temperature** controls randomness:
```
P(token) = softmax(logits / T)
T → 0: Greedy (deterministic)
T → ∞: Uniform (random)
```

**Top-k Sampling**: Only consider k highest probability tokens.

**Top-p (Nucleus)**: Consider tokens until cumulative probability reaches p.

### KV Cache

Cache key/value tensors from previous tokens:
- Without: O(n²) per token
- With: O(n) per token

### Quantization

Reduce precision: fp32 → fp16 → int8 → int4
- 4-bit: 4× memory reduction, ~1% quality loss

### Speculative Decoding

Use small model to draft, large model to verify:
```
Draft model: generates k tokens quickly
Main model: verifies/corrects in parallel
```

---

## 6.8 Hallucination

### The Problem

LLMs generate plausible-sounding but false information.

### Causes
- Training on internet text (includes errors)
- Optimized for fluency, not factuality
- No grounding in real world

### Mitigations
- RAG: Retrieve facts before generating
- Fine-tune on factual data
- Add citations/sources
- Confidence calibration

---

## 6.9 In-Context Learning

### The Phenomenon

LLMs can learn new tasks from examples in the prompt:

```
Input: "sentiment: I love this! → positive
sentiment: This is terrible → negative
sentiment: Not bad at all →"

Output: "positive"
```

No gradient updates! The model "learns" within forward pass.

### Why It Works (Theories)

1. Pre-training creates implicit task templates
2. Attention retrieves relevant examples
3. Model implements gradient descent internally (controversial)

---

## 6.10 Summary

### Key Concepts

1. **Tokenization** converts text to numbers (BPE is common)
2. **Pre-training** on next-token prediction at massive scale
3. **Fine-tuning** specializes models (LoRA for efficiency)
4. **RLHF/DPO** aligns models with human preferences
5. **Prompting** techniques extract capabilities without training
6. **Inference optimization** makes deployment practical

### Glossary Terms Covered

- Large Language Model (LLM)
- Token, Tokenization, BPE, Vocabulary
- Pre-training, Next-token prediction
- Perplexity
- Fine-tuning, LoRA, QLoRA, PEFT
- RLHF, PPO, DPO, Reward Model
- Prompt, Zero-shot, Few-shot
- Chain-of-Thought, In-context Learning
- Temperature, Top-k, Top-p
- Hallucination
- Quantization, KV Cache

### What's Next

Module 7 covers **Generative Models** beyond text: VAEs, GANs, and Diffusion models.

---

## References

- Brown et al., "Language Models are Few-Shot Learners" (GPT-3)
- Ouyang et al., "Training language models to follow instructions" (InstructGPT)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- Rafailov et al., "Direct Preference Optimization"
- CS224N: Lectures on LLMs, RLHF, Prompting
