# Module 8: Advanced Topics & Production

## Learning Objectives

By the end of this module, you will understand:
- Retrieval-Augmented Generation (RAG)
- AI agents and tool use
- Model deployment and serving
- Evaluation and benchmarking
- Ethical considerations and limitations

---

## 8.1 Retrieval-Augmented Generation (RAG)

### The Problem

LLMs have knowledge cutoffs and hallucinate. How to give them access to current, factual information?

### RAG Architecture

```
User Query
    ↓
[Embed query] → [Search vector DB] → Retrieve relevant documents
                                              ↓
    ┌─────────────────────────────────────────┘
    ↓
[Combine query + retrieved context]
    ↓
[LLM generates answer grounded in context]
    ↓
Response
```

### Components

**1. Document Processing**
```python
# Split documents into chunks
chunks = text_splitter.split(documents)

# Embed chunks
embeddings = embedding_model.encode(chunks)

# Store in vector database
vector_db.add(embeddings, chunks)
```

**2. Retrieval**
```python
# Embed query
query_embedding = embedding_model.encode(query)

# Find similar chunks
results = vector_db.search(query_embedding, k=5)
```

**3. Generation**
```python
context = "\n".join([r.text for r in results])
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
response = llm.generate(prompt)
```

### When to Use RAG

- Private/proprietary data
- Frequently updated information
- Domain-specific knowledge
- Reducing hallucinations
- Citations/sources needed

---

## 8.2 AI Agents

### What is an Agent?

An AI system that can:
1. Reason about goals
2. Plan sequences of actions
3. Use tools to interact with the world
4. Observe results and adapt

```
User: "Book me a flight to NYC next Friday"
    ↓
Agent reasons: Need to search flights, compare options, make booking
    ↓
Agent acts: [Search tool] → [Select flight] → [Booking API]
    ↓
Agent reports: "Booked flight UA123 departing 3pm for $299"
```

### Tool Use

LLMs can call functions/APIs:

```python
tools = [
    {
        "name": "search_web",
        "description": "Search the internet for information",
        "parameters": {"query": "string"}
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {"expression": "string"}
    }
]

# LLM decides which tool to call
response = llm.generate(query, tools=tools)
# Response: {"tool": "search_web", "query": "weather in NYC"}

# Execute tool
result = tools["search_web"](query="weather in NYC")

# Continue with result
final_response = llm.generate(f"Search result: {result}\n\nOriginal query: {query}")
```

### ReAct Pattern

**Re**asoning + **Act**ing in alternating steps:

```
Thought: I need to find the current stock price
Action: search("AAPL stock price")
Observation: AAPL is trading at $178.50
Thought: Now I need to calculate the market cap
Action: calculate("178.50 * 15.8 billion")
Observation: 2.82 trillion
Thought: I have enough information to answer
Answer: Apple's market cap is approximately $2.82 trillion
```

### Challenges

- **Error propagation**: Mistakes compound
- **Hallucinated actions**: LLM invents tools
- **Security**: Agents with real-world access are risky
- **Cost**: Many LLM calls per task

---

## 8.3 Model Deployment

### Serving Infrastructure

```
Users → [Load Balancer] → [Inference Servers] → [Model]
                              ↓
                         [GPU cluster]
```

### Key Metrics

- **Latency**: Time to generate response
- **Throughput**: Requests per second
- **Cost**: $/1M tokens

### Optimization Techniques

**Batching**: Process multiple requests together
```python
# Without batching: 10 requests = 10 forward passes
# With batching: 10 requests = 1 forward pass (much faster)
```

**Quantization**: Reduce precision
```
FP32 (4 bytes) → FP16 (2 bytes) → INT8 (1 byte) → INT4 (0.5 bytes)
```

**KV Cache**: Store computed key/values for reuse

**Speculative Decoding**: Draft with small model, verify with large

### Serving Frameworks

| Framework | Use Case |
|-----------|----------|
| vLLM | High-throughput LLM serving |
| TensorRT-LLM | NVIDIA GPU optimization |
| Triton | General model serving |
| Ollama | Local LLM deployment |

---

## 8.4 Evaluation & Benchmarks

### Why Evaluation is Hard

- Many capabilities to measure
- Easy to game benchmarks
- Human preferences are subjective
- Real-world performance ≠ benchmark performance

### Common Benchmarks

**Language Understanding**:
- GLUE/SuperGLUE: NLU tasks
- MMLU: Multi-task multiple choice
- HellaSwag: Commonsense reasoning

**Code**:
- HumanEval: Function completion
- MBPP: Python programming

**Math**:
- GSM8K: Grade school math
- MATH: Competition math

**Safety**:
- TruthfulQA: Factuality
- Toxicity benchmarks

### Evaluation Approaches

**1. Automated Metrics**
```python
# Perplexity
ppl = exp(avg_loss)

# BLEU (for translation)
bleu = compute_bleu(generated, reference)

# Accuracy (for classification)
acc = correct / total
```

**2. LLM-as-Judge**
```python
judge_prompt = f"""
Rate this response on a scale of 1-5:
Question: {question}
Response: {response}
"""
score = judge_llm.generate(judge_prompt)
```

**3. Human Evaluation**
- Gold standard but expensive
- Use for final validation

### Best Practices

- Use held-out test sets
- Report confidence intervals
- Test on diverse inputs
- Don't over-optimize for benchmarks

---

## 8.5 Ethics & Safety

### Key Concerns

**1. Bias and Fairness**
- Models reflect training data biases
- Can discriminate against groups
- Need diverse training data and evaluation

**2. Misinformation**
- Fluent but false content
- Deepfakes and synthetic media
- Erosion of trust

**3. Privacy**
- Training data may leak
- Memorization of personal info
- Inference from outputs

**4. Misuse**
- Spam and manipulation
- Cyberattacks
- Non-consensual content

### Mitigation Strategies

**Technical**:
- RLHF for alignment
- Content filtering
- Watermarking generated content
- Differential privacy in training

**Organizational**:
- Red teaming
- Responsible disclosure
- Access controls
- Usage policies

**Societal**:
- Regulation and standards
- Transparency requirements
- Education about AI limitations

---

## 8.6 Learning Paradigms Summary

### Supervised Learning
```
(x, y) pairs → Learn f: x → y
```
- Classification, regression
- Requires labeled data

### Self-Supervised Learning
```
x → Create (x, y) automatically → Learn f
```
- Next-token prediction
- Masked language modeling
- Contrastive learning

### Reinforcement Learning
```
State → Action → Reward → Update policy
```
- RLHF for alignment
- Game playing

### Transfer Learning
```
Pre-train on task A → Fine-tune on task B
```
- Most common paradigm today
- Pre-trained models + adaptation

### Few-Shot / Zero-Shot
```
Task description + examples → Output
```
- No gradient updates
- In-context learning

---

## 8.7 The Full Pipeline

Putting it all together:

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION                        │
│  Web scraping, datasets, human annotation, synthetic data   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       PRE-TRAINING                          │
│  Next-token prediction on massive corpus                    │
│  Result: Base model with general capabilities               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        FINE-TUNING                          │
│  SFT on instructions, RLHF/DPO for alignment               │
│  Result: Helpful, harmless assistant                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT                           │
│  Quantization, serving infrastructure, monitoring           │
│  Result: Production API                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       APPLICATION                           │
│  RAG for knowledge, agents for actions, prompting for tasks │
│  Result: End-user product                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 8.8 Summary

### Key Concepts

1. **RAG** grounds LLMs in external knowledge
2. **Agents** extend LLMs with tools and planning
3. **Deployment** requires optimization for latency/cost
4. **Evaluation** is multi-faceted and imperfect
5. **Ethics** must be considered throughout

### Glossary Terms Covered

- Retrieval-Augmented Generation (RAG)
- Vector Database
- Embedding
- Agent
- Tool Use
- Inference
- Latency
- Throughput
- Benchmark
- Bias
- Alignment
- Transfer Learning
- Self-Supervised Learning
- Contrastive Learning
- Knowledge Distillation

---

## Course Conclusion

### What You've Learned

1. **Foundations**: Neurons, networks, tensors
2. **Training**: Backprop, SGD, regularization
3. **CNNs**: Convolution, architectures, transfer learning
4. **Sequences**: RNNs, LSTM, vanishing gradients
5. **Attention**: Self-attention, Transformers
6. **LLMs**: Pre-training, fine-tuning, prompting
7. **Generation**: VAE, GAN, Diffusion
8. **Advanced**: RAG, agents, deployment, ethics

### Where to Go From Here

- **Practice**: Build projects with real data
- **Papers**: Read seminal works and recent research
- **Community**: Join ML communities and discussions
- **Specialize**: Pick a domain (NLP, vision, etc.)
- **Stay current**: Field moves fast!

### Key Takeaways

```
1. Deep learning = differentiable function composition + gradient descent
2. Scale (data, compute, parameters) drives capability
3. Pre-training + fine-tuning is the dominant paradigm
4. Attention/Transformers unified vision and language
5. Evaluation and safety are unsolved challenges
```

---

## References

- Lewis et al., "Retrieval-Augmented Generation"
- Yao et al., "ReAct: Synergizing Reasoning and Acting"
- CS224N: Advanced topics lectures
- Anthropic, OpenAI safety research
- ML Commons benchmarks
