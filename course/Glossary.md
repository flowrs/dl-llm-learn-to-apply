# Deep Learning Glossary

A comprehensive glossary of terms used throughout the course, organized alphabetically with cross-references.

---

## Quick Navigation

[A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [J](#j) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [X](#x) | [Z](#z)

---

## A

### Activation Function
A non-linear function applied to the output of neurons to introduce non-linearity into the network. Common examples include *ReLU*, *sigmoid*, and *tanh*. Without activation functions, a deep network would collapse to a single linear transformation.

**Formula (ReLU)**: f(x) = max(0, x)

**See also**: *ReLU*, *Sigmoid*, *Tanh*, *GELU*

---

### Adam (Adaptive Moment Estimation)
An optimization algorithm that combines *momentum* and *RMSprop*. Maintains running averages of both gradients (first moment) and squared gradients (second moment), with bias correction.

**Formula**:
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)  # bias correction
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**Default hyperparameters**: β₁=0.9, β₂=0.999, ε=1e-8

**See also**: *SGD*, *Momentum*, *RMSprop*, *AdamW*

---

### AdamW
A variant of *Adam* that decouples weight decay from the gradient update. Standard Adam applies weight decay to the gradient, while AdamW applies it directly to the weights.

**See also**: *Adam*, *Weight Decay*, *L2 Regularization*

---

### Alignment
The process of training AI systems to behave according to human values and intentions. In LLMs, typically achieved through *RLHF* or *DPO*.

**See also**: *RLHF*, *DPO*, *HHH*

---

### Attention
A mechanism that allows models to focus on relevant parts of the input when producing each part of the output. Computes a weighted sum of values based on query-key similarity.

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Types**: Self-attention, Cross-attention, Multi-head attention

**See also**: *Self-Attention*, *Multi-Head Attention*, *Transformer*

---

### Autoencoder
A neural network trained to reconstruct its input through a bottleneck layer. Learns compressed representations of data.

```
Input → Encoder → Latent Code (z) → Decoder → Reconstruction
```

**See also**: *VAE*, *Latent Space*

---

### Autoregressive Model
A model that generates sequences by predicting one element at a time, conditioned on previous elements.

**Formula**: P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)

**Examples**: GPT, language models

**See also**: *Language Model*, *GPT*

---

## B

### Backpropagation
The algorithm for computing gradients of the loss with respect to all parameters by applying the chain rule backwards through the network.

**Key insight**: Gradients flow backwards from loss to inputs, computing local gradients at each operation.

**See also**: *Chain Rule*, *Gradient Descent*, *Computational Graph*

---

### Batch Normalization (BatchNorm)
A technique that normalizes layer inputs across the batch dimension to stabilize training. Learns scale (γ) and shift (β) parameters.

**Formula**:
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γ * x̂ + β
```

**See also**: *Layer Normalization*, *Normalization*

---

### Batch Size
The number of training examples used in one forward/backward pass. Larger batches provide more stable gradients but require more memory.

**Trade-offs**:
- Larger batch: More stable gradients, better GPU utilization
- Smaller batch: More noise (regularization), less memory

**See also**: *Mini-batch Gradient Descent*, *Stochastic Gradient Descent*

---

### BERT (Bidirectional Encoder Representations from Transformers)
A pre-trained language model that uses bidirectional context (looks at words both before and after). Trained with masked language modeling (MLM) and next sentence prediction (NSP).

**See also**: *Transformer*, *MLM*, *Pre-training*

---

### Bias (Parameter)
An additive parameter in neural network layers that allows the model to shift the output.

**Formula**: y = Wx + b

**Not to be confused with**: Statistical bias (systematic error)

---

### BPE (Byte Pair Encoding)
A subword tokenization algorithm that iteratively merges the most frequent character pairs to build a vocabulary.

**Process**:
1. Start with character vocabulary
2. Count all adjacent pairs
3. Merge most frequent pair
4. Repeat until desired vocabulary size

**See also**: *Tokenization*, *WordPiece*, *SentencePiece*

---

## C

### Causal Mask
A mask used in autoregressive models to prevent tokens from attending to future tokens.

```
Mask (for sequence length 4):
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**See also**: *Attention*, *Autoregressive Model*

---

### Chain Rule
The calculus rule for computing derivatives of composed functions. Foundation of *backpropagation*.

**Formula**: df/dx = df/dg * dg/dx

**See also**: *Backpropagation*, *Gradient*

---

### Chain-of-Thought (CoT)
A prompting technique where the model is encouraged to show intermediate reasoning steps before giving the final answer.

**Example prompt**: "Let's think step by step..."

**See also**: *Prompting*, *In-Context Learning*

---

### Chunking
The process of splitting documents into smaller pieces for *RAG* systems. Important for managing context length and retrieval relevance.

**Strategies**: Fixed-size, overlapping, semantic, hierarchical

**See also**: *RAG*, *Embedding*

---

### Classification
A task where the model predicts which category an input belongs to.

**Types**: Binary (2 classes), Multi-class (>2 mutually exclusive), Multi-label (multiple simultaneous)

**See also**: *Cross-Entropy Loss*, *Softmax*

---

### CNN (Convolutional Neural Network)
A neural network architecture designed for processing grid-like data (images), using *convolution* operations to detect local patterns.

**Key components**: Convolution layers, pooling layers, fully connected layers

**See also**: *Convolution*, *Pooling*, *Receptive Field*

---

### Contrastive Learning
A self-supervised learning approach that learns representations by bringing similar examples closer and pushing dissimilar examples apart.

**Loss (InfoNCE)**:
```
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

**Examples**: SimCLR, CLIP, MoCo

**See also**: *Self-Supervised Learning*, *Embedding*

---

### Convolution
A mathematical operation that slides a filter/kernel over input to detect local patterns. In CNNs, learns filters that detect edges, textures, objects.

**Formula (2D)**:
```
(I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] × K[m, n]
```

**See also**: *CNN*, *Filter*, *Stride*, *Padding*

---

### Cross-Attention
Attention where queries come from one sequence and keys/values come from another. Used in encoder-decoder models.

**Example**: In translation, decoder queries attend to encoder outputs

**See also**: *Self-Attention*, *Attention*, *Encoder-Decoder*

---

### Cross-Entropy Loss
The standard loss function for classification tasks. Measures the difference between predicted probability distribution and true distribution.

**Formula**:
```
L = -Σ y_true * log(y_pred)
  = -log(p_correct_class)  # for one-hot labels
```

**See also**: *Softmax*, *Loss Function*, *Classification*

---

## D

### DDPM (Denoising Diffusion Probabilistic Model)
A generative model that learns to reverse a gradual noising process. Trained to predict the noise added at each timestep.

**See also**: *Diffusion Model*, *Generative Model*

---

### Decoder
The part of a model that generates output from a learned representation. In Transformers, uses *causal masking* for autoregressive generation.

**See also**: *Encoder*, *Encoder-Decoder*, *Transformer*

---

### Deep Learning
A subfield of machine learning using neural networks with multiple layers (deep) to learn hierarchical representations of data.

---

### Diffusion Model
A generative model that learns to reverse a gradual noising process. Generates samples by starting from noise and iteratively denoising.

**Forward process**: Gradually add noise to data
**Reverse process**: Learn to denoise (the model)

**Examples**: DDPM, Stable Diffusion, DALL-E 2

**See also**: *DDPM*, *Generative Model*, *Stable Diffusion*

---

### Discriminator
In *GANs*, the network that tries to distinguish real data from generated data.

**See also**: *GAN*, *Generator*

---

### Dropout
A regularization technique that randomly sets a fraction of activations to zero during training. Prevents co-adaptation of neurons.

**Formula (training)**: y = x * mask / (1 - p), where mask ~ Bernoulli(1-p)
**Formula (inference)**: y = x

**See also**: *Regularization*, *Overfitting*

---

### DPO (Direct Preference Optimization)
An alignment method that directly optimizes a policy from preference data, without training a separate reward model.

**See also**: *RLHF*, *Alignment*, *PPO*

---

## E

### Eigenvalue / Eigenvector
For a matrix A, eigenvector v satisfies Av = λv, where λ is the eigenvalue. Important for understanding gradient flow and matrix properties.

**See also**: *Linear Algebra*, *Gradient Flow*

---

### ELBO (Evidence Lower Bound)
The training objective for *VAEs*. Maximizing ELBO maximizes a lower bound on the data likelihood.

**Formula**:
```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
     = Reconstruction - KL Divergence
```

**See also**: *VAE*, *KL Divergence*

---

### Embedding
A dense vector representation of discrete objects (words, tokens, items) in a continuous space. Learned representations that capture semantic similarity.

**Examples**: Word2Vec, GloVe, learned token embeddings

**See also**: *Word Embedding*, *Representation Learning*

---

### Emergent Capabilities
Capabilities that appear suddenly at certain model scales, not present in smaller models. Examples include in-context learning, chain-of-thought reasoning.

**See also**: *Scaling Laws*, *LLM*

---

### Encoder
The part of a model that processes input into a learned representation. In Transformers, uses bidirectional attention.

**See also**: *Decoder*, *Encoder-Decoder*, *Representation Learning*

---

### Encoder-Decoder
An architecture with separate encoder and decoder components. Encoder processes input; decoder generates output attending to encoder states.

**Uses**: Translation, summarization, image captioning

**See also**: *Seq2Seq*, *Transformer*, *Cross-Attention*

---

### Epoch
One complete pass through the entire training dataset.

**See also**: *Training*, *Iteration*

---

## F

### Feature Map
The output of a convolutional layer. Each filter produces one feature map that highlights where the filter's pattern is detected.

**See also**: *CNN*, *Convolution*, *Filter*

---

### Few-Shot Learning
Learning from a very small number of examples. In LLMs, achieved through *in-context learning* with examples in the prompt.

**See also**: *In-Context Learning*, *Zero-Shot Learning*, *Prompting*

---

### Filter (Kernel)
A small matrix of learnable weights used in *convolution* to detect patterns. Different filters detect different features (edges, textures, etc.).

**See also**: *Convolution*, *CNN*, *Feature Map*

---

### Fine-tuning
Adapting a pre-trained model to a specific task or domain by continuing training on task-specific data.

**Types**: Full fine-tuning, *LoRA*, *QLoRA*, *Adapter*

**See also**: *Pre-training*, *Transfer Learning*, *LoRA*

---

### Forward Pass
Computing the output of a neural network given an input. Values flow forward through the network.

**See also**: *Backward Pass*, *Inference*

---

## G

### GAN (Generative Adversarial Network)
A generative model framework where a *generator* and *discriminator* are trained adversarially. Generator tries to fool discriminator; discriminator tries to detect fakes.

**Min-max objective**:
```
min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
```

**See also**: *Generator*, *Discriminator*, *Generative Model*

---

### GELU (Gaussian Error Linear Unit)
An activation function used in modern Transformers. Smoother than ReLU.

**Formula**: GELU(x) = x * Φ(x), where Φ is the Gaussian CDF

**Approximation**: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

**See also**: *Activation Function*, *ReLU*

---

### Generator
In *GANs*, the network that generates fake data from random noise.

**See also**: *GAN*, *Discriminator*

---

### Generative Model
A model that learns the data distribution and can generate new samples. Types include VAEs, GANs, diffusion models, autoregressive models.

**See also**: *GAN*, *VAE*, *Diffusion Model*, *Autoregressive Model*

---

### Gradient
The vector of partial derivatives of a function. Points in the direction of steepest increase.

**In deep learning**: ∇L = [∂L/∂θ₁, ∂L/∂θ₂, ...]

**See also**: *Gradient Descent*, *Backpropagation*

---

### Gradient Clipping
A technique to prevent exploding gradients by limiting gradient magnitude.

**Methods**:
- Clip by value: g = clip(g, -threshold, threshold)
- Clip by norm: g = g * threshold / ||g|| if ||g|| > threshold

**See also**: *Exploding Gradients*, *Vanishing Gradients*

---

### Gradient Descent
The optimization algorithm that updates parameters in the direction opposite to the gradient to minimize loss.

**Formula**: θ = θ - α * ∇L(θ)

**Variants**: SGD, mini-batch, with momentum, Adam

**See also**: *SGD*, *Adam*, *Optimization*

---

### GPT (Generative Pre-trained Transformer)
A family of autoregressive language models using Transformer decoders. Pre-trained on next-token prediction.

**See also**: *Transformer*, *Autoregressive Model*, *LLM*

---

### GRU (Gated Recurrent Unit)
A simplified variant of *LSTM* with two gates (reset and update) instead of three.

**See also**: *LSTM*, *RNN*, *Gate*

---

## H

### Hallucination
When an LLM generates false or fabricated information that sounds plausible. A major challenge in LLM reliability.

**Mitigation**: RAG, grounding, confidence calibration

**See also**: *RAG*, *Factual Accuracy*

---

### He Initialization
Weight initialization scaled by √(2/n_in), designed for ReLU activations.

**Formula**: W ~ N(0, √(2/n_in))

**See also**: *Xavier Initialization*, *Weight Initialization*

---

### HHH (Helpful, Harmless, Honest)
A framework for AI alignment goals. Models should be helpful to users, avoid causing harm, and be truthful.

**See also**: *Alignment*, *RLHF*

---

### Hidden Layer
A layer between input and output layers. Learns intermediate representations.

**See also**: *Layer*, *Deep Learning*

---

### Hidden State
In RNNs, the vector that carries information from previous timesteps. Updated at each timestep.

**Formula**: h_t = f(W_xh * x_t + W_hh * h_{t-1} + b)

**See also**: *RNN*, *LSTM*

---

### Hyperparameter
A parameter set before training (not learned). Examples: learning rate, batch size, number of layers.

**See also**: *Learning Rate*, *Batch Size*

---

## I

### In-Context Learning (ICL)
The ability of LLMs to learn from examples provided in the prompt, without any parameter updates.

**Example**:
```
Translate English to French:
sea otter => loutre de mer
hello => bonjour
cat =>
```

**See also**: *Few-Shot Learning*, *Prompting*, *LLM*

---

### Inference
Using a trained model to make predictions. Forward pass only, no gradient computation.

**See also**: *Forward Pass*, *Training*

---

### Information Bottleneck
A constraint that forces models to learn compressed representations. Appears in autoencoders and seq2seq models.

**See also**: *Autoencoder*, *Seq2Seq*, *Attention*

---

### Instruction Tuning
Fine-tuning LLMs on datasets of instructions and desired responses. Teaches models to follow instructions.

**Examples**: InstructGPT, FLAN

**See also**: *Fine-tuning*, *RLHF*

---

## J

### Jacobian
The matrix of all first-order partial derivatives. For f: R^n → R^m, the Jacobian is m × n.

**Used in**: Backpropagation through layers, normalizing flows

**See also**: *Gradient*, *Backpropagation*

---

## K

### K (Key)
In attention, the key vectors are matched against queries to determine attention weights.

**Intuition**: Keys are like "labels" that describe what information is available.

**See also**: *Attention*, *Query*, *Value*

---

### KL Divergence (Kullback-Leibler Divergence)
A measure of how one probability distribution differs from another.

**Formula**: KL(P || Q) = Σ P(x) log(P(x) / Q(x))

**Properties**: Always ≥ 0; = 0 iff P = Q; not symmetric

**See also**: *ELBO*, *VAE*, *Cross-Entropy*

---

### KV Cache
A technique for efficient autoregressive generation that caches key and value vectors from previous tokens.

**Benefit**: Reduces computation from O(n²) to O(n) per token

**See also**: *Attention*, *Inference Optimization*

---

## L

### L1 Regularization (Lasso)
Regularization that adds the sum of absolute weights to the loss. Encourages sparsity.

**Loss**: L_total = L_data + λ * Σ|w|

**See also**: *L2 Regularization*, *Regularization*

---

### L2 Regularization (Ridge, Weight Decay)
Regularization that adds the sum of squared weights to the loss. Encourages small weights.

**Loss**: L_total = L_data + λ * Σw²

**See also**: *L1 Regularization*, *Regularization*, *AdamW*

---

### Language Model
A model that assigns probabilities to sequences of words/tokens. Typically trained to predict the next token.

**Objective**: Maximize P(w₁, w₂, ..., wₙ)

**See also**: *LLM*, *Perplexity*, *Autoregressive Model*

---

### Latent Space
The lower-dimensional space learned by models like autoencoders. Captures essential features of the data.

**See also**: *Autoencoder*, *VAE*, *Representation Learning*

---

### Layer Normalization (LayerNorm)
Normalization applied across features for each example (instead of across batch). Standard in Transformers.

**Formula**: y = γ * (x - μ_layer) / √(σ²_layer + ε) + β

**See also**: *Batch Normalization*, *Normalization*, *Transformer*

---

### Learning Rate
The step size for gradient descent updates. One of the most important hyperparameters.

**Too high**: Training diverges
**Too low**: Training is slow

**See also**: *Gradient Descent*, *Learning Rate Schedule*

---

### Learning Rate Schedule
A strategy for changing learning rate during training.

**Types**: Step decay, exponential decay, cosine annealing, warmup

**See also**: *Learning Rate*, *Training*

---

### LLM (Large Language Model)
A language model with billions of parameters trained on massive text corpora. Exhibits emergent capabilities like in-context learning.

**Examples**: GPT-4, Claude, LLaMA, PaLM

**See also**: *Language Model*, *Transformer*, *Emergent Capabilities*

---

### LoRA (Low-Rank Adaptation)
An efficient fine-tuning method that adds trainable low-rank matrices to frozen model weights.

**Formula**: W' = W + BA, where B is d×r, A is r×d, r << d

**Benefits**: Much fewer trainable parameters, no inference overhead

**See also**: *Fine-tuning*, *QLoRA*

---

### Loss Function
A function that measures how well the model's predictions match the target. Training minimizes this.

**Examples**: MSE (regression), Cross-entropy (classification), Contrastive

**See also**: *Cross-Entropy Loss*, *MSE*, *Optimization*

---

### LSTM (Long Short-Term Memory)
An RNN architecture with gates that control information flow, designed to capture long-range dependencies.

**Gates**: Forget gate, input gate, output gate
**Cell state**: Long-term memory that flows through time

**See also**: *RNN*, *GRU*, *Vanishing Gradients*

---

## M

### Masked Language Modeling (MLM)
A pre-training objective where random tokens are masked and the model predicts them. Used by BERT.

**Difference from CLM**: Bidirectional context (sees both left and right)

**See also**: *BERT*, *Pre-training*

---

### Max Pooling
A pooling operation that takes the maximum value in each region. Provides translation invariance.

**See also**: *Pooling*, *CNN*, *Average Pooling*

---

### Mini-batch
A subset of training data used for one gradient update. Balance between full batch and single example.

**See also**: *Batch Size*, *SGD*

---

### MLM
See *Masked Language Modeling*

---

### MLP (Multi-Layer Perceptron)
A neural network with fully connected layers. Also called feedforward network.

**See also**: *Fully Connected*, *Neural Network*

---

### Mode Collapse
A failure mode in GAN training where the generator produces limited variety of outputs.

**See also**: *GAN*, *Training Instability*

---

### Momentum
A technique that accumulates past gradients to smooth updates and accelerate training.

**Formula**:
```
v_t = β * v_{t-1} + g_t
θ_t = θ_{t-1} - α * v_t
```

**See also**: *SGD*, *Adam*, *Gradient Descent*

---

### MSE (Mean Squared Error)
A loss function for regression that computes average squared difference.

**Formula**: L = (1/n) Σ(y_pred - y_true)²

**See also**: *Loss Function*, *Regression*

---

### Multi-Head Attention
Attention with multiple parallel attention operations (heads), each with different learned projections.

**Formula**:
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) W^O
where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

**See also**: *Attention*, *Transformer*

---

## N

### Next Token Prediction
The training objective for autoregressive language models. Predict the next token given previous tokens.

**See also**: *Language Model*, *Autoregressive Model*

---

### Normalization
Techniques to standardize activations or inputs. Improves training stability.

**Types**: Batch norm, layer norm, group norm, instance norm

**See also**: *Batch Normalization*, *Layer Normalization*

---

### NTP
See *Next Token Prediction*

---

## O

### One-Hot Encoding
A representation where each category is represented by a binary vector with a 1 in one position.

**Example**: If vocabulary has 5 words, "cat" at position 2 → [0, 0, 1, 0, 0]

**See also**: *Embedding*, *Classification*

---

### Optimization
The process of finding parameter values that minimize the loss function.

**See also**: *Gradient Descent*, *Loss Function*, *Adam*

---

### Overfitting
When a model performs well on training data but poorly on new data. Memorizes rather than generalizes.

**Solutions**: Regularization, dropout, more data, early stopping

**See also**: *Regularization*, *Dropout*, *Generalization*

---

## P

### Padding
Adding zeros around input (in CNNs) or to sequences (in NLP) to control output size or enable batching.

**Types**:
- Same padding: Output same size as input
- Valid padding: No padding, output smaller

**See also**: *Convolution*, *Stride*

---

### Parameter
A learnable value in the model (weights and biases). Updated during training.

**See also**: *Hyperparameter*, *Weight*

---

### Perplexity
A metric for language models. Lower is better. Exponential of cross-entropy loss.

**Formula**: PPL = exp(L) = exp(-1/N Σ log P(wᵢ | context))

**Interpretation**: "Effective vocabulary size" for next word prediction

**See also**: *Language Model*, *Cross-Entropy*

---

### Pooling
A downsampling operation that reduces spatial dimensions while retaining important information.

**Types**: Max pooling, average pooling, global pooling

**See also**: *Max Pooling*, *CNN*

---

### Position Encoding (Positional Embedding)
A way to inject sequence position information into Transformers, which don't inherently understand order.

**Types**: Sinusoidal (fixed), learned, RoPE, ALiBi

**See also**: *Transformer*, *Attention*

---

### PPO (Proximal Policy Optimization)
A reinforcement learning algorithm used in RLHF. Updates policy while staying close to the previous policy.

**See also**: *RLHF*, *DPO*

---

### Pre-training
Training a model on a large dataset before fine-tuning for specific tasks. Learns general representations.

**Example**: GPT pre-trained on web text, then fine-tuned for chat

**See also**: *Fine-tuning*, *Transfer Learning*

---

### Prompt
The input text given to an LLM. Can include instructions, context, examples.

**See also**: *Prompting*, *In-Context Learning*

---

### Prompting
The practice of crafting inputs to get desired outputs from LLMs without modifying model parameters.

**Techniques**: Zero-shot, few-shot, chain-of-thought, system prompts

**See also**: *In-Context Learning*, *Chain-of-Thought*

---

## Q

### Q (Query)
In attention, the query vectors determine what information to look for.

**Intuition**: Query is the "question" being asked

**See also**: *Attention*, *Key*, *Value*

---

### QLoRA
LoRA applied to a quantized model. Enables fine-tuning large models on limited hardware.

**See also**: *LoRA*, *Quantization*, *Fine-tuning*

---

### Quantization
Reducing the precision of model weights (e.g., from 32-bit to 8-bit or 4-bit) to reduce memory and speed up inference.

**Types**: Post-training quantization, quantization-aware training

**See also**: *Inference Optimization*, *Model Compression*

---

## R

### RAG (Retrieval-Augmented Generation)
A technique that augments LLM generation with retrieved relevant documents. Reduces hallucination and enables access to external knowledge.

**Pipeline**: Query → Retrieve → Augment prompt → Generate

**See also**: *Vector Database*, *Embedding*, *Chunking*

---

### ReAct
An agent framework that interleaves reasoning and acting. Generates thoughts and actions alternately.

**Pattern**: Thought → Action → Observation → Thought → ...

**See also**: *Agent*, *Tool Use*

---

### Receptive Field
The region of input that affects a particular neuron's output. In CNNs, grows with depth.

**See also**: *CNN*, *Convolution*

---

### Regression
A task where the model predicts continuous values.

**See also**: *MSE*, *Classification*

---

### Regularization
Techniques to prevent overfitting by constraining the model.

**Types**: L1, L2/weight decay, dropout, early stopping, data augmentation

**See also**: *Overfitting*, *Dropout*, *L2 Regularization*

---

### ReLU (Rectified Linear Unit)
The most common activation function. Simple and effective.

**Formula**: ReLU(x) = max(0, x)

**Issues**: Dead neurons (always output 0)

**See also**: *Activation Function*, *Leaky ReLU*, *GELU*

---

### Reparameterization Trick
A technique in VAEs to enable backpropagation through random sampling.

**Instead of**: z ~ N(μ, σ²)
**Use**: z = μ + σ * ε, where ε ~ N(0, 1)

**See also**: *VAE*, *ELBO*

---

### Representation Learning
Learning useful representations of data that make downstream tasks easier.

**See also**: *Embedding*, *Pre-training*, *Self-Supervised Learning*

---

### Residual Connection (Skip Connection)
A connection that adds the input of a layer directly to its output. Enables training of very deep networks.

**Formula**: y = F(x) + x

**See also**: *ResNet*, *Transformer*

---

### ResNet (Residual Network)
A CNN architecture that uses residual connections. Enabled training of networks with 100+ layers.

**See also**: *Residual Connection*, *CNN*

---

### Reward Model
A model that scores responses according to human preferences. Used in RLHF.

**Training data**: Pairs of (preferred response, rejected response)

**See also**: *RLHF*, *Alignment*

---

### RLHF (Reinforcement Learning from Human Feedback)
A technique for aligning LLMs with human preferences using reinforcement learning.

**Pipeline**:
1. Supervised fine-tuning (SFT)
2. Train reward model on preferences
3. Optimize policy with PPO

**See also**: *Alignment*, *Reward Model*, *PPO*, *DPO*

---

### RMSprop
An optimizer that adapts learning rate per parameter based on recent gradient magnitudes.

**Formula**:
```
v_t = β * v_{t-1} + (1-β) * g_t²
θ_t = θ_{t-1} - α * g_t / √(v_t + ε)
```

**See also**: *Adam*, *Gradient Descent*

---

### RNN (Recurrent Neural Network)
A neural network with connections that form cycles, allowing it to process sequences with memory.

**Formula**: h_t = f(W_xh * x_t + W_hh * h_{t-1})

**See also**: *LSTM*, *GRU*, *Hidden State*

---

### RoPE (Rotary Position Embedding)
A position encoding method that encodes position through rotation of the embedding vectors.

**See also**: *Position Encoding*, *Transformer*

---

## S

### Scaling Laws
Empirical relationships between model size, data size, compute, and performance.

**Chinchilla Law**: Optimal training uses ~20 tokens per parameter

**See also**: *LLM*, *Pre-training*

---

### Self-Attention
Attention where queries, keys, and values all come from the same sequence. Each position attends to all positions.

**See also**: *Attention*, *Transformer*, *Cross-Attention*

---

### Self-Supervised Learning
Learning from unlabeled data by creating supervised tasks from the data itself.

**Examples**: Masked language modeling, contrastive learning, next token prediction

**See also**: *Pre-training*, *Contrastive Learning*

---

### Seq2Seq (Sequence-to-Sequence)
An architecture that maps one sequence to another using an encoder-decoder structure.

**Uses**: Translation, summarization, dialogue

**See also**: *Encoder-Decoder*, *Attention*

---

### SFT (Supervised Fine-Tuning)
Fine-tuning on labeled examples of desired behavior. First step in RLHF pipeline.

**See also**: *Fine-tuning*, *RLHF*

---

### SGD (Stochastic Gradient Descent)
Gradient descent using one or a few examples per update instead of the full dataset.

**Formula**: θ = θ - α * ∇L(θ; x_i, y_i)

**See also**: *Gradient Descent*, *Mini-batch*, *Momentum*

---

### Sigmoid
An activation function that squashes values to (0, 1). Used for binary outputs and gates.

**Formula**: σ(x) = 1 / (1 + e^(-x))

**See also**: *Activation Function*, *Tanh*

---

### Skip Connection
See *Residual Connection*

---

### Softmax
A function that converts logits to a probability distribution.

**Formula**: softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)

**See also**: *Cross-Entropy Loss*, *Classification*

---

### Speculative Decoding
An inference optimization that uses a small model to draft tokens, verified in parallel by the large model.

**See also**: *Inference Optimization*, *KV Cache*

---

### Stable Diffusion
A latent diffusion model for text-to-image generation. Operates in latent space of a VAE for efficiency.

**See also**: *Diffusion Model*, *VAE*, *Text-to-Image*

---

### Stride
The step size when moving a filter across input in convolution. Stride > 1 reduces output size.

**See also**: *Convolution*, *Padding*

---

## T

### Tanh (Hyperbolic Tangent)
An activation function that squashes values to (-1, 1). Zero-centered unlike sigmoid.

**Formula**: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**See also**: *Activation Function*, *Sigmoid*

---

### Temperature
A parameter that controls randomness in sampling from probability distributions.

- T < 1: More deterministic (sharper distribution)
- T > 1: More random (flatter distribution)
- T = 0: Greedy (always pick most likely)

**See also**: *Sampling*, *Softmax*

---

### Token
The basic unit of text in language models. Can be words, subwords, or characters.

**See also**: *Tokenization*, *BPE*

---

### Tokenization
The process of splitting text into tokens for processing by a language model.

**Methods**: Character-level, word-level, subword (BPE, WordPiece)

**See also**: *Token*, *BPE*, *Vocabulary*

---

### Tool Use
The ability of LLM agents to call external tools (APIs, calculators, search) to accomplish tasks.

**See also**: *Agent*, *ReAct*

---

### Top-k Sampling
A decoding method that samples from only the k most likely tokens.

**See also**: *Top-p Sampling*, *Temperature*, *Sampling*

---

### Top-p Sampling (Nucleus Sampling)
A decoding method that samples from the smallest set of tokens whose cumulative probability exceeds p.

**See also**: *Top-k Sampling*, *Temperature*

---

### Training
The process of optimizing model parameters to minimize loss on training data.

**See also**: *Inference*, *Optimization*, *Backpropagation*

---

### Transfer Learning
Using knowledge learned on one task/dataset for a different but related task.

**See also**: *Pre-training*, *Fine-tuning*

---

### Transformer
The dominant architecture for sequence modeling, based entirely on attention mechanisms.

**Components**: Multi-head attention, feedforward layers, residual connections, layer norm

**Paper**: "Attention Is All You Need" (Vaswani et al., 2017)

**See also**: *Attention*, *Self-Attention*, *BERT*, *GPT*

---

## U

### Underfitting
When a model is too simple to capture patterns in the data. Poor performance on both training and test data.

**Solutions**: Larger model, more features, less regularization

**See also**: *Overfitting*, *Bias-Variance Tradeoff*

---

## V

### V (Value)
In attention, the value vectors contain the information to be retrieved/combined.

**Intuition**: Values are the "content" that gets weighted and summed

**See also**: *Attention*, *Query*, *Key*

---

### Vanishing Gradients
A problem where gradients become exponentially small as they propagate backward through many layers.

**Causes**: Saturating activations (sigmoid, tanh), deep networks
**Solutions**: ReLU, residual connections, LSTM/GRU, normalization

**See also**: *Gradient*, *LSTM*, *Residual Connection*

---

### VAE (Variational Autoencoder)
A generative model that learns a latent distribution. Combines autoencoder structure with probabilistic modeling.

**Components**: Encoder (q(z|x)), decoder (p(x|z)), prior (p(z))

**See also**: *Autoencoder*, *ELBO*, *Generative Model*

---

### Vector Database
A database optimized for storing and searching high-dimensional vectors using similarity search.

**Examples**: Pinecone, Weaviate, Milvus, FAISS

**See also**: *RAG*, *Embedding*, *Similarity Search*

---

### Vocabulary
The set of all tokens known to a language model.

**See also**: *Token*, *Tokenization*, *BPE*

---

## W

### Warmup
A training technique that starts with a low learning rate and gradually increases it.

**Purpose**: Allows model to find a good region before taking large steps

**See also**: *Learning Rate Schedule*

---

### Weight
A learnable parameter that scales inputs. The main learned values in neural networks.

**See also**: *Parameter*, *Bias*

---

### Weight Decay
Adding L2 penalty to weights during optimization. Implemented differently in Adam vs AdamW.

**See also**: *L2 Regularization*, *AdamW*

---

### Weight Initialization
The method for setting initial parameter values before training.

**Methods**: Xavier (sigmoid/tanh), He (ReLU), small random

**See also**: *Xavier Initialization*, *He Initialization*

---

### Word Embedding
A dense vector representation of words that captures semantic relationships.

**Examples**: Word2Vec, GloVe, FastText

**See also**: *Embedding*, *Representation Learning*

---

### WordPiece
A subword tokenization algorithm similar to BPE, used by BERT. Selects merges based on likelihood improvement.

**See also**: *BPE*, *Tokenization*

---

## X

### Xavier Initialization
Weight initialization scaled by √(2/(n_in + n_out)). Designed for sigmoid/tanh activations.

**Formula**: W ~ N(0, √(2/(n_in + n_out)))

**See also**: *He Initialization*, *Weight Initialization*

---

## Z

### Zero-Shot Learning
Making predictions on tasks/classes not seen during training, without any examples.

**In LLMs**: Following instructions without in-context examples

**See also**: *Few-Shot Learning*, *In-Context Learning*

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| x | Input |
| y | Target/label |
| ŷ | Prediction |
| W | Weight matrix |
| b | Bias vector |
| θ | Parameters (all weights and biases) |
| α, η | Learning rate |
| λ | Regularization strength |
| ∇ | Gradient operator |
| L | Loss function |
| σ | Sigmoid function or standard deviation |
| μ | Mean |
| ε | Small constant for numerical stability |
| β | Momentum coefficient or KL weight |
| ⊙ | Element-wise multiplication |
| @ | Matrix multiplication |
| T | Transpose |
| || · || | Norm |
| E[·] | Expectation |
| P(·) | Probability |
| log | Natural logarithm |
| exp | Exponential function |
| softmax | Softmax function |
| ReLU | Rectified Linear Unit |

---

## References

- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville
- [CS231n](http://cs231n.stanford.edu/) - Stanford CNN course
- [CS224n](http://cs224n.stanford.edu/) - Stanford NLP course
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [Lilian Weng's Blog](https://lilianweng.github.io/) - Excellent technical posts

---

*Last updated: January 2026*
