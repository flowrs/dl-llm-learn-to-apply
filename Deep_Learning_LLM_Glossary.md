# Deep Learning & LLM Glossary

A comprehensive reference of fundamental terms in deep learning and large language models, with example sentences to clarify each concept.

---

## Table of Contents

1. [Core Neural Network Concepts](#1-core-neural-network-concepts)
2. [Training & Optimization](#2-training--optimization)
3. [Network Architectures](#3-network-architectures)
4. [Activation Functions](#4-activation-functions)
5. [Regularization & Normalization](#5-regularization--normalization)
6. [Attention & Transformers](#6-attention--transformers)
7. [Large Language Models (LLMs)](#7-large-language-models-llms)
8. [Training Paradigms](#8-training-paradigms)
9. [Generative Models](#9-generative-models)
10. [Deployment & Optimization](#10-deployment--optimization)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Data & Preprocessing](#12-data--preprocessing)
13. [FAQ](#13-faq)

---

## 1. Core Neural Network Concepts

### Artificial Neural Network (ANN)
A computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers that process information.

> "An **artificial neural network** learns to recognize handwritten digits by adjusting millions of connections between its neurons."

### Neuron (Node, Unit)
The basic computational unit in a neural network that receives inputs, applies weights and a bias, then passes the result through an activation function.

> "Each **neuron** in the hidden layer computes a weighted sum of its inputs, adds a bias, and applies ReLU activation."

### Weight
A learnable parameter that determines the strength of the connection between two neurons; weights are multiplied with input values.

> "During training, the network adjusts its **weights** to minimize the difference between predictions and actual labels."

### Bias
A learnable parameter added to the weighted sum before activation, allowing the neuron to shift its activation threshold.

> "The **bias** term allows a neuron to activate even when all inputs are zero, giving the model more flexibility."

### Parameter
Any learnable value in a neural network, including weights and biases; the total count indicates model size.

> "GPT-3 has 175 billion **parameters**, making it one of the largest language models ever trained."

### Layer
A collection of neurons that process data at the same stage; networks have input, hidden, and output layers.

> "Adding more hidden **layers** allows the network to learn increasingly abstract representations of the data."

### Deep Learning
A subset of machine learning using neural networks with multiple hidden layers to learn hierarchical representations.

> "**Deep learning** enabled breakthroughs in image recognition by automatically learning features from raw pixels."

### Forward Pass (Forward Propagation)
The process of passing input data through the network from input to output layer to generate predictions.

> "During the **forward pass**, the image flows through convolutional layers, and the network outputs class probabilities."

### Inference
Using a trained model to make predictions on new, unseen data (as opposed to training).

> "After training, the model performs **inference** on user queries in under 100 milliseconds."

### Latent Space (Embedding Space)
A lower-dimensional representation where similar inputs are mapped to nearby points.

> "In the **latent space**, images of cats cluster together, separate from images of dogs."

### Tensor
A multi-dimensional array of numbers; the fundamental data structure in deep learning frameworks.

> "A color image is represented as a 3D **tensor** with dimensions [height, width, channels]."

### Hidden State
Internal memory in recurrent networks that carries information from previous time steps.

> "The RNN's **hidden state** accumulates context as it reads each word in the sentence."

---

## 2. Training & Optimization

### Training
The process of adjusting model parameters to minimize a loss function using training data.

> "**Training** the model on 1 million images took 3 days on 8 GPUs."

### Loss Function (Cost Function, Objective Function)
A function that measures how wrong the model's predictions are; training aims to minimize this value.

> "We used cross-entropy as our **loss function** because it works well for classification tasks."

### Gradient
The partial derivative of the loss with respect to each parameter, indicating how to adjust weights to reduce loss.

> "The **gradient** tells us that increasing this weight by 0.01 would reduce the loss by 0.005."

### Gradient Descent
An optimization algorithm that iteratively adjusts parameters in the direction that reduces the loss.

> "**Gradient descent** slowly walks downhill on the loss landscape, searching for the lowest point."

### Stochastic Gradient Descent (SGD)
A variant of gradient descent that updates parameters using a random subset (mini-batch) of training data.

> "**SGD** is faster than batch gradient descent because it doesn't need to process all training examples at once."

### Backpropagation (Backprop)
An algorithm that efficiently computes gradients by propagating errors backward through the network using the chain rule.

> "**Backpropagation** calculates how much each weight contributed to the error, starting from the output layer."

### Learning Rate
A hyperparameter controlling how large the parameter updates are during training.

> "Setting the **learning rate** too high caused the loss to oscillate wildly instead of decreasing."

### Optimizer
An algorithm that determines how to update weights based on gradients (e.g., SGD, Adam, AdamW).

> "We switched from SGD to the **Adam optimizer** and saw faster convergence."

### Epoch
One complete pass through the entire training dataset.

> "After 10 **epochs**, the model's accuracy plateaued at 94%."

### Batch (Mini-batch)
A subset of training examples processed together before updating weights.

> "We used a **batch** size of 32, meaning the model sees 32 images before each weight update."

### Batch Size
The number of training examples in one batch; affects training speed and memory usage.

> "Increasing the **batch size** from 16 to 64 sped up training but required more GPU memory."

### Iteration (Step)
One weight update; processing one batch constitutes one iteration.

> "With 1000 training examples and batch size 100, each epoch contains 10 **iterations**."

### Convergence
When the loss stops decreasing significantly and the model has learned as much as it can.

> "The model reached **convergence** after 50 epochs, so we stopped training early."

### Overfitting
When a model learns the training data too well, including noise, and fails to generalize to new data.

> "The model achieved 99% training accuracy but only 70% test accuracy, a clear sign of **overfitting**."

### Underfitting
When a model is too simple to capture the underlying patterns in the data.

> "Our linear model was **underfitting**; switching to a neural network improved accuracy by 20%."

### Generalization
A model's ability to perform well on new, unseen data.

> "Good **generalization** means the model learned true patterns, not just memorized training examples."

### Hyperparameter
Configuration values set before training that aren't learned (e.g., learning rate, number of layers).

> "We tuned **hyperparameters** like dropout rate and layer count to improve validation accuracy."

### Validation Set
A subset of data used to evaluate the model during training and tune hyperparameters.

> "We monitored loss on the **validation set** to detect overfitting early."

### Test Set
A held-out dataset used only for final evaluation after all training and tuning is complete.

> "The model achieved 92% accuracy on the **test set**, confirming it generalizes well."

### Vanishing Gradient Problem
When gradients become extremely small in deep networks, preventing early layers from learning.

> "LSTMs were invented to solve the **vanishing gradient problem** in standard RNNs."

### Exploding Gradient Problem
When gradients become extremely large, causing unstable training with huge weight updates.

> "Gradient clipping prevents the **exploding gradient problem** by capping gradient magnitudes."

### Gradient Clipping
A technique that limits gradient magnitude to prevent exploding gradients.

> "We applied **gradient clipping** with a max norm of 1.0 to stabilize training."

---

## 3. Network Architectures

### Feedforward Neural Network (FNN)
A network where information flows in one direction from input to output, with no cycles.

> "A simple **feedforward neural network** with two hidden layers can learn XOR."

### Convolutional Neural Network (CNN)
A network designed for grid-like data (images) using convolutional layers that detect local patterns.

> "The **CNN** learned to detect edges in early layers and faces in deeper layers."

### Convolution
An operation that slides a small filter across an image, computing dot products to detect features.

> "**Convolution** with a 3x3 edge-detection filter highlights vertical lines in the image."

### Filter (Kernel)
A small matrix of learnable weights used in convolution to detect specific features.

> "The network learned 64 different **filters** in the first layer, each detecting a different pattern."

### Feature Map
The output of applying a filter to an input; represents where a feature was detected.

> "The **feature map** showed high activations wherever the image contained horizontal edges."

### Pooling
An operation that reduces spatial dimensions by taking the max or average of neighboring values.

> "**Max pooling** with 2x2 windows halves the image resolution while keeping the strongest features."

### Stride
The step size when sliding a filter across the input during convolution.

> "Using **stride** 2 instead of 1 reduces the output size by half."

### Padding
Adding zeros around the input border to control output dimensions after convolution.

> "**Same padding** preserves the input dimensions, while valid padding shrinks them."

### Recurrent Neural Network (RNN)
A network with loops that process sequential data by maintaining hidden state across time steps.

> "The **RNN** reads the sentence word by word, updating its hidden state to track context."

### Long Short-Term Memory (LSTM)
An RNN variant with gates that control information flow, solving the vanishing gradient problem.

> "**LSTM** networks can remember information across hundreds of time steps."

### Gated Recurrent Unit (GRU)
A simplified LSTM variant with fewer gates but similar performance.

> "We used **GRU** instead of LSTM because it trains faster with similar accuracy."

### Sequence-to-Sequence (Seq2Seq)
An architecture with encoder and decoder RNNs for transforming one sequence into another.

> "Machine translation uses **seq2seq** models to convert English sentences to French."

### Encoder
A network component that compresses input into a dense representation.

> "The **encoder** transforms the input sentence into a 512-dimensional context vector."

### Decoder
A network component that generates output from an encoded representation.

> "The **decoder** produces the translation one word at a time, conditioned on the encoded input."

### Autoencoder
A network trained to compress input to a latent space and reconstruct it, learning efficient representations.

> "The **autoencoder** learned to compress images to 1% of their original size with minimal quality loss."

### Residual Connection (Skip Connection)
A shortcut that adds a layer's input directly to its output, enabling deeper networks.

> "**Residual connections** allow gradients to flow directly backward, making 100+ layer networks trainable."

### ResNet (Residual Network)
A CNN architecture using residual connections that enabled training of very deep networks.

> "**ResNet-152** achieved state-of-the-art image classification by stacking 152 layers with skip connections."

### U-Net
A CNN architecture with symmetric encoder-decoder and skip connections, popular for segmentation.

> "**U-Net** segments medical images by combining high-resolution details with semantic context."

---

## 4. Activation Functions

### Activation Function
A non-linear function applied to a neuron's output, enabling the network to learn complex patterns.

> "Without **activation functions**, a neural network would just be a linear transformation."

### ReLU (Rectified Linear Unit)
An activation function that outputs the input if positive, zero otherwise: f(x) = max(0, x).

> "**ReLU** is the most popular activation function because it's simple and helps prevent vanishing gradients."

### Sigmoid
An activation function that squashes values to the range (0, 1): f(x) = 1/(1 + e^(-x)).

> "**Sigmoid** is used in the output layer for binary classification to produce probabilities."

### Tanh (Hyperbolic Tangent)
An activation function that squashes values to the range (-1, 1).

> "**Tanh** is often preferred over sigmoid in hidden layers because it's zero-centered."

### Softmax
An activation function that converts a vector of values into a probability distribution summing to 1.

> "The **softmax** output shows the model is 90% confident this is a cat and 10% confident it's a dog."

### Leaky ReLU
A ReLU variant that allows small negative values instead of zero: f(x) = max(0.01x, x).

> "**Leaky ReLU** prevents 'dead neurons' that occur when standard ReLU neurons always output zero."

### GELU (Gaussian Error Linear Unit)
A smooth activation function used in modern transformers: x * P(X <= x) where X is standard normal.

> "**GELU** is the default activation in BERT and GPT models, slightly outperforming ReLU."

### Swish (SiLU)
A smooth, non-monotonic activation function: f(x) = x * sigmoid(x).

> "**Swish** activation improved accuracy in some image models compared to ReLU."

---

## 5. Regularization & Normalization

### Regularization
Techniques that prevent overfitting by constraining model complexity.

> "**Regularization** helped our model generalize better to unseen data."

### L1 Regularization (Lasso)
Adding the sum of absolute weight values to the loss, encouraging sparse weights.

> "**L1 regularization** pushed many weights to exactly zero, effectively removing unimportant features."

### L2 Regularization (Ridge, Weight Decay)
Adding the sum of squared weight values to the loss, encouraging smaller weights.

> "**L2 regularization** prevents any single weight from becoming too large."

### Dropout
Randomly setting a fraction of neuron outputs to zero during training to prevent co-adaptation.

> "With 50% **dropout**, each neuron learns to work independently rather than relying on specific partners."

### Batch Normalization (BatchNorm)
Normalizing layer inputs across a mini-batch to stabilize and accelerate training.

> "Adding **batch normalization** after each convolutional layer allowed us to use higher learning rates."

### Layer Normalization (LayerNorm)
Normalizing across features for each example, commonly used in transformers.

> "Transformers use **layer normalization** instead of batch normalization because it works with variable-length sequences."

### Early Stopping
Halting training when validation performance stops improving to prevent overfitting.

> "**Early stopping** ended training at epoch 23, before the model started memorizing the training data."

### Data Augmentation
Artificially expanding training data by applying transformations (rotation, cropping, etc.).

> "**Data augmentation** doubled our effective dataset size by randomly flipping and rotating images."

---

## 6. Attention & Transformers

### Attention Mechanism
A technique allowing models to focus on relevant parts of the input when producing each output.

> "The **attention mechanism** lets the model focus on 'cat' when translating to 'chat' in French."

### Self-Attention
Attention applied within a single sequence, allowing each position to attend to all other positions.

> "**Self-attention** enables the word 'it' to connect with 'the cat' earlier in the sentence."

### Query, Key, Value (Q, K, V)
Three projections of input used in attention: Query asks, Key answers, Value provides content.

> "In attention, the **Query** represents 'what am I looking for?', the **Key** represents 'what do I contain?', and the **Value** is the actual content retrieved."

### Attention Score
The computed similarity between a query and key, determining how much to attend to each position.

> "High **attention scores** between 'bank' and 'river' helped the model understand it means riverbank, not financial bank."

### Multi-Head Attention
Running multiple attention operations in parallel, each learning different relationship types.

> "**Multi-head attention** with 8 heads allows the model to simultaneously track syntax, semantics, and coreference."

### Transformer
An architecture based entirely on attention mechanisms, without recurrence or convolution.

> "The **Transformer** processes all words in parallel, making it much faster to train than RNNs."

### Positional Encoding (Positional Embedding)
Information added to embeddings to indicate token position, since transformers have no inherent order.

> "Without **positional encoding**, the transformer couldn't distinguish 'dog bites man' from 'man bites dog'."

### Causal Masking (Autoregressive Masking)
Preventing attention to future tokens in decoder models to maintain left-to-right generation.

> "**Causal masking** ensures the model can only see previous words when predicting the next word."

### Cross-Attention
Attention where queries come from one sequence and keys/values from another (e.g., decoder attending to encoder).

> "In translation, **cross-attention** allows each French word to look at the relevant English words."

### Context Window (Context Length)
The maximum number of tokens a model can process at once.

> "GPT-4's 128K **context window** lets it read an entire book in one prompt."

### KV Cache (Key-Value Cache)
Storing computed keys and values during generation to avoid redundant computation.

> "**KV cache** speeds up generation because we don't recompute attention for already-generated tokens."

---

## 7. Large Language Models (LLMs)

### Large Language Model (LLM)
A neural network with billions of parameters trained on massive text corpora to understand and generate language.

> "**LLMs** like GPT-4 can write essays, code, and answer questions across diverse domains."

### Token
The basic unit of text processed by language models, typically a word or subword piece.

> "The word 'unbelievable' might be split into three **tokens**: 'un', 'believ', 'able'."

### Tokenization
The process of converting text into tokens that the model can process.

> "**Tokenization** turns 'Hello, world!' into token IDs like [15496, 11, 995, 0]."

### Tokenizer
The component that performs tokenization and converts between text and token IDs.

> "The **tokenizer** uses a vocabulary of 50,000 subword tokens to represent any text."

### Byte-Pair Encoding (BPE)
A tokenization algorithm that iteratively merges frequent character pairs into single tokens.

> "**BPE** learns that 'ing' appears frequently, so it becomes a single token rather than three letters."

### Vocabulary
The set of all tokens a model can recognize and generate.

> "The model's **vocabulary** of 32,000 tokens includes common words, subwords, and special symbols."

### Embedding
A dense vector representation of a token, word, or concept in continuous space.

> "The **embedding** for 'king' minus 'man' plus 'woman' approximately equals the embedding for 'queen'."

### Embedding Dimension
The size of embedding vectors, typically 768 to 8192 in modern models.

> "BERT-base uses an **embedding dimension** of 768, while GPT-3 uses 12,288."

### Pre-training
Training a model on a large general corpus before fine-tuning for specific tasks.

> "**Pre-training** on billions of words gives the model general language understanding."

### Next-Token Prediction (Autoregressive Language Modeling)
Training objective where the model predicts the next token given previous context.

> "GPT learns through **next-token prediction**: given 'The cat sat on the', predict 'mat'."

### Masked Language Modeling (MLM)
Training objective where the model predicts masked tokens in the input.

> "BERT uses **masked language modeling**: given 'The [MASK] sat on the mat', predict 'cat'."

### Perplexity
A metric measuring how surprised a language model is by test text; lower is better.

> "The model's **perplexity** of 15 means it's as uncertain as randomly choosing from 15 words."

### Fine-Tuning
Adapting a pre-trained model to a specific task by training on task-specific data.

> "We **fine-tuned** GPT on customer service transcripts to create a support chatbot."

### Prompt
The input text given to a language model to guide its generation.

> "The **prompt** 'Write a poem about coding' tells the model what to generate."

### Prompt Engineering
Crafting effective prompts to elicit desired model behavior without changing weights.

> "Good **prompt engineering** can dramatically improve model outputs without any retraining."

### Few-Shot Learning
Providing a few examples in the prompt to teach the model a task.

> "With **few-shot learning**, showing 3 translation examples helps the model translate new sentences."

### Zero-Shot Learning
Asking the model to perform a task without any examples.

> "**Zero-shot**, GPT-4 can translate to languages it wasn't explicitly trained to translate."

### In-Context Learning (ICL)
The ability of LLMs to learn tasks from examples provided in the prompt without weight updates.

> "**In-context learning** is remarkable: the model 'learns' from examples at inference time."

### Chain-of-Thought (CoT)
Prompting the model to show reasoning steps before giving the final answer.

> "**Chain-of-thought** prompting improved math accuracy from 18% to 78% by making the model 'think aloud'."

### Hallucination
When a model generates plausible-sounding but factually incorrect information.

> "The model **hallucinated** a citation to a paper that doesn't exist."

### Grounding
Connecting model outputs to verified external knowledge sources.

> "**Grounding** the model with a search API reduces hallucinations about current events."

### Temperature
A parameter controlling randomness in generation; higher = more random, lower = more deterministic.

> "We set **temperature** to 0.2 for factual tasks and 0.8 for creative writing."

### Top-k Sampling
Sampling only from the k most probable next tokens.

> "**Top-k** sampling with k=50 prevents the model from choosing very unlikely words."

### Top-p Sampling (Nucleus Sampling)
Sampling from the smallest set of tokens whose cumulative probability exceeds p.

> "**Top-p** sampling with p=0.9 adapts the candidate pool based on model confidence."

### Beam Search
A decoding strategy that keeps track of multiple candidate sequences.

> "**Beam search** with beam size 5 finds better translations than greedy decoding."

### Greedy Decoding
Always selecting the highest-probability next token.

> "**Greedy decoding** is fast but can miss better sequences that start with less likely tokens."

---

## 8. Training Paradigms

### Supervised Learning
Training with labeled data where the correct output is known.

> "In **supervised learning**, the model learns from 10,000 images each labeled with their category."

### Unsupervised Learning
Training without explicit labels, learning patterns or structure in data.

> "**Unsupervised learning** discovered customer segments without being told how many groups to find."

### Self-Supervised Learning (SSL)
Creating labels automatically from the data itself (e.g., predicting masked words).

> "**Self-supervised learning** lets us use billions of unlabeled web pages for pre-training."

### Semi-Supervised Learning
Combining a small amount of labeled data with a large amount of unlabeled data.

> "**Semi-supervised learning** leveraged 100 labeled examples plus 1 million unlabeled images."

### Reinforcement Learning (RL)
Training an agent to make decisions by rewarding good actions and penalizing bad ones.

> "Through **reinforcement learning**, the AI learned to play chess by playing millions of games against itself."

### Reward Model
A model trained to predict human preferences, used to guide RL training.

> "The **reward model** learned that humans prefer helpful, honest, and harmless responses."

### RLHF (Reinforcement Learning from Human Feedback)
Fine-tuning language models using human preference feedback via reinforcement learning.

> "**RLHF** aligned ChatGPT's behavior with human values by training on human preference data."

### PPO (Proximal Policy Optimization)
A stable reinforcement learning algorithm commonly used in RLHF.

> "We used **PPO** to fine-tune the model because it prevents catastrophically large policy updates."

### DPO (Direct Preference Optimization)
A simpler alternative to RLHF that directly optimizes on preference data without a reward model.

> "**DPO** achieves similar results to RLHF but is easier to implement and more stable."

### Transfer Learning
Using knowledge from one task to improve performance on a different task.

> "**Transfer learning** let us build a medical image classifier using features learned from ImageNet."

### Multi-Task Learning
Training a single model on multiple tasks simultaneously.

> "**Multi-task learning** on translation, summarization, and QA created a more capable general model."

### Curriculum Learning
Training on easy examples first, gradually increasing difficulty.

> "**Curriculum learning** started with short sentences before introducing long, complex ones."

### Contrastive Learning
Learning representations by pulling similar examples together and pushing dissimilar ones apart.

> "**Contrastive learning** taught the model that 'dog' and 'puppy' should have similar embeddings."

### Knowledge Distillation
Training a smaller model to mimic a larger model's outputs.

> "**Knowledge distillation** compressed our 10B parameter model into a 1B model with 95% of the accuracy."

---

## 9. Generative Models

### Generative Model
A model that learns the data distribution and can generate new samples.

> "**Generative models** can create new faces that look like real people but don't exist."

### Discriminative Model
A model that learns to distinguish between classes or predict labels.

> "A **discriminative model** predicts whether an email is spam, while a generative model could write spam emails."

### Autoregressive Model
A model that generates output one element at a time, conditioning on previous outputs.

> "GPT is **autoregressive**: it generates text token by token, each conditioned on previous tokens."

### Variational Autoencoder (VAE)
A generative model that learns a structured latent space using variational inference.

> "The **VAE** learned to generate new celebrity faces by sampling from its latent space."

### Generative Adversarial Network (GAN)
Two networks (generator and discriminator) trained adversarially to generate realistic samples.

> "The **GAN** generated fake photographs so realistic that humans couldn't distinguish them from real ones."

### Diffusion Model
A generative model that learns to reverse a gradual noising process to generate samples.

> "**Diffusion models** like DALL-E 3 create images by starting with noise and gradually refining it."

### Latent Diffusion
Performing diffusion in a compressed latent space for efficiency.

> "Stable Diffusion uses **latent diffusion** to generate high-resolution images on consumer GPUs."

### Classifier-Free Guidance
A technique to improve conditional generation by contrasting conditioned and unconditioned outputs.

> "**Classifier-free guidance** makes DALL-E images match the prompt more closely."

### Text-to-Image Model
A model that generates images from text descriptions.

> "The **text-to-image model** turned 'a cat wearing sunglasses on a beach' into a realistic photo."

### Multimodal Model
A model that can process and/or generate multiple types of data (text, images, audio, etc.).

> "**Multimodal models** like GPT-4V can understand both images and text in the same conversation."

---

## 10. Deployment & Optimization

### Model Compression
Techniques to reduce model size while maintaining performance.

> "**Model compression** reduced our model from 2GB to 200MB for mobile deployment."

### Quantization
Reducing numerical precision of weights (e.g., 32-bit to 8-bit) to save memory and speed up inference.

> "**Quantization** to 4-bit reduced model size by 8x with only 2% accuracy loss."

### Pruning
Removing unimportant weights or neurons to create a smaller, faster model.

> "**Pruning** removed 90% of the weights while maintaining 95% of the original accuracy."

### LoRA (Low-Rank Adaptation)
An efficient fine-tuning method that trains small adapter matrices instead of full weights.

> "**LoRA** let us fine-tune a 7B model on a single GPU by only training 0.1% of the parameters."

### QLoRA (Quantized LoRA)
Combining quantization with LoRA for even more memory-efficient fine-tuning.

> "**QLoRA** enabled fine-tuning a 65B model on a single consumer GPU."

### Adapter
A small trainable module inserted into a frozen pre-trained model.

> "We trained a separate **adapter** for each language, keeping the base model frozen."

### PEFT (Parameter-Efficient Fine-Tuning)
A family of techniques that fine-tune only a small subset of model parameters.

> "**PEFT** methods like LoRA and adapters make it feasible to customize large models on limited hardware."

### Batching
Processing multiple inputs together to improve GPU utilization.

> "**Batching** 32 requests together increased throughput by 10x compared to processing one at a time."

### KV Cache
Storing key-value pairs from attention to avoid recomputation during generation.

> "The **KV cache** grows with sequence length, limiting maximum generation length on limited memory."

### Speculative Decoding
Using a smaller model to draft tokens, verified by the larger model.

> "**Speculative decoding** achieved 2x speedup by letting a fast model propose tokens in parallel."

### Model Serving
Deploying models to handle production inference requests.

> "We used Triton Inference Server for **model serving** to handle 1000 requests per second."

### Latency
The time from sending a request to receiving a response.

> "Users expect **latency** under 200ms for interactive chat applications."

### Throughput
The number of requests a system can process per unit time.

> "Our optimized system achieved **throughput** of 500 queries per second."

### FLOPS (Floating Point Operations Per Second)
A measure of computational speed.

> "Training GPT-4 required approximately 10^25 **FLOPS** of computation."

---

## 11. Evaluation Metrics

### Accuracy
The proportion of correct predictions.

> "The model achieved 94% **accuracy** on the test set, correctly classifying 9,400 of 10,000 images."

### Precision
Of all positive predictions, the proportion that were correct.

> "With 90% **precision**, 9 out of 10 spam predictions were actually spam."

### Recall
Of all actual positives, the proportion that were correctly identified.

> "With 80% **recall**, the model found 8 out of 10 actual spam emails."

### F1 Score
The harmonic mean of precision and recall.

> "The **F1 score** of 0.85 balances the trade-off between precision and recall."

### BLEU (Bilingual Evaluation Understudy)
A metric for evaluating machine translation quality based on n-gram overlap.

> "The translation system achieved a **BLEU** score of 32, indicating good quality."

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Metrics for evaluating summarization by comparing with reference summaries.

> "Our summarization model achieved **ROUGE-L** of 42, measuring longest common subsequence overlap."

### Perplexity
How surprised the model is by test data; lower values indicate better fit.

> "The language model's **perplexity** dropped from 50 to 20 after fine-tuning on domain data."

### Cross-Entropy Loss
A loss function measuring the difference between predicted and true probability distributions.

> "**Cross-entropy loss** penalizes confident wrong predictions more than uncertain ones."

### Mean Squared Error (MSE)
A loss function measuring average squared difference between predictions and targets.

> "We minimized **MSE** to train the regression model predicting house prices."

### AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
A metric measuring classification performance across all thresholds.

> "An **AUC-ROC** of 0.95 indicates excellent ability to distinguish positive from negative cases."

### Benchmark
A standardized test dataset or task for comparing model performance.

> "MMLU is a popular **benchmark** for evaluating LLM knowledge across 57 subjects."

---

## 12. Data & Preprocessing

### Dataset
A collection of examples used for training, validation, or testing.

> "ImageNet is a **dataset** of 14 million images across 22,000 categories."

### Training Data
The subset of data used to train the model's parameters.

> "We split our data 80/10/10 into **training**, validation, and test sets."

### Label (Target, Ground Truth)
The correct output for a given input in supervised learning.

> "Each training image has a **label** indicating whether it contains a cat or dog."

### Annotation
Adding labels or metadata to data, often by human annotators.

> "**Annotation** of the medical dataset required trained radiologists."

### Feature
An individual measurable property of the data used as model input.

> "The model used 50 **features** including age, income, and purchase history."

### Feature Engineering
Manually creating features from raw data to improve model performance.

> "**Feature engineering** created a 'days since last purchase' feature that improved predictions."

### Feature Extraction
Automatically learning useful features from raw data (as neural networks do).

> "CNNs perform **feature extraction** automatically, eliminating manual feature engineering."

### One-Hot Encoding
Representing categorical values as binary vectors.

> "**One-hot encoding** transformed 'red', 'green', 'blue' into [1,0,0], [0,1,0], [0,0,1]."

### Normalization
Scaling data to a standard range (e.g., 0 to 1).

> "**Normalization** of pixel values from 0-255 to 0-1 improved training stability."

### Standardization
Transforming data to have zero mean and unit variance.

> "**Standardization** ensured all features contributed equally regardless of their original scale."

### Data Pipeline
The system for loading, preprocessing, and feeding data to the model.

> "Our **data pipeline** augmented images on-the-fly during training."

### Corpus
A large collection of text used for training language models.

> "The model was pre-trained on a **corpus** of 500 billion words from the web."

### Synthetic Data
Artificially generated data used to augment or replace real data.

> "We used **synthetic data** from GPT-4 to train the smaller model."

---

## 13. FAQ

### General Concepts

**Q: What's the difference between AI, Machine Learning, and Deep Learning?**

A: They're nested concepts: AI is the broadest (machines doing intelligent things), Machine Learning is a subset (learning from data), and Deep Learning is a subset of ML (using deep neural networks). All modern LLMs are deep learning, which is ML, which is AI.

**Q: Why do we need non-linear activation functions?**

A: Without non-linearity, stacking multiple layers would be equivalent to a single linear transformation (matrix multiplication). Non-linear activations allow neural networks to learn complex, non-linear relationships in data.

**Q: What's the difference between parameters and hyperparameters?**

A: Parameters (weights, biases) are learned during training through backpropagation. Hyperparameters (learning rate, number of layers, batch size) are set before training and control how training happens.

### Training & Optimization

**Q: Why do larger batch sizes require higher learning rates?**

A: Larger batches give more accurate gradient estimates, so the model can take larger steps without overshooting. A rule of thumb: if you double the batch size, increase the learning rate by sqrt(2).

**Q: How do I know if my model is overfitting?**

A: Watch for a gap between training and validation performance. If training loss keeps decreasing but validation loss starts increasing, you're overfitting. Other signs: near-perfect training accuracy with poor test accuracy.

**Q: When should I use Adam vs. SGD?**

A: Adam is generally easier to use (adapts learning rate automatically) and converges faster. SGD with momentum can achieve better final performance but requires more careful learning rate tuning. Use Adam for prototyping, SGD for squeezing out final performance.

### Architectures

**Q: When should I use a CNN vs. RNN vs. Transformer?**

A: CNNs are best for grid data (images). RNNs work for sequential data but have largely been replaced by Transformers. Transformers are now the default for most sequence tasks (text, audio, even images as patches) due to parallelization and performance.

**Q: Why are Transformers better than RNNs for language?**

A: RNNs process tokens sequentially (slow, can't parallelize) and struggle with long-range dependencies. Transformers process all tokens in parallel (fast) and can directly attend to any position regardless of distance.

**Q: What is the purpose of residual connections?**

A: Residual connections add the input of a layer to its output. This helps gradients flow through deep networks (solving vanishing gradients) and allows the network to learn identity mappings when needed.

### LLMs & NLP

**Q: Why do LLMs use subword tokenization instead of words or characters?**

A: Word tokenization creates huge vocabularies and can't handle new words. Character tokenization creates very long sequences and loses word-level meaning. Subword tokenization (like BPE) balances these: reasonable vocabulary size, handles any word, captures meaningful units.

**Q: What is the context window limitation and why does it exist?**

A: The context window limits how many tokens the model can process at once. It exists because self-attention has O(n^2) complexity in sequence length, and KV cache memory grows linearly with length. Longer contexts require more memory and computation.

**Q: How do LLMs "know" things without a database?**

A: Knowledge is encoded implicitly in the network weights during pre-training. When the model saw "Paris is the capital of France" many times in training data, this pattern got encoded into the weights. The model doesn't "look up" facts; it reconstructs them from learned patterns.

**Q: What causes hallucinations and how can they be reduced?**

A: Hallucinations occur because LLMs are trained to produce fluent text, not truthful text. They can be reduced through: RLHF training for truthfulness, retrieval augmentation (RAG), grounding with external tools, and prompting techniques that encourage honesty.

### Training Paradigms

**Q: What's the difference between pre-training, fine-tuning, and prompting?**

A: Pre-training learns general knowledge from massive unlabeled data. Fine-tuning adapts the pre-trained model to specific tasks by training on labeled task data. Prompting uses the model as-is, controlling behavior through input text without changing weights.

**Q: Why is RLHF important for LLMs?**

A: Pre-trained LLMs learn to predict text, not to be helpful or safe. RLHF trains the model to optimize for human preferences (helpfulness, honesty, harmlessness), aligning model behavior with human values.

**Q: What's the difference between LoRA and full fine-tuning?**

A: Full fine-tuning updates all model parameters, requiring significant memory and potentially causing catastrophic forgetting. LoRA only trains small low-rank adapter matrices (typically <1% of parameters), using much less memory and preserving base model capabilities.

### Practical Considerations

**Q: How much data do I need to train a model?**

A: It depends on the task complexity and model size. Rule of thumb: you need at least 10x more examples than parameters for small models. For fine-tuning LLMs, even 100-1000 high-quality examples can work due to transfer learning. More data generally helps until you saturate.

**Q: How do I choose the right model size?**

A: Consider your constraints (latency, memory, cost) and task difficulty. Larger models are more capable but slower and more expensive. Start with a small model, measure performance, and scale up only if needed. For many tasks, a 7B model matches or beats GPT-4 when fine-tuned.

**Q: Why is quantization important for deployment?**

A: Model size directly impacts memory usage, loading time, and inference speed. A 4-bit quantized model is 8x smaller than 32-bit and often 2-4x faster, with minimal accuracy loss. This makes it feasible to run large models on consumer hardware.

**Q: What's the trade-off between temperature 0 and temperature 1?**

A: Temperature 0 always picks the most likely token (deterministic, good for factual tasks). Temperature 1 samples according to the model's probability distribution (creative, diverse, but may generate errors). Most applications use 0.3-0.7 as a balance.

---

## Quick Reference Card

| Term | One-Line Definition |
|------|---------------------|
| Weight | A learnable number controlling connection strength |
| Bias | A learnable offset added before activation |
| Activation | A non-linear function enabling complex learning |
| Gradient | Direction to adjust weights to reduce loss |
| Backprop | Algorithm to compute gradients efficiently |
| Epoch | One pass through all training data |
| Batch | Subset of data processed together |
| CNN | Network using convolutions for grid data |
| RNN | Network with memory for sequential data |
| LSTM | RNN variant solving vanishing gradients |
| Transformer | Attention-based architecture, no recurrence |
| Attention | Mechanism to focus on relevant inputs |
| Token | Basic unit of text for LLMs |
| Embedding | Dense vector representation |
| Pre-training | Learning general knowledge from large data |
| Fine-tuning | Adapting to specific tasks |
| RLHF | Aligning models with human preferences |
| LoRA | Efficient fine-tuning via small adapters |
| Quantization | Reducing numerical precision for efficiency |
| Inference | Using trained model for predictions |

---

*This glossary is part of the Deep Learning to LLM Applications course.*
