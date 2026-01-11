# CSE 493G1: Deep Learning - Comprehensive Study Plan

## Course Overview

**University:** University of Washington
**Course:** CSE 493G1 - Deep Learning
**Based on:** Stanford CS231n: Deep Learning for Computer Vision

### Course Description
This course provides an intensive exploration of deep learning fundamentals with emphasis on neural networks for visual and language tasks. Students learn to implement and train neural networks from scratch, gaining detailed understanding of cutting-edge research in deep learning.

### Prerequisites
- **Mathematics:** Linear Algebra, Calculus, Statistics/Probability
- **Programming:** Python proficiency, NumPy basics
- Prior ML/CV experience helpful but not required

### Course Components
- Lectures (twice weekly)
- Recitations (weekly hands-on sessions)
- 3 Programming Assignments (30%)
- Midterm Exam (15%)
- Final Project (35%)
- Quizzes/Participation (20%)

---

## Week-by-Week Study Schedule

### Week 1: Foundations

#### Lecture 1: Introduction to Deep Learning
**Topics:**
- What is deep learning and why now?
- Applications: Computer vision, NLP, robotics, healthcare
- Course overview and logistics

**Key Concepts:**
- Data-driven approach vs. hand-engineered features
- The deep learning revolution (AlexNet 2012)
- GPU computing and big data enabling deep learning

**Readings:**
- [CS231n: Image Classification](https://cs231n.github.io/classification/)

---

#### Lecture 2: Image Classification
**Topics:**
- The image classification problem
- Data-driven approach: train/predict paradigm
- Nearest Neighbor classifier
- k-Nearest Neighbor (k-NN)
- Validation sets and hyperparameter tuning

**Key Concepts:**
- **Data-Driven Approach:** Instead of hardcoding rules, provide examples and let the algorithm learn
- **Distance Metrics:** L1 (Manhattan) and L2 (Euclidean) distance
- **k-NN Algorithm:** Vote among k nearest training examples
- **Train/Val/Test Split:** Never use test set for hyperparameter tuning
- **Cross-validation:** For small datasets, use k-fold validation

**Practical Notes:**
- k-NN achieves ~38% on CIFAR-10 (humans: 94%, CNNs: 95%+)
- Pixel distance is semantically meaningless
- k-NN is slow at test time, motivating parametric approaches

**Readings:**
- [CS231n: Image Classification Notes](https://cs231n.github.io/classification/)

---

#### Recitation 1: Python and NumPy Basics
**Topics:**
- Python fundamentals for deep learning
- NumPy array operations
- Vectorized computations
- Jupyter/Colab workflow

**Exercises:**
- Array broadcasting
- Vectorized distance computation
- Basic image manipulation

---

### Week 2: Linear Classification and Optimization

#### Lecture 3: Regularization and Optimization
**Topics:**
- Linear classifiers: score function f(x,W,b) = Wx + b
- Loss functions: SVM and Softmax
- Regularization: L1, L2, Dropout
- Optimization: Gradient Descent

**Key Concepts:**

**Linear Score Function:**
```
f(x, W, b) = Wx + b
```
- W is template/weight matrix
- Each row of W acts as a class template
- Score = dot product between image and template

**SVM Loss (Hinge Loss):**
```
L_i = Σ_{j≠y_i} max(0, s_j - s_{y_i} + Δ)
```
- Enforces margin between correct and incorrect classes
- Satisfied when correct class beats others by margin Δ

**Softmax Loss (Cross-Entropy):**
```
L_i = -log(e^{s_{y_i}} / Σ_j e^{s_j})
```
- Interprets scores as unnormalized log-probabilities
- Continuously pushes for higher correct-class probability

**Regularization:**
```
L = (1/N)Σ_i L_i + λR(W)
```
- L2: λΣw² (encourages smaller, distributed weights)
- L1: λΣ|w| (encourages sparse weights)

**Optimization Strategies:**
1. Random Search (bad: 15.5% accuracy)
2. Random Local Search (better: 21.4%)
3. **Gradient Descent** (best: follows steepest descent)

**Gradient Computation:**
- Numerical gradient: slow but simple (for debugging)
- Analytic gradient: fast and exact (for training)

**Mini-batch SGD:**
```python
for iteration in range(num_iterations):
    batch = sample(training_data, batch_size)  # typically 32-256
    gradient = compute_gradient(batch)
    weights -= learning_rate * gradient
```

**Readings:**
- [CS231n: Linear Classification](https://cs231n.github.io/linear-classify/)
- [CS231n: Optimization](https://cs231n.github.io/optimization-1/)

---

#### Lecture 4: Neural Networks and Backpropagation
**Topics:**
- Biological inspiration vs. mathematical model
- Activation functions
- Multi-layer networks
- Backpropagation algorithm
- Computational graphs

**Key Concepts:**

**Single Neuron:**
```
output = activation(Σ(w_i * x_i) + b)
```

**Activation Functions:**
| Function | Formula | Pros | Cons |
|----------|---------|------|------|
| Sigmoid | σ(x) = 1/(1+e^(-x)) | Smooth, bounded | Vanishing gradients, not zero-centered |
| Tanh | tanh(x) | Zero-centered | Vanishing gradients |
| ReLU | max(0,x) | Fast, no saturation | Dead neurons |
| Leaky ReLU | max(0.01x, x) | No dead neurons | Extra hyperparameter |

**Recommendation:** Use ReLU, be careful with learning rates, monitor dead neurons

**Network Architecture:**
```
INPUT -> [FC -> ReLU]* -> FC -> SOFTMAX
```
- Fully-connected layers: every neuron connects to all previous neurons
- Depth (layers) vs. width (neurons per layer)

**Backpropagation:**
- Chain rule for computing gradients
- Forward pass: compute outputs
- Backward pass: compute gradients

**Computational Graph Pattern:**
```
Local gradient × Upstream gradient = Downstream gradient
```

**Readings:**
- [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/)

---

#### Recitation 2: Backpropagation Deep Dive
**Topics:**
- Chain rule derivations
- Gradient flow patterns
- Common gradient computations

---

### Week 3: Convolutional Neural Networks

#### Lecture 5: Convolutional Neural Networks (CNNs)
**Topics:**
- Motivation: spatial structure and parameter sharing
- Convolution operation
- Pooling layers
- CNN architectures

**Key Concepts:**

**Why CNNs?**
- Images have spatial structure
- Local connectivity: each neuron sees only a local region
- Parameter sharing: same filter applied across entire image

**Convolution Layer:**
```
Output size = (W - F + 2P) / S + 1
```
- W = input size
- F = filter size
- P = padding
- S = stride

**Common Settings:**
- Filter size: 3×3 or 5×5
- Stride: 1 (with padding to preserve size) or 2 (for downsampling)
- Padding: (F-1)/2 to preserve spatial dimensions

**Pooling Layer:**
- Reduces spatial dimensions
- Max pooling: take maximum in each region
- Common: 2×2 with stride 2 → halves spatial dimensions

**CNN Architecture Pattern:**
```
INPUT → [[CONV → RELU]*N → POOL?]*M → [FC → RELU]*K → FC
```

**Readings:**
- [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)

---

#### Lecture 6: Training Neural Networks (Part 1)
**Topics:**
- Data preprocessing
- Weight initialization
- Batch normalization
- Regularization techniques

**Key Concepts:**

**Data Preprocessing:**
1. **Mean subtraction:** X -= np.mean(X, axis=0)
2. **Normalization:** X /= np.std(X, axis=0)
3. Apply same transform to train/val/test

**Weight Initialization:**
- Never use all zeros (symmetry breaking fails)
- Xavier/He initialization:
```python
w = np.random.randn(n) * np.sqrt(2.0/n)  # He init for ReLU
```

**Batch Normalization:**
- Normalize activations to unit Gaussian
- Insert after FC/Conv, before activation
- Enables higher learning rates
- Reduces sensitivity to initialization

**Regularization:**
- L2 weight decay: most common
- Dropout: randomly zero neurons during training
- Inverted dropout: scale by 1/p during training

**Readings:**
- [CS231n: Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/)

---

#### Recitation 3: Project Design
**Topics:**
- Project ideation
- Dataset selection
- Baseline establishment

---

### Week 4: Advanced Training Techniques

#### Lecture 7: Training Neural Networks (Part 2)
**Topics:**
- Learning rate schedules
- Advanced optimizers
- Hyperparameter search
- Monitoring training

**Key Concepts:**

**Optimizer Comparison:**
| Optimizer | Key Feature | When to Use |
|-----------|-------------|-------------|
| SGD | Simple, robust | With momentum, still competitive |
| SGD + Momentum | Accelerates consistent gradients | General use |
| Nesterov | Look-ahead gradient | Slight improvement over momentum |
| Adagrad | Per-parameter rates | Sparse gradients |
| RMSprop | Adagrad fix for deep nets | Good default |
| **Adam** | RMSprop + momentum | **Recommended default** |

**Learning Rate Schedules:**
- Step decay: Reduce by factor every N epochs
- Exponential: α = α₀ × e^(-kt)
- Cosine annealing: Smooth decay following cosine curve

**Hyperparameter Search:**
- Random search > Grid search
- Log-uniform sampling for learning rate
- Coarse-to-fine: start broad, narrow down

**Monitoring Training:**
- Loss curves: learning rate too high/low?
- Train/val gap: overfitting?
- Update ratios: ~1e-3 is healthy

**Readings:**
- [CS231n: Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/)

---

#### Lecture 8: Visualizing and Understanding CNNs
**Topics:**
- What do CNNs learn?
- Feature visualization
- Saliency maps
- Adversarial examples

**Key Concepts:**
- First layer filters learn edge detectors
- Deeper layers learn increasingly abstract features
- Gradient-based visualization techniques
- Adversarial perturbations can fool networks

---

#### Recitation 4: PyTorch Tutorial
**Topics:**
- Tensor operations
- Autograd
- nn.Module
- Training loops
- GPU acceleration

---

### Week 5: Self-Supervised Learning

#### Lectures 9-10: Self-Supervised Learning
**Topics:**
- Pretext tasks
- Contrastive learning (SimCLR, MoCo)
- BYOL and non-contrastive methods
- Benefits for downstream tasks

**Key Concepts:**

**Self-Supervised Paradigm:**
- Learn representations without labels
- Design pretext tasks that require understanding
- Transfer to downstream tasks

**Contrastive Learning:**
- Positive pairs: augmentations of same image
- Negative pairs: different images
- Learn embeddings where positives are close, negatives far

**Popular Methods:**
- SimCLR: Simple framework with heavy augmentation
- MoCo: Memory bank for more negatives
- BYOL: No negatives needed (momentum encoder)

---

#### Recitation 5: Midterm Preparation
**Topics:**
- Review of key concepts
- Practice problems
- Exam strategies

---

### Week 6: Language and Sequences

#### Lecture 11: Vision and Language + RNNs
**Topics:**
- Language modeling
- Recurrent Neural Networks
- Vanishing/exploding gradients
- LSTM and GRU

**Key Concepts:**

**RNN Formula:**
```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)
y_t = W_hy × h_t
```

**LSTM (Long Short-Term Memory):**
- Forget gate: what to discard
- Input gate: what to update
- Output gate: what to output
- Cell state: long-term memory highway

**Applications:**
- Image captioning
- Visual question answering
- Video understanding

---

#### Midterm Exam
**Topics Covered:**
- Image classification
- Linear classifiers (SVM, Softmax)
- Neural networks and backpropagation
- CNNs
- Training techniques
- Optimization

---

### Week 7: Attention and Modern Architectures

#### Lecture 12: Attention and Transformers
**Topics:**
- Attention mechanism
- Self-attention
- Transformer architecture
- Vision Transformers (ViT)

**Key Concepts:**

**Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- Query, Key, Value projections
- Scaled dot-product attention
- Multi-head attention for different relationships

**Transformer:**
- Self-attention + Feed-forward networks
- Positional encoding for sequence order
- Layer normalization and residual connections

**Vision Transformer:**
- Split image into patches
- Treat patches as tokens
- Apply standard transformer

---

#### Lecture 13: Modern Architectures
**Topics:**
- ResNet and skip connections
- DenseNet
- EfficientNet
- Neural Architecture Search

**Key Concepts:**

**ResNet Skip Connections:**
```
y = F(x) + x
```
- Enables very deep networks (100+ layers)
- Gradient highway for better optimization
- Identity mapping if F(x) = 0

**Architecture Evolution:**
- AlexNet (2012): 8 layers, 60M params
- VGGNet (2014): 19 layers, 138M params
- GoogLeNet (2014): 22 layers, 4M params (inception modules)
- ResNet (2015): 152 layers, 60M params

---

### Week 8: Detection, Segmentation, and Generation

#### Lecture 14: Structured Prediction
**Topics:**
- Object detection (R-CNN, YOLO, SSD)
- Semantic segmentation (FCN, U-Net)
- Instance segmentation (Mask R-CNN)

**Key Concepts:**

**Object Detection Pipeline:**
1. Region proposals (or grid-based)
2. Feature extraction
3. Classification + bounding box regression

**Segmentation:**
- Semantic: classify each pixel
- Instance: distinguish individual objects
- Panoptic: combine both

---

#### Lecture 15: Generative Models
**Topics:**
- Autoencoders and VAEs
- Generative Adversarial Networks (GANs)
- Diffusion models

**Key Concepts:**

**VAE:**
- Encoder: x → μ, σ
- Reparameterization: z = μ + σ × ε
- Decoder: z → x̂
- Loss: reconstruction + KL divergence

**GAN:**
- Generator: noise → fake images
- Discriminator: real vs. fake
- Adversarial training

---

#### Recitation: Foundation Models
**Topics:**
- Large-scale pretraining
- CLIP, DALL-E, GPT
- Prompt engineering

---

### Week 9: Advanced Topics

#### Lecture 16: Reinforcement Learning
**Topics:**
- MDP formulation
- Policy gradients
- Deep Q-Learning
- Applications in robotics/games

---

#### Lecture 17: Guest Lecture - Segment Anything
**Speaker:** Ross Girshick (Meta AI)
**Topics:**
- Foundation models for segmentation
- Zero-shot generalization
- Prompting for vision

---

#### Recitation: NanoGPT
**Topics:**
- Minimal GPT implementation
- Training language models
- Generation strategies

---

### Week 10: Frontier Research

#### Lecture 18: Interpretable ML
**Topics:**
- Why interpretability matters
- Attention visualization
- Concept-based explanations
- Debugging neural networks

---

#### Lecture 19: From LLMs to Self-Improving AI
**Topics:**
- Large Language Models
- Chain-of-thought reasoning
- Tool use and agents
- Future directions

---

### Week 11: Final Project

#### Final Exam
- Comprehensive coverage
- Focus on conceptual understanding

#### Project Poster Session
- Present your work
- Peer feedback

#### Final Report Due
- CVPR format (5-6 pages)

---

## Assignments

### Assignment 1: Image Classification (Due: Week 2)
**Topics:** k-NN, SVM, Softmax, Two-layer Neural Network

**Tasks:**
1. **Q1: k-Nearest Neighbor** - Implement kNN classifier with vectorized distance computation
2. **Q2: SVM** - Implement multiclass SVM loss and gradient
3. **Q3: Softmax** - Implement cross-entropy loss and gradient
4. **Q4: Two-Layer Net** - Build a neural network from scratch
5. **Q5: Feature Representations** - Compare raw pixels vs. HOG/color features

**Learning Goals:**
- Understand data-driven classification
- Master vectorized NumPy operations
- Implement gradient computation
- Compare classifier performance

---

### Assignment 2: Neural Networks (Due: Week 5)
**Topics:** Deep networks, Batch Norm, Dropout, CNNs, PyTorch

**Tasks:**
1. **Q1: Fully-Connected Networks** - Modular layer implementation
2. **Q2: Batch Normalization** - Forward/backward implementation
3. **Q3: Dropout** - Regularization implementation
4. **Q4: Convolutional Networks** - Conv and pooling layers
5. **Q5: PyTorch** - Train CNN on CIFAR-10
6. **Q6: Network Visualization** - Saliency maps, adversarial examples

**Learning Goals:**
- Build modular network components
- Understand training dynamics
- Use deep learning frameworks
- Visualize learned representations

---

### Assignment 3: RNNs, Transformers, and Genertic Models (Due: Week 9)
**Topics:** RNNs, Transformers, GANs, Self-supervised Learning

**Tasks:**
1. **Q1: Vanilla RNNs** - Image captioning on COCO
2. **Q2: Transformers** - Attention-based captioning
3. **Q3: GANs** - Generate images
4. **Q4: Self-Supervised Learning** - Contrastive pretraining
5. **Extra Credit: LSTM** - Implement LSTM cells

**Learning Goals:**
- Implement sequence models
- Understand attention mechanisms
- Train generative models
- Apply self-supervised learning

---

## Final Project (35% of Grade)

### Timeline
| Milestone | Due Date | Weight |
|-----------|----------|--------|
| Proposal | Week 4 | 5% |
| Milestone Report | Week 8 | 5% |
| Poster Session | Week 11 | 10% |
| Final Report | Week 11 | 15% |

### Project Ideas
- Novel model architecture for a vision task
- New dataset creation and benchmarking
- Improving existing methods
- Application to real-world problem
- Reproduction and extension of research paper

### Format
- IEEE CVPR conference paper format
- 5-6 pages (excluding references)
- Sections: Abstract, Introduction, Related Work, Methods, Experiments, Discussion

---

## Essential Reading Materials

### Core Course Notes (CS231n)
1. [Image Classification](https://cs231n.github.io/classification/) - Data-driven approach, k-NN
2. [Linear Classification](https://cs231n.github.io/linear-classify/) - SVM, Softmax, loss functions
3. [Optimization](https://cs231n.github.io/optimization-1/) - Gradient descent
4. [Backpropagation](https://cs231n.github.io/optimization-2/) - Chain rule, computational graphs
5. [Neural Networks 1](https://cs231n.github.io/neural-networks-1/) - Architecture, activations
6. [Neural Networks 2](https://cs231n.github.io/neural-networks-2/) - Preprocessing, initialization, regularization
7. [Neural Networks 3](https://cs231n.github.io/neural-networks-3/) - Training dynamics, hyperparameters
8. [Convolutional Networks](https://cs231n.github.io/convolutional-networks/) - CNN architecture
9. [Transfer Learning](https://cs231n.github.io/transfer-learning/) - Fine-tuning pretrained models

### Textbooks
- **Deep Learning** by Goodfellow, Bengio, Courville - [deeplearningbook.org](https://www.deeplearningbook.org/)
- **Dive into Deep Learning** - [d2l.ai](https://d2l.ai/)

### Key Research Papers
1. **AlexNet** (2012) - ImageNet Classification with Deep CNNs
2. **VGGNet** (2014) - Very Deep Convolutional Networks
3. **ResNet** (2015) - Deep Residual Learning
4. **Batch Normalization** (2015) - Ioffe & Szegedy
5. **Dropout** (2014) - Srivastava et al.
6. **Adam Optimizer** (2014) - Kingma & Ba
7. **Attention Is All You Need** (2017) - Transformer architecture
8. **BERT** (2018) - Bidirectional Transformers
9. **Vision Transformer** (2020) - ViT
10. **SimCLR** (2020) - Contrastive self-supervised learning

---

## Study Tips

### For Assignments
1. Read assignment instructions completely before starting
2. Understand the math before coding
3. Use gradient checking to verify implementations
4. Start early - debugging can be time-consuming
5. Use office hours and EdStem

### For Exams
1. Understand concepts, don't just memorize
2. Practice deriving gradients by hand
3. Know computational graph patterns
4. Understand tradeoffs between methods
5. Review lecture slides and practice problems

### For Project
1. Start with a working baseline
2. Iterate quickly with small experiments
3. Track all experiments systematically
4. Read related papers thoroughly
5. Communicate progress early with TAs

---

## Tools and Resources

### Software Setup
- Python 3.8+
- NumPy, Matplotlib
- PyTorch (primary framework)
- Jupyter/Google Colab
- CUDA/GPU access

### Compute Resources
- Google Colab (free GPU)
- Kaggle Notebooks
- AWS/GCP credits (if available)
- University GPU clusters

### Useful Links
- [Course Website](https://courses.cs.washington.edu/courses/cse493g1/23sp/)
- [CS231n Notes](https://cs231n.github.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Papers With Code](https://paperswithcode.com/)

---

## Quick Reference: Key Formulas

### Loss Functions
```
SVM Loss:     L_i = Σ_{j≠y} max(0, s_j - s_y + 1)
Softmax Loss: L_i = -log(exp(s_y) / Σ_j exp(s_j))
```

### Gradient Descent
```
θ = θ - α × ∇L(θ)
```

### Batch Normalization
```
μ = (1/m) Σ x_i
σ² = (1/m) Σ (x_i - μ)²
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β
```

### Convolution Output Size
```
O = (W - F + 2P) / S + 1
```

### Attention
```
Attention(Q,K,V) = softmax(QK^T / √d_k) × V
```

---

*Good luck with your deep learning journey!*
