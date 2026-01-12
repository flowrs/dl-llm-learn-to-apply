# Week 1-2: Deep Learning Foundations

> Combined guide merging CSE 493G1/CS231n course materials with Stanford CS231n notes

## Overview

This document covers the foundational concepts of deep learning:
- Image classification and the data-driven approach
- Nearest Neighbor and k-NN classifiers
- Linear classifiers (SVM and Softmax)
- Neural networks and backpropagation
- Optimization fundamentals

---

## Part 1: Image Classification

### The Problem

**Image Classification** is the task of assigning an input image one label from a fixed set of categories. Despite its simplicity, it has enormous practical applications and forms the foundation for more complex tasks like detection and segmentation.

```
                    Image Classification Pipeline
                    ═════════════════════════════

    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │   INPUT     │      │  LEARNING   │      │ EVALUATION  │
    │  N images   │ ──▶  │  Train a    │ ──▶  │  Predict on │
    │  K classes  │      │  classifier │      │  test set   │
    └─────────────┘      └─────────────┘      └─────────────┘
```

### Challenges in Image Classification

Images are represented as 3D arrays of numbers (Width × Height × 3 RGB channels). A computer must handle:

- **Viewpoint variation**: Objects can be oriented in many ways
- **Scale variation**: Objects appear at different sizes
- **Deformation**: Non-rigid objects can bend and twist
- **Occlusion**: Objects may be partially hidden
- **Illumination**: Lighting dramatically affects pixel values
- **Background clutter**: Objects may blend into surroundings
- **Intra-class variation**: Categories like "chair" have many forms

### The Data-Driven Approach

Instead of hardcoding rules, we:
1. Collect a dataset of labeled images (training set)
2. Use machine learning to train a classifier
3. Evaluate on new, unseen images (test set)

```python
# The universal classifier API
class Classifier:
    def train(self, X, y):
        """Learn from training data X with labels y"""
        pass

    def predict(self, X):
        """Predict labels for new data X"""
        pass
```

---

## Part 2: Nearest Neighbor Classifier

### Basic Idea

The simplest classifier: predict the label of the most similar training image.

```
    Nearest Neighbor Classifier
    ═══════════════════════════

    Test Image          Training Set
    ┌─────┐            ┌─────┬─────┬─────┬─────┐
    │  ?  │  ──find──▶ │ cat │ dog │ car │ ... │
    └─────┘   closest  └─────┴─────┴─────┴─────┘
                              │
                              ▼
                        Predict: cat
```

### Distance Metrics

**L1 Distance (Manhattan)**:
$$d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$$

**L2 Distance (Euclidean)**:
$$d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$$

```python
# L1 distance implementation
distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)

# L2 distance implementation
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1))
```

### k-Nearest Neighbor (k-NN)

Instead of using just the closest neighbor, vote among the k closest:

```
    k-NN Decision Boundaries (k=1 vs k=5)
    ═════════════════════════════════════

    k=1: Noisy boundaries          k=5: Smoother boundaries
    ┌────────────────────┐         ┌────────────────────┐
    │ ●●●○○○●●●○○        │         │ ●●●●●●●●○○○        │
    │ ●●●●○○○●○○○        │         │ ●●●●●●○○○○○        │
    │ ●●●●●○○○○○○        │         │ ●●●●●○○○○○○        │
    │ ●●●●●●○○○○○        │         │ ●●●●○○○○○○○        │
    └────────────────────┘         └────────────────────┘
    (outliers create islands)      (smoothed, better generalization)
```

### Hyperparameter Tuning

**Critical Rule**: Never use the test set for tuning hyperparameters!

```
    Correct Data Split
    ══════════════════

    ┌─────────────────────────────────────────────────────────┐
    │                    All Data                              │
    ├─────────────────────────────────┬───────────┬───────────┤
    │         Training Set            │Validation │   Test    │
    │           (learn)               │  (tune)   │ (report)  │
    │            ~70%                 │   ~15%    │   ~15%    │
    └─────────────────────────────────┴───────────┴───────────┘
```

**Cross-Validation**: For small datasets, use k-fold cross-validation:
- Split training data into k folds
- Train on k-1 folds, validate on remaining fold
- Rotate and average results

### Pros and Cons of k-NN

| Pros | Cons |
|------|------|
| Simple to implement | Slow at test time (compare to all training) |
| No training time | Memory intensive (store all data) |
| Works on any data | Pixel distances are semantically meaningless |

**Key Insight**: k-NN achieves ~38% on CIFAR-10 (humans: 94%, CNNs: 95%+). Pixel-based distances don't capture semantic similarity.

---

## Part 3: Linear Classification

### The Score Function

Linear classifiers map images to class scores using a simple function:

$$f(x_i, W, b) = Wx_i + b$$

```
    Linear Classifier Score Function
    ════════════════════════════════

    Image (flattened)     Weights        Bias      Scores
    [3072 × 1]          [10 × 3072]   [10 × 1]   [10 × 1]
         x          ×        W       +    b    =    s

    ┌───┐           ┌───────────┐   ┌───┐     ┌───┐
    │   │           │ class 0   │   │   │     │3.2│ cat
    │ p │           │ class 1   │   │   │     │1.3│ dog
    │ i │     ×     │ class 2   │ + │ b │  =  │2.1│ ship
    │ x │           │    ...    │   │   │     │...│
    │   │           │ class 9   │   │   │     │0.8│ frog
    └───┘           └───────────┘   └───┘     └───┘
```

### Interpreting Linear Classifiers

**Template Matching View**: Each row of W acts as a template for one class.

```
    Learned Templates (CIFAR-10)
    ════════════════════════════

    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
    │plane│  │ car │  │horse│  │ship │  │truck│
    │blue │  │ red │  │2head│  │blue │  │ red │
    │back │  │blob │  │blob │  │water│  │blob │
    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘

    Linear classifiers can only learn ONE template per class
    (hence the two-headed horse from multiple orientations)
```

### Loss Functions

#### Multiclass SVM Loss (Hinge Loss)

Wants correct class to beat others by margin Δ:

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

```python
def svm_loss(scores, y, delta=1.0):
    correct_score = scores[y]
    margins = np.maximum(0, scores - correct_score + delta)
    margins[y] = 0  # Don't count correct class
    return np.sum(margins)
```

#### Softmax Loss (Cross-Entropy)

Interprets scores as log-probabilities:

$$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

```python
def softmax_loss(scores, y):
    # Numerical stability: shift scores
    scores -= np.max(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores))
    return -np.log(probs[y])
```

### SVM vs Softmax

```
    SVM vs Softmax Behavior
    ═══════════════════════

    Scores: [10, 9, 9] for classes [correct, wrong1, wrong2]

    SVM:     Loss = 0 (margin satisfied, done)
    Softmax: Loss > 0 (always wants higher probability for correct)

    SVM is "satisfied" once margins are met
    Softmax continuously pushes for improvement
```

### Regularization

Prevents overfitting by penalizing large weights:

$$L = \frac{1}{N}\sum_i L_i + \lambda R(W)$$

**L2 Regularization**: $R(W) = \sum_{k,l} W_{k,l}^2$
- Prefers smaller, diffuse weights
- Encourages using all features

**L1 Regularization**: $R(W) = \sum_{k,l} |W_{k,l}|$
- Encourages sparse weights
- Feature selection effect

---

## Part 4: Neural Networks

### From Linear to Non-Linear

```
    Evolution: Linear → Neural Network
    ══════════════════════════════════

    Linear:    s = Wx

    2-Layer:   s = W₂ max(0, W₁x)
                      └──────┘
                      non-linearity!

    3-Layer:   s = W₃ max(0, W₂ max(0, W₁x))
```

**Key Insight**: Without non-linearity, multiple linear layers collapse to one!

### The Neuron

```
    Single Neuron Model
    ═══════════════════

    Inputs      Weights     Sum + Bias    Activation    Output

    x₀ ──w₀──┐
             │
    x₁ ──w₁──┼──▶ Σ + b ──▶ f(·) ──▶ output
             │
    x₂ ──w₂──┘

    output = f(Σᵢ wᵢxᵢ + b)
```

### Activation Functions

```
    Activation Functions Comparison
    ═══════════════════════════════

    Sigmoid: σ(x) = 1/(1+e⁻ˣ)        Tanh: tanh(x)
    ┌────────────────────┐           ┌────────────────────┐
    │         ___________│           │         ___________│
    │        /           │           │        /           │
    │       /            │           │───────/────────────│
    │______/             │           │      /             │
    │                    │           │_____/              │
    └────────────────────┘           └────────────────────┘
    Range: [0, 1]                    Range: [-1, 1]
    Problems: Vanishing gradients,   Better: zero-centered
              not zero-centered      Still: vanishing gradients

    ReLU: max(0, x)                  Leaky ReLU: max(0.01x, x)
    ┌────────────────────┐           ┌────────────────────┐
    │               /    │           │               /    │
    │              /     │           │              /     │
    │             /      │           │         ___/       │
    │____________/       │           │________/           │
    │                    │           │                    │
    └────────────────────┘           └────────────────────┘
    Simple, fast, no saturation      Fixes "dying ReLU" problem
    Problem: Dead neurons
```

**Recommendation**: Use ReLU. Be careful with learning rates. Never use sigmoid.

### Network Architecture

```
    Fully-Connected Neural Network
    ══════════════════════════════

    INPUT        HIDDEN         HIDDEN         OUTPUT
    [3072]        [100]          [100]          [10]

      ●────────────●────────────●────────────●
      ●────────────●────────────●────────────●
      ●────────────●────────────●────────────●
      ⋮            ⋮            ⋮            ⋮
      ●────────────●────────────●────────────●

    Every neuron connects to ALL neurons in adjacent layers

    Parameters:
    - Layer 1: 3072 × 100 + 100 = 307,300
    - Layer 2: 100 × 100 + 100 = 10,100
    - Layer 3: 100 × 10 + 10 = 1,010
    - Total: 318,410 parameters
```

### Forward Pass

```python
# 3-layer neural network forward pass
def forward(x, W1, b1, W2, b2, W3, b3):
    # Layer 1
    h1 = np.maximum(0, np.dot(W1, x) + b1)  # ReLU
    # Layer 2
    h2 = np.maximum(0, np.dot(W2, h1) + b2)  # ReLU
    # Output (no activation for scores)
    out = np.dot(W3, h2) + b3
    return out
```

### Universal Approximation

Neural networks with one hidden layer can approximate any continuous function. But:
- More layers = more efficient representations
- Deep networks learn hierarchical features
- Depth matters more than width for many problems

---

## Part 5: Backpropagation

### The Chain Rule

Backpropagation computes gradients using the chain rule:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

```
    Computational Graph and Gradients
    ══════════════════════════════════

    Forward Pass (compute values):

    x ──▶ [×] ──▶ z ──▶ [+] ──▶ q ──▶ [f] ──▶ L
           ▲            ▲
           │            │
           w            b

    Backward Pass (compute gradients):

         ∂L/∂z     ∂L/∂q     ∂L/∂L=1
           ◀──────   ◀──────   ◀──────
           local ×   local ×
           upstream  upstream
```

### Gradient Patterns

```
    Common Gradient Patterns
    ════════════════════════

    ADD gate:      Distributes gradient equally
    [x + y]        ∂L/∂x = upstream, ∂L/∂y = upstream

    MUL gate:      Swaps and multiplies
    [x × y]        ∂L/∂x = y × upstream, ∂L/∂y = x × upstream

    MAX gate:      Routes gradient to max input
    [max(x,y)]     ∂L/∂x = upstream if x>y else 0

    ReLU:          Gradient "gate"
    [max(0,x)]     ∂L/∂x = upstream if x>0 else 0
```

### Backprop Implementation

```python
# Simple backprop example
def forward_backward(x, W, y):
    # Forward pass
    scores = np.dot(W, x)
    probs = softmax(scores)
    loss = -np.log(probs[y])

    # Backward pass
    dscores = probs.copy()
    dscores[y] -= 1        # Gradient of softmax loss
    dW = np.outer(dscores, x)  # Gradient for weights
    dx = np.dot(W.T, dscores)  # Gradient for input

    return loss, dW, dx
```

---

## Part 6: Optimization

### Gradient Descent

Update parameters in the direction of steepest descent:

$$\theta = \theta - \alpha \nabla L(\theta)$$

```python
# Vanilla gradient descent
for i in range(num_iterations):
    gradient = compute_gradient(data, weights)
    weights -= learning_rate * gradient
```

### Mini-Batch SGD

Use small batches instead of full dataset:

```python
# Mini-batch stochastic gradient descent
for i in range(num_iterations):
    batch = sample(training_data, batch_size=32)
    gradient = compute_gradient(batch, weights)
    weights -= learning_rate * gradient
```

**Why mini-batches?**
- Full batch: stable but slow
- Single example: fast but noisy
- Mini-batch: good trade-off

### Learning Rate

```
    Learning Rate Effects
    ═════════════════════

    Too High:                    Too Low:
    Loss                         Loss
     │\    /\    /\              │\
     │ \  /  \  /  \             │ \
     │  \/    \/    \            │  \____________________
     │                           │
     └──────────────────         └──────────────────────
          time                         time
    (diverges/oscillates)        (very slow convergence)

    Just Right:
    Loss
     │\
     │ \
     │  \___
     │      \___________
     └──────────────────
          time
    (steady decrease, then plateau)
```

---

## Summary

| Concept | Key Points |
|---------|------------|
| **Image Classification** | Data-driven approach, train/val/test splits |
| **k-NN** | Simple baseline, O(N) test time, semantic distance issue |
| **Linear Classifiers** | Score function f=Wx+b, templates per class |
| **Loss Functions** | SVM (margin-based), Softmax (probabilistic) |
| **Regularization** | L2/L1 penalties prevent overfitting |
| **Neural Networks** | Non-linear, hierarchical feature learning |
| **Activation Functions** | ReLU preferred, enables non-linearity |
| **Backpropagation** | Chain rule, computational graphs |
| **Optimization** | SGD, learning rate critical |

---

## Practical Checklist

### Before Training
- [ ] Normalize data (zero mean, unit variance)
- [ ] Split data: train/val/test
- [ ] Sanity check: overfit small batch first
- [ ] Initialize weights properly (Xavier/He)

### During Training
- [ ] Monitor loss curves
- [ ] Watch for train/val gap (overfitting)
- [ ] Adjust learning rate if needed
- [ ] Use gradient checking for debugging

### Hyperparameter Search
- [ ] Use random search, not grid search
- [ ] Log-scale for learning rate
- [ ] Coarse-to-fine: broad first, then narrow

---

## References

**Course Materials:**
- CS231n: https://cs231n.github.io/
- CSE 493G1: https://courses.cs.washington.edu/courses/cse493g1/

**Key Readings:**
- [Image Classification Notes](https://cs231n.github.io/classification/)
- [Linear Classification Notes](https://cs231n.github.io/linear-classify/)
- [Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- [Optimization Notes](https://cs231n.github.io/optimization-1/)
- [Backpropagation Notes](https://cs231n.github.io/optimization-2/)

**Papers:**
- AlexNet (2012): ImageNet Classification with Deep CNNs
- Dropout (2014): Srivastava et al.
- Batch Normalization (2015): Ioffe & Szegedy
