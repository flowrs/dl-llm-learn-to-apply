# Module 1: Neural Network Foundations

## Learning Objectives

By the end of this module, you will understand:
- What neural networks are and how they compute
- The role of neurons, weights, biases, and activation functions
- How to construct multi-layer networks
- The forward pass and score function computation

---

## 1.1 From Linear Classifiers to Neural Networks

### The Limitations of Linear Models

In linear classification, we compute scores using a simple formula:

```
s = Wx
```

Where:
- `W` is a weight matrix [K × D] (K classes, D features)
- `x` is the input vector [D × 1]
- `s` is the score vector [K × 1]

For CIFAR-10 images (32×32×3 = 3072 pixels, 10 classes):
- `x` is [3072 × 1]
- `W` is [10 × 3072]
- `s` is [10 × 1] class scores

**Problem**: Linear classifiers can only learn linear decision boundaries. Real-world data often requires non-linear boundaries.

### Adding Non-linearity

A neural network adds intermediate transformations with non-linearities:

```
s = W₂ · max(0, W₁x)
```

Here:
- `W₁` transforms input to intermediate representation (e.g., [100 × 3072])
- `max(0, ·)` is the ReLU non-linearity (applied element-wise)
- `W₂` transforms intermediate to output (e.g., [10 × 100])

The non-linearity is **critical**. Without it, the two matrices collapse to one:
```
W₂(W₁x) = (W₂W₁)x = Wx  (still linear!)
```

---

## 1.2 The Neuron

### Mathematical Model

A single neuron computes:

```
y = f(Σᵢ wᵢxᵢ + b) = f(w·x + b)
```

Where:
- **Inputs** `x = [x₁, x₂, ..., xₙ]`: values from previous layer or data
- **Weights** `w = [w₁, w₂, ..., wₙ]`: learnable parameters controlling input strength
- **Bias** `b`: learnable offset allowing the neuron to shift its activation
- **Activation function** `f`: non-linear transformation
- **Output** `y`: single scalar value

```
        x₁ ----w₁--→
                    \
        x₂ ----w₂--→ [Σ + b] --→ [f] --→ y
                    /
        x₃ ----w₃--→

         Inputs     Weights    Sum+Bias  Activation  Output
```

### Python Implementation

```python
class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = 0

    def forward(self, inputs):
        """Compute neuron output"""
        # Weighted sum
        weighted_sum = np.sum(inputs * self.weights) + self.bias
        # Activation (sigmoid in this example)
        output = 1.0 / (1.0 + np.exp(-weighted_sum))
        return output
```

### A Neuron as a Linear Classifier

A single neuron with sigmoid activation can perform binary classification:
- Output near 1: predict class 1
- Output near 0: predict class 0

The decision boundary is where `w·x + b = 0` (a hyperplane).

---

## 1.3 Activation Functions

Activation functions introduce non-linearity, enabling networks to learn complex patterns.

### Sigmoid

```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Properties:**
- Range: (0, 1)
- Historically popular (biological interpretation as "firing rate")

**Problems:**
1. **Vanishing gradients**: At extreme values, gradient ≈ 0 (neurons "saturate")
2. **Not zero-centered**: All outputs positive, causing zig-zag gradient updates

```
Output:  1 |      ___________
           |    /
         0 |___/
           +----------------→ x
              -4   0    4
```

### Tanh

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ) = 2σ(2x) - 1
```

**Properties:**
- Range: (-1, 1)
- **Zero-centered** (preferred over sigmoid)
- Still suffers from saturation at extremes

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

**Properties:**
- Range: [0, ∞)
- **No saturation** in positive region
- **Computationally efficient** (just thresholding)
- **Sparse activations** (many neurons output 0)

**Problem: "Dying ReLU"**
- Large negative inputs → output always 0 → gradient always 0
- Neuron stops learning permanently

```
Output:    |     /
           |    /
           |   /
         0 |__/
           +----------------→ x
```

### Leaky ReLU

```
LeakyReLU(x) = max(αx, x)  where α ≈ 0.01
```

Allows small gradient when input is negative, preventing dying neurons.

### GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x)  where Φ is the standard normal CDF
```

Used in modern transformers (BERT, GPT). Smooth approximation that weights inputs by their value.

### Softmax

```
softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ
```

Converts a vector of scores into probabilities (sum to 1). Used in output layer for multi-class classification.

### Recommendations

| Situation | Recommendation |
|-----------|---------------|
| Hidden layers | ReLU (default), Leaky ReLU if dying neurons |
| Transformers | GELU |
| Output (classification) | Softmax |
| Output (binary) | Sigmoid |
| Never | Sigmoid in hidden layers |

---

## 1.4 Neural Network Architectures

### Layer-wise Organization

Neural networks organize neurons into **layers**:

```
Input Layer    Hidden Layer(s)    Output Layer
    ●               ●                  ●
    ●               ●                  ●
    ●               ●                  ●
    ●               ●
                    ●
```

**Fully-connected (FC) layers**: Every neuron connects to every neuron in adjacent layers.

### Naming Convention

- **N-layer network**: N learnable weight layers (not counting input)
- **2-layer network**: 1 hidden layer + 1 output layer
- **Deep learning**: Multiple hidden layers

### Network Dimensions

Consider a 3-layer network for CIFAR-10:

```
Input:  [3072]  (32×32×3 pixels)
   ↓ W₁: [100 × 3072]
Hidden1: [100]
   ↓ W₂: [100 × 100]
Hidden2: [100]
   ↓ W₃: [10 × 100]
Output: [10]  (class scores)
```

**Parameter count:**
- W₁: 3072 × 100 = 307,200
- b₁: 100
- W₂: 100 × 100 = 10,000
- b₂: 100
- W₃: 100 × 10 = 1,000
- b₃: 10
- **Total: 318,410 parameters**

### Forward Pass Computation

```python
def forward_pass(x, W1, b1, W2, b2, W3, b3):
    """3-layer neural network forward pass"""
    # Layer 1
    z1 = np.dot(W1, x) + b1   # Linear transformation
    h1 = np.maximum(0, z1)     # ReLU activation

    # Layer 2
    z2 = np.dot(W2, h1) + b2  # Linear transformation
    h2 = np.maximum(0, z2)     # ReLU activation

    # Output layer (no activation for scores)
    scores = np.dot(W3, h2) + b3

    return scores
```

**Key insight**: The forward pass is just matrix multiplications interwoven with activation functions.

---

## 1.5 Representational Power

### Universal Approximation Theorem

A neural network with **one hidden layer** and enough neurons can approximate any continuous function to arbitrary precision.

**However**, this doesn't mean single-layer networks are optimal:
- May require exponentially many neurons
- Deeper networks learn hierarchical representations more efficiently
- Empirically, depth helps more than width

### Why Depth Matters

Deep networks learn **hierarchical features**:

```
Layer 1: Edges, corners
Layer 2: Simple shapes, textures
Layer 3: Object parts
Layer 4: Objects
Layer 5: Scenes, concepts
```

This mirrors how complex concepts are built from simpler ones.

---

## 1.6 Network Sizing Guidelines

### Choosing Architecture

**Width** (neurons per layer):
- More neurons → more capacity → can learn more complex functions
- Too many → overfitting, slower training

**Depth** (number of layers):
- More layers → hierarchical features → better for complex data
- Diminishing returns after certain depth (depends on problem)

### Overfitting vs. Underfitting

```
                    |
        Underfitting|        Overfitting
                    |
        (too simple)|        (too complex)
                    |
Error               |    ∪
                    |   /|\
                    |  / | \
                    | /  |  \
                    |/   |   \---
        ____________|____|________
                    |
                Model Complexity
                    |
              Optimal Point
```

**Key principle**: Don't reduce network size to prevent overfitting. Instead:
- Use larger networks
- Apply regularization (L2, dropout, etc.)

Larger networks have better loss landscapes with more "good" local minima.

---

## 1.7 Tensors: The Data Structure of Deep Learning

### What is a Tensor?

A **tensor** is a multi-dimensional array:

```
Scalar (0D):  5
Vector (1D):  [1, 2, 3]
Matrix (2D):  [[1, 2], [3, 4]]
3D Tensor:    [[[1,2], [3,4]], [[5,6], [7,8]]]
```

### Common Tensor Shapes in Deep Learning

| Data Type | Shape | Example |
|-----------|-------|---------|
| Grayscale image | [H, W] | [28, 28] for MNIST |
| Color image | [H, W, C] or [C, H, W] | [224, 224, 3] |
| Batch of images | [N, C, H, W] | [32, 3, 224, 224] |
| Text sequence | [N, L] | [16, 512] (batch, length) |
| Embeddings | [N, L, D] | [16, 512, 768] |

### PyTorch Tensor Operations

```python
import torch

# Create tensors
x = torch.randn(32, 3, 224, 224)  # Batch of 32 RGB images
W = torch.randn(64, 3, 3, 3)      # 64 conv filters

# Common operations
x.shape           # torch.Size([32, 3, 224, 224])
x.reshape(32, -1) # Flatten: [32, 150528]
x.permute(0, 2, 3, 1)  # Reorder dims: [32, 224, 224, 3]
x.mean(dim=1)     # Average over channels
```

---

## 1.8 Summary

### Key Concepts

1. **Neural Networks** extend linear classifiers with non-linear activation functions
2. **Neurons** compute weighted sums followed by activation: `y = f(w·x + b)`
3. **Activation functions** (ReLU, GELU, etc.) introduce non-linearity
4. **Layers** organize neurons; fully-connected layers connect all neurons
5. **Forward pass** computes output from input via matrix operations
6. **Tensors** are multi-dimensional arrays, the fundamental data structure

### Glossary Terms Covered

- Artificial Neural Network (ANN)
- Neuron (Node, Unit)
- Weight
- Bias
- Parameter
- Layer
- Deep Learning
- Forward Pass
- Activation Function
- ReLU, Sigmoid, Tanh, Softmax, GELU
- Tensor
- Feedforward Neural Network

### What's Next

Module 2 covers **training**: how to learn the weights through backpropagation, gradient descent, and optimization algorithms.

---

## Exercises

1. **Calculate parameters**: A network has layers [784, 256, 128, 10]. How many total parameters?

2. **ReLU by hand**: Compute ReLU([-2, 0.5, 3, -1, 0])

3. **Softmax**: Given scores [2.0, 1.0, 0.1], compute softmax probabilities.

4. **Code**: Implement a 2-layer neural network forward pass in NumPy.

---

## References

- CS231n: Neural Networks Part 1
- CS224N: Lecture 3 - Backpropagation and Neural Networks
- Goodfellow et al., "Deep Learning" Chapter 6
