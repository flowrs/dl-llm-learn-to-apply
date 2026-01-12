# Module 2: Training & Optimization

## Learning Objectives

By the end of this module, you will understand:
- Loss functions and how they quantify model error
- Gradient descent and its variants (SGD, momentum, Adam)
- Backpropagation: computing gradients efficiently
- Weight initialization strategies
- Regularization techniques (L2, dropout, batch normalization)
- Practical training considerations and debugging

---

## 2.1 The Training Problem

### What Does "Training" Mean?

Training is the process of finding parameter values that make our network perform well on a task. We need three components:

```
The Training Setup:
────────────────────────────────────────────────────────────────

1. SCORE FUNCTION (Forward Pass)
   Maps inputs to predictions
   ┌─────────────────────────────────────────────┐
   │  f(x; W) : input x → predictions ŷ         │
   │                                             │
   │  Example: f(image; W) → class scores       │
   └─────────────────────────────────────────────┘
                    ↓

2. LOSS FUNCTION (How wrong are we?)
   Measures prediction quality
   ┌─────────────────────────────────────────────┐
   │  L(ŷ, y) : (prediction, truth) → scalar    │
   │                                             │
   │  Example: Cross-entropy loss               │
   └─────────────────────────────────────────────┘
                    ↓

3. OPTIMIZATION (Find better parameters)
   Adjust W to minimize loss
   ┌─────────────────────────────────────────────┐
   │  W* = argmin_W Σᵢ L(f(xᵢ; W), yᵢ)          │
   │                                             │
   │  Method: Gradient descent                  │
   └─────────────────────────────────────────────┘
```

### The Loss Landscape

Imagine plotting loss as a function of all parameters:

```
Loss Landscape Visualization (2D slice):
────────────────────────────────────────────────────────────────

Loss L
    │
    │     ╱╲        ╱╲
    │    ╱  ╲      ╱  ╲         Global
    │   ╱    ╲    ╱    ╲        Minimum
    │  ╱      ╲  ╱      ╲        ↓
    │ ╱        ╲╱        ╲____●____
    │╱                              ╲
    └───────────────────────────────────→ W (parameters)
              ↑
           Local
           Minimum

We want to find the lowest point (minimum loss).
Challenge: High-dimensional space, many local minima.
```

---

## 2.2 Loss Functions

### Cross-Entropy Loss (Softmax Loss)

The most common loss for multi-class classification:

```
Cross-Entropy Loss Explained:
────────────────────────────────────────────────────────────────

Step 1: Network outputs raw scores (logits)
        s = [2.0, 1.0, 0.1]  (for 3 classes)

Step 2: Convert to probabilities via softmax
        p = softmax(s) = [e^2.0, e^1.0, e^0.1] / Z
                       = [7.39, 2.72, 1.11] / 11.22
                       = [0.659, 0.242, 0.099]

Step 3: Compute negative log probability of correct class
        If true class is 0:
        L = -log(p[0]) = -log(0.659) = 0.417

        If true class is 1:
        L = -log(p[1]) = -log(0.242) = 1.419

        If true class is 2:
        L = -log(p[2]) = -log(0.099) = 2.312


Intuition:
────────────────────────────────────────────────────────────────

-log(p) penalizes low confidence in the correct class:

    p (confidence)     -log(p) (loss)
    ────────────────────────────────
    1.0 (perfect)      0.0
    0.9 (high)         0.105
    0.5 (medium)       0.693
    0.1 (low)          2.303
    0.01 (very low)    4.605
    → 0                → ∞

    High confidence in wrong class = very high loss!
```

**Mathematical Formula**:

```
Cross-Entropy = -log(p_y) = -log(e^{s_y} / Σⱼ e^{s_j})
              = -s_y + log(Σⱼ e^{s_j})
```

**Numerical Stability**: Always subtract max before exp to prevent overflow.

```python
def cross_entropy_loss(scores, y):
    """
    Compute cross-entropy loss.

    Args:
        scores: [N, C] raw scores (logits)
        y: [N] true class indices

    Returns:
        loss: scalar average loss
    """
    N = scores.shape[0]

    # Numerical stability: subtract max
    scores_stable = scores - np.max(scores, axis=1, keepdims=True)

    # Softmax
    exp_scores = np.exp(scores_stable)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Cross-entropy
    correct_log_probs = -np.log(probs[range(N), y])
    loss = np.mean(correct_log_probs)

    return loss, probs
```

### Mean Squared Error (MSE)

For regression tasks:

```
MSE Loss:
────────────────────────────────────────────────────────────────

L = (1/N) Σᵢ (ŷᵢ - yᵢ)²

Example:
    Predictions: [2.5, 3.0, 4.5]
    Targets:     [2.0, 3.5, 4.0]

    Errors:      [0.5, -0.5, 0.5]
    Squared:     [0.25, 0.25, 0.25]

    MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25


When to use:
────────────────────────────────────────────────────────────────

Task                    Loss Function
─────────────────────────────────────
Classification          Cross-entropy (preferred)
Binary classification   Binary cross-entropy
Regression             MSE (or Huber for robustness)
```

---

## 2.3 Gradient Descent

### The Core Idea

Move parameters in the direction that reduces loss:

```
Gradient Descent Intuition:
────────────────────────────────────────────────────────────────

The GRADIENT ∇L tells us:
  - Direction of steepest INCREASE in loss
  - Magnitude of the slope

To DECREASE loss, move OPPOSITE the gradient:

    W_new = W_old - α × ∇L

    Where α = learning rate (step size)


Visual:
────────────────────────────────────────────────────────────────

Loss
  │
  │    ╲
  │     ╲    ← Gradient points "uphill"
  │      ╲
  │       ╲  ● Current position
  │        ╲ │
  │         ╲│ Move opposite = downhill
  │          ↓
  │           ╲
  │            ╲
  │             ╲____●  New position (lower loss!)
  └──────────────────────────────────→ W
```

### The Gradient

For a loss function L and parameters W:

```
Gradient Computation:
────────────────────────────────────────────────────────────────

∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]

Each component tells us:
  "How much does L change if I change wᵢ slightly?"

Example:
  If ∂L/∂w₁ = 2.5, then:
  - Increasing w₁ by 0.01 increases L by ~0.025
  - Decreasing w₁ by 0.01 decreases L by ~0.025

  → To reduce loss, decrease w₁!
```

### Learning Rate: The Most Important Hyperparameter

```
Learning Rate Effects:
────────────────────────────────────────────────────────────────

Too SMALL (α = 0.0001):
Loss │
     │╲
     │ ╲
     │  ╲
     │   ╲
     │    ╲
     │     ╲
     │      ↓ Very slow progress
     └──────────────→ Iterations

     Problem: Takes forever to converge
     May get stuck in bad local minima


Just RIGHT (α = 0.01):
Loss │
     │╲
     │  ╲
     │    ╲
     │      ╲___
     │          ‾‾‾───___
     │                    ‾‾‾────
     └──────────────────────────→ Iterations

     Steady progress, converges well


Too LARGE (α = 1.0):
Loss │
     │    ╱╲      ╱╲
     │   ╱  ╲    ╱  ╲
     │  ╱    ╲  ╱    ╲
     │ ╱      ╲╱      ╲
     │╱                 ╲╱ Oscillates!
     └──────────────────────────→ Iterations

     Problem: Overshoots the minimum
     Loss may explode (diverge)


WAY Too LARGE (α = 10):
Loss │
     │                    ╱
     │                   ╱
     │                  ╱
     │                 ╱  Explodes!
     │╲               ╱
     │ ╲_____________╱
     └──────────────────────────→ Iterations

     Problem: Gradients explode, NaN values
```

### Batch Gradient Descent vs Stochastic Gradient Descent

```
Types of Gradient Descent:
────────────────────────────────────────────────────────────────

1. BATCH Gradient Descent
   Use ALL training data for each update

   gradient = (1/N) Σᵢ ∇L(xᵢ, yᵢ)
   W = W - α × gradient

   ✓ Accurate gradient estimate
   ✗ Slow (must process entire dataset)
   ✗ Requires all data in memory


2. STOCHASTIC Gradient Descent (SGD)
   Use ONE example for each update

   For each example (xᵢ, yᵢ):
       gradient = ∇L(xᵢ, yᵢ)
       W = W - α × gradient

   ✓ Fast updates
   ✓ Can escape local minima (noisy)
   ✗ Very noisy gradient estimates
   ✗ Doesn't utilize GPU parallelism


3. MINI-BATCH SGD (Standard Practice)
   Use a small batch (32-256 examples)

   For each batch B:
       gradient = (1/|B|) Σᵢ∈B ∇L(xᵢ, yᵢ)
       W = W - α × gradient

   ✓ Balanced: reasonably accurate + efficient
   ✓ Utilizes GPU parallelism
   ✓ Some noise helps escape local minima

   This is what everyone uses!
```

### Training Terminology

```
Terminology:
────────────────────────────────────────────────────────────────

BATCH SIZE: Number of examples per gradient update
    Typical values: 32, 64, 128, 256, 512

ITERATION (Step): One gradient update
    Process one batch → compute gradient → update weights

EPOCH: One complete pass through training data
    If N=50,000 examples and batch_size=100:
    One epoch = 500 iterations

Example:
────────────────────────────────────────────────────────────────

Dataset: 50,000 images
Batch size: 128
Training: 10 epochs

Iterations per epoch: 50,000 / 128 ≈ 391
Total iterations: 391 × 10 = 3,910 weight updates
```

---

## 2.4 Momentum and Advanced Optimizers

### The Problem with Vanilla SGD

```
SGD Oscillation Problem:
────────────────────────────────────────────────────────────────

Imagine a loss surface that's steep in one direction, shallow in another:

    w₂
     │
     │    ╲        The optimal point is far in w₁
     │     ╲       but close in w₂
     │      ╲
     │       ●→   SGD takes tiny steps in w₁
     │      ╱↓    (shallow gradient)
     │     ╱ ↓
     │    ╱  ↓    but oscillates in w₂
     │   ╱   ↓    (steep gradient)
     │  ╱    ↓
     │ ●→→→→●→→→→→→→ Optimal
     └─────────────────→ w₁

Result: Slow progress, zig-zag path
```

### Momentum

Idea: Accumulate velocity in consistent directions.

```
SGD with Momentum:
────────────────────────────────────────────────────────────────

Standard SGD:
    W = W - α × g        (g = gradient)

With Momentum:
    v = β × v + g        (accumulate gradient with "friction" β)
    W = W - α × v        (update using velocity)

Typical β = 0.9 (keep 90% of previous velocity)


Intuition: Ball Rolling Down Hill
────────────────────────────────────────────────────────────────

     │
     │  ●  Ball starts
     │   ╲
     │    ╲ Picks up speed
     │     ╲
     │      ●→ Has momentum
     │       ╲
     │        ╲
     │         ●══════→ Rolls past small bumps
     │               ↗
     │       ___●___╱  Can climb out of shallow minima
     │
     └────────────────────→

Benefits:
✓ Smooths out oscillations (perpendicular components cancel)
✓ Accelerates in consistent directions
✓ Can escape shallow local minima
```

### Nesterov Momentum

Look ahead before computing gradient:

```python
# Standard momentum: compute gradient at current position
v = β * v + gradient(W)
W = W - α * v

# Nesterov: compute gradient at "look-ahead" position
v = β * v + gradient(W - α * β * v)  # Look ahead
W = W - α * v
```

Small improvement over standard momentum.

### Adam (Adaptive Moment Estimation)

Combines momentum with per-parameter adaptive learning rates:

```
Adam Algorithm:
────────────────────────────────────────────────────────────────

Maintain two moving averages:
  m = momentum (1st moment, mean of gradients)
  v = velocity (2nd moment, variance of gradients)

For each iteration t:
    g = gradient

    # Update biased moments
    m = β₁ × m + (1 - β₁) × g        # Momentum
    v = β₂ × v + (1 - β₂) × g²       # Adaptive scaling

    # Bias correction (important early in training)
    m̂ = m / (1 - β₁ᵗ)
    v̂ = v / (1 - β₂ᵗ)

    # Update weights
    W = W - α × m̂ / (√v̂ + ε)


Intuition:
────────────────────────────────────────────────────────────────

  m (momentum): Smooth gradient direction
  v (variance): Track how much each parameter "fluctuates"

  m̂ / √v̂: Scale update by signal-to-noise ratio

  High variance → uncertain direction → smaller step
  Low variance → consistent direction → normal step


Typical hyperparameters:
  α = 0.001 (learning rate)
  β₁ = 0.9  (momentum decay)
  β₂ = 0.999 (variance decay)
  ε = 1e-8  (numerical stability)
```

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.t += 1

        for key in params:
            # Update biased moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### AdamW (Adam with Decoupled Weight Decay)

Adam + proper L2 regularization:

```python
# Standard Adam with L2 (wrong way):
g = gradient + λ * W  # L2 added to gradient
# Problem: λ interacts with adaptive learning rate

# AdamW (correct way):
g = gradient
# ... Adam update ...
W = W - α * λ * W  # Weight decay applied separately
```

**AdamW is the default choice for most modern models**, especially Transformers.

### Optimizer Comparison

```
Optimizer Recommendations:
────────────────────────────────────────────────────────────────

┌──────────────────┬────────────────────────────────────────────┐
│ Optimizer        │ When to Use                                │
├──────────────────┼────────────────────────────────────────────┤
│ SGD + Momentum   │ CNNs (ImageNet), when you can tune well   │
│ Adam             │ Default starting point, fast convergence  │
│ AdamW            │ Transformers, LLMs, modern best practice  │
│ Adafactor        │ Very large models (memory efficient)      │
└──────────────────┴────────────────────────────────────────────┘

If unsure: Start with AdamW, lr=1e-4 to 3e-4
```

---

## 2.5 Backpropagation

### The Chain Rule

Backpropagation computes gradients efficiently using the chain rule:

```
Chain Rule Review:
────────────────────────────────────────────────────────────────

For composed functions y = f(g(x)):

    dy/dx = dy/dg × dg/dx

For deep composition y = f(g(h(x))):

    dy/dx = dy/df × df/dg × dg/dh × dh/dx

Each term is a "local gradient" that we multiply together.
```

### Computational Graphs

Any neural network can be represented as a computational graph:

```
Computational Graph for y = σ(Wx + b):
────────────────────────────────────────────────────────────────

Forward Pass (left to right):

   x ──→ [×] ──→ [+] ──→ [σ] ──→ y
          ↑       ↑
          W       b

   x=2, W=3: Wx = 6
   b=1: Wx+b = 7
   σ(7) ≈ 0.999


Backward Pass (right to left):
────────────────────────────────────────────────────────────────

Start with ∂L/∂y = 1 (or gradient from downstream)

   ∂L/∂x ←── [×] ←── [+] ←── [σ] ←── ∂L/∂y
              ↓       ↓
           ∂L/∂W   ∂L/∂b

At each node:
  - Receive gradient from downstream
  - Compute local gradient
  - Multiply and pass upstream
```

### Local Gradients for Common Operations

```
Local Gradients:
────────────────────────────────────────────────────────────────

Addition: z = x + y
────────────────────
    ∂z/∂x = 1
    ∂z/∂y = 1

    Gradient "passes through" unchanged.


Multiplication: z = x × y
────────────────────
    ∂z/∂x = y  (swap!)
    ∂z/∂y = x  (swap!)

    Gradient is scaled by the OTHER input.


Matrix Multiply: z = Wx (W is [M×N], x is [N×1])
────────────────────
    ∂L/∂W = (∂L/∂z) × xᵀ    (outer product)
    ∂L/∂x = Wᵀ × (∂L/∂z)    (multiply by transposed weights)


ReLU: z = max(0, x)
────────────────────
    ∂z/∂x = 1 if x > 0
    ∂z/∂x = 0 if x ≤ 0

    Gradient flows through if active, blocked if not.


Sigmoid: z = σ(x) = 1/(1+e^(-x))
────────────────────
    ∂z/∂x = σ(x)(1 - σ(x)) = z(1-z)

    Can compute from output alone!
```

### Backprop Through a Layer

```
Full Example: Linear Layer + ReLU
────────────────────────────────────────────────────────────────

Forward:
    z = Wx + b        (linear)
    h = ReLU(z)       (activation)

Given: ∂L/∂h (gradient flowing back)

Backward through ReLU:
    ∂L/∂z = ∂L/∂h × (z > 0)    (element-wise)
            └──────┘
            This is the ReLU derivative

Backward through Linear:
    ∂L/∂W = (∂L/∂z) × xᵀ       (gradient for weights)
    ∂L/∂b = sum(∂L/∂z, axis=0) (gradient for bias)
    ∂L/∂x = Wᵀ × (∂L/∂z)       (gradient to pass back)
```

### Full Backprop Implementation

```python
def backward_pass(x, y, cache, W1, b1, W2, b2):
    """
    Backward pass for 2-layer network.

    cache = (z1, h1, z2, probs) from forward pass
    """
    z1, h1, z2, probs = cache
    N = x.shape[0]

    # ═══════════════════════════════════════════════════════
    # Gradient of softmax cross-entropy loss
    # ═══════════════════════════════════════════════════════
    # This is a special case: combined softmax + cross-entropy
    # has a simple gradient: probs - one_hot(y)

    dz2 = probs.copy()
    dz2[range(N), y] -= 1   # Subtract 1 from correct class
    dz2 /= N                 # Average over batch

    # ═══════════════════════════════════════════════════════
    # Backprop through Layer 2: z2 = W2 @ h1 + b2
    # ═══════════════════════════════════════════════════════
    dW2 = h1.T @ dz2         # [H1 × K]
    db2 = np.sum(dz2, axis=0) # [K]
    dh1 = dz2 @ W2.T         # [N × H1]

    # ═══════════════════════════════════════════════════════
    # Backprop through ReLU: h1 = ReLU(z1)
    # ═══════════════════════════════════════════════════════
    dz1 = dh1 * (z1 > 0)     # Zero out where ReLU was inactive

    # ═══════════════════════════════════════════════════════
    # Backprop through Layer 1: z1 = W1 @ x + b1
    # ═══════════════════════════════════════════════════════
    dW1 = x.T @ dz1          # [D × H1]
    db1 = np.sum(dz1, axis=0) # [H1]

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

### Why Backprop is Efficient

```
Efficiency of Backpropagation:
────────────────────────────────────────────────────────────────

Naive approach: Compute each ∂L/∂wᵢ independently
  → For N parameters, need N forward passes
  → Cost: O(N × forward_pass)
  → For millions of parameters: impossibly slow

Backpropagation: Reuse intermediate computations
  → One forward pass + one backward pass
  → Cost: O(2 × forward_pass)
  → Same cost regardless of number of parameters!


Example: Network with 10 million parameters
────────────────────────────────────────────────────────────────

Naive:  10,000,000 × 1ms = 10,000 seconds = 2.8 hours per batch
Backprop: 2 × 1ms = 2ms per batch

Backprop is ~5,000,000× faster!
```

---

## 2.6 Weight Initialization

### Why Initialization Matters

```
Bad Initialization Problems:
────────────────────────────────────────────────────────────────

All Zeros:
    W = 0 for all weights

    Problem: All neurons compute the same thing!
    → All gradients identical
    → All neurons update identically
    → Network learns nothing useful

    "Symmetry breaking" is essential.


All Same Value:
    W = 0.01 for all weights

    Same problem as zeros - no symmetry breaking.


Too Large:
    W = np.random.randn(...) * 1.0

    With many inputs: z = Σ wᵢxᵢ becomes very large
    → Activations saturate (sigmoid/tanh)
    → Gradients vanish

Too Small:
    W = np.random.randn(...) * 0.0001

    Activations become tiny
    → Signal diminishes through layers
    → "Dying" problem
```

### Xavier (Glorot) Initialization

For tanh/sigmoid activations:

```
Xavier Initialization:
────────────────────────────────────────────────────────────────

Principle: Keep variance of activations stable through layers

Var(output) = Var(input)

For layer with nᵢₙ inputs:
    W ~ N(0, 1/nᵢₙ)  or  W ~ N(0, 2/(nᵢₙ + nₒᵤₜ))

Implementation:
    W = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)

    Or (Xavier uniform):
    limit = np.sqrt(6.0 / (n_in + n_out))
    W = np.random.uniform(-limit, limit, (n_out, n_in))
```

### He (Kaiming) Initialization

For ReLU activations:

```
He Initialization:
────────────────────────────────────────────────────────────────

ReLU zeros out half the outputs on average.
Need 2× variance to compensate:

    W ~ N(0, 2/nᵢₙ)

Implementation:
    W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)


Derivation Intuition:
────────────────────────────────────────────────────────────────

For ReLU: half of outputs are 0, half pass through.
Var(ReLU(z)) ≈ 0.5 × Var(z)

To maintain Var(output) = Var(input):
    Need Var(z) = 2 × Var(input)
    → Initialize with 2× variance
```

### Initialization in PyTorch

```python
import torch.nn as nn

# Automatically uses appropriate initialization
layer = nn.Linear(256, 128)

# Manual initialization
nn.init.xavier_uniform_(layer.weight)  # For tanh/sigmoid
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU
nn.init.zeros_(layer.bias)

# For entire network
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
```

---

## 2.7 Regularization

### The Overfitting Problem Revisited

```
Overfitting Visualization:
────────────────────────────────────────────────────────────────

Training Data:        Test Data:
    ●   ○              ●   ○
      ●   ○              ●   ○
    ●   ○   ●          ●   ○   ●
      ○   ●              ○   ●


Overfit Model:        Regularized Model:
    ●╱─●○              ●   ○
     ╲╱  ╲○        ────────────────
   ●╱●  ○╱●          ●   ○   ●
    ╲○──●              ○   ●

Memorizes training    Learns general pattern
points exactly        that generalizes


Training: 100% accuracy     Training: 95% accuracy
Test: 70% accuracy          Test: 92% accuracy
```

### L2 Regularization (Weight Decay)

```
L2 Regularization:
────────────────────────────────────────────────────────────────

Add penalty for large weights to loss:

    L_total = L_data + (λ/2) × Σᵢ wᵢ²
                       └───────────────┘
                       Regularization term

Gradient includes weight penalty:
    ∂L_total/∂w = ∂L_data/∂w + λw

Update rule:
    w = w - α(∂L_data/∂w + λw)
    w = (1 - αλ)w - α×∂L_data/∂w
        └────────┘
        "Weight decay" - shrinks weights toward 0


Effect:
────────────────────────────────────────────────────────────────

Without L2:              With L2:
Weights can grow         Weights stay small
arbitrarily large

Large weights =          Small weights =
Model fits noise         Smoother function
(overfitting)            (better generalization)


Typical λ values: 1e-4 to 1e-2
```

### L1 Regularization

```
L1 Regularization:
────────────────────────────────────────────────────────────────

L_total = L_data + λ × Σᵢ |wᵢ|

Gradient:
    ∂L_total/∂w = ∂L_data/∂w + λ × sign(w)

Effect:
    Drives many weights to EXACTLY zero
    → Sparse models
    → Feature selection


L1 vs L2:
────────────────────────────────────────────────────────────────

          L1                    L2
          ──                    ──
Penalty:  |w|                   w²
Gradient: sign(w) (constant)    2w (proportional)
Result:   Sparse (many 0s)      Small but non-zero
Use for:  Feature selection     General regularization
```

### Dropout

```
Dropout Mechanism:
────────────────────────────────────────────────────────────────

During TRAINING:
  Randomly set neurons to 0 with probability (1-p)
  Scale remaining by 1/p to maintain expected value

      Input layer      Hidden layer      Output
                       (with dropout)

        ●─────────────────●─────────────────●
        │                 │                 │
        ●────────────────[×]────────────────●
        │                 0 (dropped)       │
        ●─────────────────●─────────────────●
        │               scale by 1/p        │
        ●────────────────[×]────────────────●
                          0 (dropped)


During TESTING:
  Use all neurons (no dropout)
  No scaling needed if "inverted dropout" used


Implementation (Inverted Dropout):
────────────────────────────────────────────────────────────────

def dropout(x, p=0.5, training=True):
    if not training:
        return x

    # Create random mask
    mask = (np.random.rand(*x.shape) < p) / p
    return x * mask

# Training: randomly drops and scales
# Testing: returns input unchanged
```

```python
# PyTorch
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        x = self.fc2(x)
        return x
```

### Batch Normalization

```
Batch Normalization:
────────────────────────────────────────────────────────────────

Normalize activations within each mini-batch:

For a batch of activations x = [x₁, x₂, ..., xₙ]:

1. Compute batch statistics:
       μ = (1/N) Σᵢ xᵢ           (batch mean)
       σ² = (1/N) Σᵢ (xᵢ - μ)²   (batch variance)

2. Normalize:
       x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

3. Scale and shift (learnable):
       yᵢ = γ × x̂ᵢ + β

       γ, β are learnable parameters


Visualization:
────────────────────────────────────────────────────────────────

Before BatchNorm:          After BatchNorm:

   │  ●                        │     ●
   │       ●                   │   ●   ●
   │ ●                         │ ●       ●
   │         ●                 │   ●   ●
   │   ●                       │     ●
   └──────────              ───┼───────────
                               │   Centered at 0
   Activations vary            Standardized
   wildly                      distribution


Benefits:
────────────────────────────────────────────────────────────────

✓ Allows higher learning rates (training is more stable)
✓ Less sensitive to initialization
✓ Acts as regularization (batch statistics add noise)
✓ Reduces "internal covariate shift"

Placement:
    Conv → BatchNorm → ReLU    (most common)
    or
    Conv → ReLU → BatchNorm
```

### Layer Normalization

```
Layer Normalization (for Transformers):
────────────────────────────────────────────────────────────────

Normalize across features (not batch):

For each example x of shape [D]:
    μ = (1/D) Σⱼ xⱼ
    σ² = (1/D) Σⱼ (xⱼ - μ)²
    x̂ = (x - μ) / √(σ² + ε)
    y = γ × x̂ + β


BatchNorm vs LayerNorm:
────────────────────────────────────────────────────────────────

BatchNorm: normalize across batch for each feature
    Shape: [N, D] → compute mean/var for each column

LayerNorm: normalize across features for each example
    Shape: [N, D] → compute mean/var for each row


When to use:
────────────────────────────────────────────────────────────────

BatchNorm: CNNs, fixed-size inputs
LayerNorm: Transformers, RNNs, variable-length sequences
    (LayerNorm doesn't depend on batch statistics)
```

---

## 2.8 Learning Rate Schedules

```
Learning Rate Schedules:
────────────────────────────────────────────────────────────────

Start high for fast progress, decrease for fine-tuning.


1. STEP DECAY
────────────────────────────────────────────────────────────────

    lr
    │
0.1 │────────┐
    │        │
0.01│        └───────┐
    │                │
0.001               └─────────
    └─────────────────────────→ epochs
         30       60      90

    Every N epochs, multiply LR by factor (e.g., 0.1)


2. EXPONENTIAL DECAY
────────────────────────────────────────────────────────────────

    lr = lr_0 × decay_rate^(step / decay_steps)

    lr
    │
    │╲
    │ ╲
    │  ╲
    │   ╲___
    │       ‾‾‾───___
    └─────────────────→ steps


3. COSINE ANNEALING
────────────────────────────────────────────────────────────────

    lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))

    lr
    │
    │‾‾‾╲
    │    ╲
    │     ╲
    │      ╲
    │       ╲___
    │           ‾‾───
    └─────────────────→ steps

    Smooth decay following cosine curve.
    Popular for training from scratch.


4. WARMUP + DECAY (for Transformers)
────────────────────────────────────────────────────────────────

    lr
    │
    │      ╱‾‾‾‾╲
    │     ╱      ╲
    │    ╱        ╲
    │   ╱          ╲___
    │  ╱               ‾‾───
    │ ╱
    │╱
    └─────────────────────────→ steps
     warmup  peak   decay

    Start from 0, linearly increase, then decay.
    Critical for Transformer training stability.
```

```python
# PyTorch learning rate schedulers
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    OneCycleLR, LinearLR, SequentialLR
)

# Step decay: multiply by 0.1 every 30 epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Warmup + cosine (common for Transformers)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=1000)
cosine = CosineAnnealingLR(optimizer, T_max=9000)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[1000])

# Training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()  # Update learning rate
```

---

## 2.9 Gradient Problems and Solutions

### Vanishing Gradients

```
Vanishing Gradient Problem:
────────────────────────────────────────────────────────────────

In deep networks, gradients can become exponentially small:

    ∂L/∂W₁ = ∂L/∂Wₙ × ∂Wₙ/∂Wₙ₋₁ × ... × ∂W₂/∂W₁
             └──────────────────────────────────────┘
              Product of many numbers < 1

If each term ≈ 0.5, after 10 layers:
    0.5^10 = 0.001 (gradient is 1000× smaller!)


Symptoms:
────────────────────────────────────────────────────────────────

    │ Loss
    │
    │ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  Loss plateaus quickly
    │
    │
    └──────────────────────→ Steps

    Early layers barely change
    Only late layers learn


Solutions:
────────────────────────────────────────────────────────────────

1. Use ReLU (gradient = 1 for positive inputs)
2. Residual connections (skip connections)
3. Proper initialization (He/Xavier)
4. Batch/Layer normalization
5. LSTM/GRU for sequences (gating mechanisms)
```

### Exploding Gradients

```
Exploding Gradient Problem:
────────────────────────────────────────────────────────────────

Opposite problem: gradients become exponentially large

If each term ≈ 2, after 10 layers:
    2^10 = 1024 (gradient is 1000× larger!)


Symptoms:
────────────────────────────────────────────────────────────────

    │ Loss
    │                  ↗
    │               ↗
    │            ↗     Suddenly explodes
    │         ↗
    │ ───────╱
    │
    └──────────────────────→ Steps

    Loss becomes NaN
    Weights become NaN


Solution: GRADIENT CLIPPING
────────────────────────────────────────────────────────────────

If gradient norm exceeds threshold, scale it down:

    if ||g|| > max_norm:
        g = g × (max_norm / ||g||)

This caps the maximum gradient magnitude while preserving direction.
```

```python
# Gradient clipping in PyTorch
optimizer.zero_grad()
loss.backward()

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

---

## 2.10 Practical Training Recipe

### Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ═══════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ═══════════════════════════════════════════════════════════════
# 2. MODEL
# ═══════════════════════════════════════════════════════════════
model = MyNetwork()
model.apply(init_weights)  # He initialization

# ═══════════════════════════════════════════════════════════════
# 3. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()

# ═══════════════════════════════════════════════════════════════
# 4. OPTIMIZER
# ═══════════════════════════════════════════════════════════════
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# ═══════════════════════════════════════════════════════════════
# 5. LEARNING RATE SCHEDULER
# ═══════════════════════════════════════════════════════════════
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ═══════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    # Learning rate update
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    print(f'Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, '
          f'val_loss={val_loss:.4f}, val_acc={accuracy:.4f}')
```

### Debugging Checklist

```
Training Debugging Checklist:
────────────────────────────────────────────────────────────────

□ 1. SANITY CHECKS
   □ Can model overfit ONE batch? (should reach ~0 loss)
   □ Initial loss ≈ -log(1/num_classes) for classification?
   □ Data pipeline correct? (visualize samples)
   □ Labels correct? (check a few manually)

□ 2. LOSS NOT DECREASING
   □ Learning rate too high? (try 10× smaller)
   □ Learning rate too low? (try 10× larger)
   □ Gradients vanishing? (check gradient norms)
   □ Gradients exploding? (add gradient clipping)
   □ Weight initialization correct?

□ 3. LOSS IS NaN
   □ Learning rate too high
   □ Division by zero somewhere
   □ Log of zero (add epsilon)
   □ Numerical overflow (use log-sum-exp trick)

□ 4. OVERFITTING
   □ Add dropout
   □ Increase weight decay
   □ Add data augmentation
   □ Try smaller model
   □ Get more training data

□ 5. UNDERFITTING
   □ Try larger model
   □ Reduce regularization
   □ Train longer
   □ Check for bugs in model architecture
```

---

## 2.11 Summary

### Key Concepts

```
Module 2 Summary:
────────────────────────────────────────────────────────────────

1. LOSS FUNCTIONS quantify prediction error
   • Cross-entropy for classification
   • MSE for regression

2. GRADIENT DESCENT minimizes loss iteratively
   • W = W - α × ∇L
   • Learning rate α is critical

3. SGD variants improve convergence
   • Momentum: smooth oscillations
   • Adam: adaptive learning rates per parameter
   • AdamW: best for Transformers

4. BACKPROPAGATION computes gradients efficiently
   • Chain rule through computational graph
   • One forward + one backward pass

5. INITIALIZATION prevents symmetry and vanishing/exploding
   • Xavier for sigmoid/tanh
   • He (Kaiming) for ReLU

6. REGULARIZATION prevents overfitting
   • L2 (weight decay): small weights
   • Dropout: randomly drop neurons
   • BatchNorm/LayerNorm: stabilize activations

7. LEARNING RATE SCHEDULES improve training
   • Warmup + decay for Transformers
   • Cosine annealing is popular
```

### Glossary Terms Covered

| Term | Definition |
|------|------------|
| **Loss Function** | Measures how wrong predictions are |
| **Gradient** | Direction of steepest increase in loss |
| **Gradient Descent** | Iterative optimization by moving against gradient |
| **SGD** | Stochastic Gradient Descent using mini-batches |
| **Learning Rate** | Step size in gradient descent |
| **Momentum** | Accumulate gradient history for smoother updates |
| **Adam** | Adaptive learning rates + momentum |
| **Backpropagation** | Efficient gradient computation via chain rule |
| **Epoch** | One pass through entire training dataset |
| **Batch Size** | Number of examples per gradient update |
| **Overfitting** | Model memorizes training data, poor generalization |
| **Regularization** | Techniques to prevent overfitting |
| **Weight Decay** | L2 penalty on weight magnitudes |
| **Dropout** | Randomly zero neurons during training |
| **Batch Normalization** | Normalize activations within mini-batch |
| **Vanishing Gradient** | Gradients become too small in deep networks |
| **Gradient Clipping** | Cap gradient magnitude to prevent explosion |

---

## References and Further Reading

### Lectures
- [CS231n Lecture 3: Loss Functions and Optimization](http://cs231n.stanford.edu/slides/2024/lecture_3.pdf)
- [CS231n Lecture 7: Training Neural Networks](http://cs231n.stanford.edu/slides/2024/lecture_7.pdf)
- [CS224N Lecture 3: Backprop and Neural Networks](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture03-neuralnets.pdf)

### Course Notes
- [CS231n: Optimization](https://cs231n.github.io/optimization-1/)
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/)
- [CS231n: Neural Networks Part 2 (Training)](https://cs231n.github.io/neural-networks-2/)
- [CS231n: Neural Networks Part 3 (Learning)](https://cs231n.github.io/neural-networks-3/)

### Papers
- [Kingma & Ba (2014): Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Loshchilov & Hutter (2017): Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [Ioffe & Szegedy (2015): Batch Normalization](https://arxiv.org/abs/1502.03167)
- [Ba et al. (2016): Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Srivastava et al. (2014): Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
- [He et al. (2015): Delving Deep into Rectifiers (He initialization)](https://arxiv.org/abs/1502.01852)
- [Glorot & Bengio (2010): Understanding difficulty of training (Xavier init)](http://proceedings.mlr.press/v9/glorot10a.html)

### Books
- [Goodfellow et al.: Deep Learning, Chapters 6-8](https://www.deeplearningbook.org/)
- [Zhang et al.: Dive into Deep Learning, Chapter 4](https://d2l.ai/chapter_linear-classification/index.html)

### Tools and Tutorials
- [PyTorch Optimization Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Weights & Biases: Loss Landscape Visualization](https://wandb.ai/site/articles/loss-landscape)
- [Distill: Why Momentum Really Works](https://distill.pub/2017/momentum/)
