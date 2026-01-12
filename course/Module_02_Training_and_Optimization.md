# Module 2: Training & Optimization

## Learning Objectives

By the end of this module, you will understand:
- Loss functions and how they measure model quality
- Gradient descent and its variants
- Backpropagation: computing gradients efficiently
- Modern optimizers (SGD, Adam, AdamW)
- Regularization techniques to prevent overfitting
- Practical training considerations

---

## 2.1 The Training Problem

### What is Training?

**Training** is finding the parameters (weights and biases) that make our network perform well. We quantify "well" using a **loss function**.

```
Training Data: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
                     ↓
         Network with parameters θ
                     ↓
              Predictions ŷᵢ
                     ↓
         Loss = L(ŷᵢ, yᵢ) measures error
                     ↓
    Goal: Find θ that minimizes total loss
```

### The Three Components

1. **Score function**: Maps input to predictions → `f(x; W) = Wx`
2. **Loss function**: Measures prediction quality → `L(f(x), y)`
3. **Optimization**: Finds parameters minimizing loss

---

## 2.2 Loss Functions

### Cross-Entropy Loss (Softmax Loss)

For classification with C classes:

```
L = -log(p_y) = -log(e^{s_y} / Σⱼ e^{s_j})
```

Where:
- `s` = raw scores from network (logits)
- `s_y` = score for the correct class
- `p_y` = probability assigned to correct class

**Intuition**: Penalizes low confidence in correct class. If network assigns 90% probability to correct class, loss is low. If only 10%, loss is high.

```python
def cross_entropy_loss(scores, y):
    """
    scores: [N, C] raw class scores
    y: [N] correct class indices
    """
    # Softmax probabilities
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Negative log probability of correct class
    N = scores.shape[0]
    loss = -np.sum(np.log(probs[range(N), y])) / N
    return loss
```

### Mean Squared Error (MSE)

For regression tasks:

```
L = (1/N) Σᵢ (ŷᵢ - yᵢ)²
```

**When to use**:
- Cross-entropy: Classification (prefer this)
- MSE: Regression (predicting continuous values)

### Total Loss

The total loss typically includes:

```
L_total = L_data + λ · L_regularization
```

Where `λ` controls regularization strength.

---

## 2.3 Gradient Descent

### The Optimization Landscape

Imagine the loss as a surface over parameter space. We want to find the lowest point (minimum).

```
Loss
  │
  │     ╱╲
  │    ╱  ╲    ╱╲
  │   ╱    ╲__╱  ╲___
  │  ╱                ╲___○ ← We want to reach here
  │_╱
  └──────────────────────────→ Parameters
```

### The Gradient

The **gradient** tells us:
1. **Direction** of steepest increase in loss
2. **Magnitude** of slope at current point

Moving **opposite** the gradient decreases loss.

```
∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]
```

### Vanilla Gradient Descent

```python
while not converged:
    gradient = compute_gradient(loss_function, data, weights)
    weights = weights - learning_rate * gradient
```

**Key hyperparameter: Learning Rate (α)**

```
Too small:           Just right:          Too large:
    ○                    ○                    ○
     ↓                    ↓                    ↓
      ↓                    ↓                   ↑
       ↓                    ↓                 ↑
        ↓                   ◎               ↑
         ↓               (converges)      (diverges!)
         ... (very slow)
```

### Stochastic Gradient Descent (SGD)

Computing gradient over ALL training data is expensive. Instead, use **mini-batches**:

```python
for epoch in range(num_epochs):
    for batch in data_loader:
        x_batch, y_batch = batch
        gradient = compute_gradient(loss, x_batch, y_batch, weights)
        weights = weights - learning_rate * gradient
```

**Terminology**:
- **Batch**: Subset of training data (e.g., 32, 64, 128 examples)
- **Iteration/Step**: One weight update
- **Epoch**: One pass through entire training dataset

**Example**: 50,000 training images, batch size 100
- Iterations per epoch: 500
- 10 epochs = 5,000 iterations

### SGD with Momentum

Plain SGD can oscillate. Momentum smooths updates:

```python
velocity = 0
for iteration in range(num_iterations):
    gradient = compute_gradient(...)
    velocity = momentum * velocity - learning_rate * gradient
    weights = weights + velocity
```

```
Without momentum:      With momentum:
      ↓                     ↓
     ↙                     ↓
    →                       ↓
   ↓                        ↓
  ↙                         ◎
 → (oscillates)        (smoother path)
```

Typical momentum value: 0.9

---

## 2.4 Backpropagation

### The Chain Rule

Backpropagation computes gradients efficiently using the **chain rule** of calculus.

For composed functions: `L = f(g(h(x)))`

```
∂L/∂x = (∂L/∂f) · (∂f/∂g) · (∂g/∂h) · (∂h/∂x)
```

### Computational Graph View

Every network is a computational graph:

```
x ──→ [×W] ──→ [+b] ──→ [ReLU] ──→ [×W] ──→ [Softmax] ──→ [Loss]
        ↑        ↑                    ↑
        W₁       b₁                   W₂

Forward: Left → Right (compute outputs)
Backward: Right → Left (compute gradients)
```

### Local Gradients

Each operation has a **local gradient**:

| Operation | Forward | Local Gradient |
|-----------|---------|----------------|
| Add: `z = x + y` | z | ∂z/∂x = 1, ∂z/∂y = 1 |
| Multiply: `z = x × y` | z | ∂z/∂x = y, ∂z/∂y = x |
| ReLU: `z = max(0, x)` | z | ∂z/∂x = 1 if x > 0 else 0 |
| Sigmoid: `z = σ(x)` | z | ∂z/∂x = z(1-z) |

### Backprop Algorithm

```python
def backward_pass(x, y, W1, b1, W2, b2, cache):
    """Compute gradients for 2-layer network"""
    h1, z1, scores = cache  # From forward pass
    N = x.shape[0]

    # Gradient of loss w.r.t. scores (softmax + cross-entropy)
    probs = softmax(scores)
    dscores = probs.copy()
    dscores[range(N), y] -= 1
    dscores /= N

    # Backprop to W2, b2
    dW2 = h1.T.dot(dscores)
    db2 = np.sum(dscores, axis=0)

    # Backprop to hidden layer
    dh1 = dscores.dot(W2.T)

    # Backprop through ReLU
    dz1 = dh1 * (z1 > 0)  # ReLU gradient

    # Backprop to W1, b1
    dW1 = x.T.dot(dz1)
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2
```

### Why Backprop is Efficient

**Naive approach**: Compute each gradient independently → O(n²) for n parameters

**Backprop**: Reuse intermediate computations → O(n)

For a network with millions of parameters, backprop is essential.

---

## 2.5 Modern Optimizers

### Problems with Vanilla SGD

1. **Same learning rate for all parameters** (some need larger updates)
2. **Oscillation** in directions with high curvature
3. **Slow progress** in flat regions

### Adam (Adaptive Moment Estimation)

Combines momentum with adaptive learning rates:

```python
m = 0  # First moment (mean of gradients)
v = 0  # Second moment (variance of gradients)
t = 0

for iteration in range(num_iterations):
    t += 1
    g = compute_gradient(...)

    # Update moments
    m = β₁ * m + (1 - β₁) * g        # Momentum-like
    v = β₂ * v + (1 - β₂) * g²       # Adaptive scaling

    # Bias correction (important early in training)
    m_hat = m / (1 - β₁^t)
    v_hat = v / (1 - β₂^t)

    # Update weights
    weights = weights - α * m_hat / (√v_hat + ε)
```

**Typical hyperparameters**:
- α = 0.001 (learning rate)
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

### AdamW (Adam with Weight Decay)

Adam with proper L2 regularization (decoupled weight decay):

```python
weights = weights - α * m_hat / (√v_hat + ε) - α * λ * weights
```

**Recommended** for most deep learning tasks.

### Optimizer Comparison

| Optimizer | Strengths | Best For |
|-----------|-----------|----------|
| SGD + Momentum | Simple, good generalization | CNNs, when tuned well |
| Adam | Fast convergence, less tuning | Default starting point |
| AdamW | Better generalization than Adam | Transformers, modern models |

### Learning Rate Schedules

Learning rate typically decreases during training:

**Step decay**: Reduce by factor every N epochs
```python
if epoch % 30 == 0:
    lr = lr * 0.1
```

**Cosine annealing**: Smooth decay following cosine curve
```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

**Warmup**: Start low, increase, then decay (common for Transformers)
```python
if step < warmup_steps:
    lr = lr_max * step / warmup_steps
else:
    lr = cosine_decay(step)
```

---

## 2.6 Regularization

### The Overfitting Problem

```
                Training      Validation
                  Loss          Loss
                   │             │
Underfitting      █████         █████
                   │             │
Good fit          ███           ████
                   │             │
Overfitting       █            ████████
                   │             │
                   ↓             ↓
             (very low)      (high!)
```

### L2 Regularization (Weight Decay)

Add squared weight magnitudes to loss:

```
L_total = L_data + (λ/2) Σᵢ wᵢ²
```

**Effect**: Encourages small weights, prevents any single weight from dominating.

**Gradient**: `∂L_reg/∂w = λw`

During update: `w = w - α(∇L_data + λw) = (1 - αλ)w - α∇L_data`

The term `(1 - αλ)` "decays" weights toward zero each step.

### L1 Regularization

```
L_total = L_data + λ Σᵢ |wᵢ|
```

**Effect**: Encourages **sparse** weights (many exactly zero).

### Dropout

Randomly "drop" neurons during training:

```python
def dropout_forward(x, p=0.5, training=True):
    """
    p: probability of KEEPING a neuron
    """
    if training:
        mask = (np.random.rand(*x.shape) < p) / p  # Scale by 1/p
        return x * mask
    else:
        return x  # No dropout at test time
```

```
Training:                    Testing:
  ●───●───●                   ●───●───●
  │ ╲ │ ╱ │                   │ ╲ │ ╱ │
  ●   ✗   ●  (drop 50%)       ●───●───●  (use all)
  │ ╱ │ ╲ │                   │ ╱ │ ╲ │
  ●───●───●                   ●───●───●
```

**Why it works**:
- Prevents co-adaptation of neurons
- Ensemble effect (averages many sub-networks)
- Acts like noise injection

**Typical values**: p = 0.5 for hidden layers, p = 0.8-0.9 for input

### Batch Normalization

Normalize activations within each mini-batch:

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    x: [N, D] activations
    gamma, beta: learnable scale and shift [D]
    """
    # Normalize
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Scale and shift
    out = gamma * x_norm + beta
    return out
```

**Benefits**:
- Stabilizes training (can use higher learning rates)
- Acts as regularization
- Reduces sensitivity to initialization

**Placement**: After linear layer, before activation

```
Linear → BatchNorm → ReLU
```

### Layer Normalization

Normalize across features (not batch):

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """Normalize each example independently"""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**Used in**: Transformers (works with variable sequence lengths)

### Early Stopping

Stop training when validation loss stops improving:

```python
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    train(model)
    val_loss = evaluate(model, val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

### Data Augmentation

Create variations of training data:

**For images**:
- Random crops, flips, rotations
- Color jittering
- Cutout, Mixup, CutMix

**For text**:
- Back-translation
- Synonym replacement
- Random insertion/deletion

---

## 2.7 Gradient Problems

### Vanishing Gradients

In deep networks, gradients can become very small:

```
∂L/∂W₁ = ∂L/∂Wₙ × (∂Wₙ/∂Wₙ₋₁) × ... × (∂W₂/∂W₁)
                   └────────────────────────────────┘
                      Product of many small numbers
                         → vanishes toward 0
```

**Symptoms**: Early layers learn very slowly

**Solutions**:
- ReLU activation (gradient = 1 for positive inputs)
- Residual connections
- Proper initialization
- Batch/Layer normalization

### Exploding Gradients

Opposite problem—gradients become huge:

**Symptoms**: NaN losses, unstable training

**Solutions**:
- Gradient clipping: `if ||g|| > threshold: g = g × (threshold / ||g||)`
- Proper initialization
- Lower learning rate

### Gradient Clipping

```python
def clip_gradients(gradients, max_norm):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        gradients = [g * clip_coef for g in gradients]
    return gradients
```

Common in RNNs and Transformers.

---

## 2.8 Practical Training Recipe

### Initial Setup

```python
# 1. Data
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# 2. Model
model = MyNetwork()
model.apply(weight_init)  # He initialization for ReLU

# 3. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 4. Learning rate schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 5. Loss function
criterion = nn.CrossEntropyLoss()
```

### Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

    # Update learning rate
    scheduler.step()

    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
```

### Debugging Checklist

1. **Overfit small batch first**: Can model memorize 10 examples?
2. **Check loss at initialization**: Should be `log(num_classes)` for softmax
3. **Monitor gradients**: Not too small (vanishing) or large (exploding)
4. **Verify data pipeline**: Visualize augmented samples
5. **Learning rate finder**: Try range of LRs, pick where loss decreases fastest

---

## 2.9 Summary

### Key Concepts

1. **Loss functions** measure how wrong predictions are (cross-entropy, MSE)
2. **Gradient descent** iteratively updates weights to minimize loss
3. **Backpropagation** efficiently computes gradients via chain rule
4. **SGD, Adam, AdamW** are optimization algorithms with different tradeoffs
5. **Regularization** (L2, dropout, batch norm) prevents overfitting
6. **Gradient clipping** handles exploding gradients

### Glossary Terms Covered

- Training
- Loss Function (Cost Function, Objective Function)
- Gradient
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Backpropagation
- Learning Rate
- Optimizer
- Epoch
- Batch (Mini-batch)
- Batch Size
- Iteration
- Convergence
- Overfitting
- Underfitting
- Generalization
- Hyperparameter
- Validation Set
- Test Set
- Vanishing Gradient Problem
- Exploding Gradient Problem
- Gradient Clipping
- L1/L2 Regularization
- Dropout
- Batch Normalization
- Layer Normalization
- Early Stopping
- Data Augmentation

### What's Next

Module 3 introduces **Convolutional Neural Networks** for processing images with spatial structure.

---

## Exercises

1. **Gradient calculation**: For `L = (wx - y)²`, compute `∂L/∂w`.

2. **Learning rate**: Given loss values [10, 8, 9, 11, 15] over 5 iterations, what might be wrong with your learning rate?

3. **Regularization tradeoff**: Training loss = 0.1, validation loss = 2.0. What should you try?

4. **Code**: Implement SGD with momentum from scratch.

---

## References

- CS231n: Optimization, Backpropagation
- CS224N: Lecture 3, 4 - Neural Networks and Backprop
- Kingma & Ba, "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW)
