# Deep Learning Pitfalls: A Survival Guide

## Why Smart People Fail at Deep Learning

Deep learning is deceptively simple in concept but treacherous in practice.
This guide covers the mistakes that trap students and practitioners alike,
with concrete examples and solutions.

```
THE PITFALL LANDSCAPE
=====================

                    SUCCESS
                       △
                      /|\
                     / | \
                    /  |  \
                   /   |   \
         Data     /    |    \    Evaluation
         Traps   /     |     \   Traps
                /      |      \
               /   Training    \
              /     Traps       \
             /         |         \
            /          |          \
           /    Architecture       \
          /        Traps            \
         /             |             \
        ───────────────────────────────
                 FAILURE ZONE

Most practitioners spend 80% of their time in the failure zone
before finding their way up.
```

---

## Part 1: Conceptual Pitfalls

### Pitfall 1.1: "More Layers = Better Model"

**The Misconception:**
"My model isn't working. Let me add more layers!"

**The Reality:**
Deeper networks are harder to train and need specific techniques (residuals,
proper initialization, normalization) to work. More layers without these
safeguards often perform WORSE.

```
DEPTH vs PERFORMANCE (without residuals)
=========================================

Accuracy
   │
   │      ____
   │     /    \
   │    /      \___________
   │   /                    \____
   │  /                          \___
   │ /
   └──────────────────────────────────> Layers
        20      56      110

The infamous "degradation problem" - deeper = worse!
(This is what motivated ResNets)


THE FIX:

1. Start shallow, add depth only when needed
2. Always use residual connections for >10 layers
3. Use batch normalization
4. Verify gradients are flowing (check magnitudes)
```

**Concrete Example:**

```python
# BAD: Just stacking layers
class BadDeepNet(nn.Module):
    def __init__(self):
        self.layers = nn.Sequential(
            *[nn.Linear(256, 256) for _ in range(50)],  # 50 layers!
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)  # Gradients will vanish/explode

# GOOD: Residual connections
class GoodDeepNet(nn.Module):
    def __init__(self):
        self.blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(50)
        ])
        self.final = nn.Linear(256, 10)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)  # Residual connection!
        return self.final(x)
```

---

### Pitfall 1.2: "My Model Has 99% Training Accuracy, So It's Great"

**The Misconception:**
High training accuracy means the model works.

**The Reality:**
A model can memorize the training data completely while learning nothing
generalizable. Training accuracy tells you almost nothing.

```
THE OVERFITTING TRAP
====================

        Training Acc    Validation Acc    Actual Performance
        ────────────    ──────────────    ──────────────────
Model A:    99%              95%          Good! (generalizes)
Model B:    99%              55%          Terrible! (memorized)
Model C:    85%              83%          Okay (might underfit)

The gap between training and validation is what matters!


LOSS CURVES TELL THE STORY:

Loss
  │
  │ \
  │  \___  Training
  │      \_____
  │             \______
  │
  │    \
  │     \___  Validation
  │         \_____
  │              \____/¯¯¯¯¯  <- validation rising = STOP!
  │
  └────────────────────────────> Epochs
                      ↑
               Stop training here
               (early stopping)
```

**The Fix:**
- ALWAYS monitor validation loss, not training loss
- Use early stopping
- Plot both curves during training
- If gap is large: add regularization (dropout, weight decay, augmentation)

---

### Pitfall 1.3: "Cross-Entropy Loss Means My Probabilities Are Calibrated"

**The Misconception:**
If the model outputs 0.9 for a class, it means 90% confident.

**The Reality:**
Neural networks are notoriously overconfident. A model saying 99% confident
might be wrong 20% of the time at that confidence level.

```
CALIBRATION DIAGRAM
===================

Expected accuracy given confidence:

Accuracy
  100%|                          ╱ Perfect calibration
     |                        ╱
  80%|                      ╱
     |            ____-----
  60%|      -----     <- Actual model (overconfident!)
     |   ---
  40%|  -
     | -
  20%|-
     |
   0%└─────────────────────────────
     0%   20%   40%   60%   80%  100%
                Confidence

When model says "90% sure", it's often only 70% accurate.
```

**The Fix:**
- Use temperature scaling for calibration
- Don't trust raw probabilities for critical decisions
- Evaluate with calibration metrics (ECE, reliability diagrams)

```python
# Temperature scaling for calibration
class CalibratedModel(nn.Module):
    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature  # Scale down confidence
```

---

### Pitfall 1.4: "Backprop Is Magic That Just Works"

**The Misconception:**
Just call `loss.backward()` and gradients flow correctly.

**The Reality:**
Gradients can vanish, explode, or get blocked by bad architectural choices.

```
GRADIENT FLOW PROBLEMS
======================

1. VANISHING GRADIENTS

   Layer 1    Layer 2    Layer 3    Layer 4    Loss
      │          │          │          │         │
      ◄──────────◄──────────◄──────────◄─────────◄
   0.0001     0.001      0.01       0.1        1.0

   Each layer multiplies by <1, gradients shrink exponentially!

   Causes:
   - Sigmoid/tanh activations (saturate at extremes)
   - Very deep networks without residuals
   - Improper initialization

2. EXPLODING GRADIENTS

   Layer 1    Layer 2    Layer 3    Layer 4    Loss
      │          │          │          │         │
      ◄──────────◄──────────◄──────────◄─────────◄
    1e10       1e8        1e6        1e4        1.0

   Causes:
   - Large weights
   - Long sequences in RNNs
   - No gradient clipping

3. DEAD NEURONS (ReLU)

      x ──→ [ReLU] ──→ 0  (if x < 0, gradient = 0 forever!)

   Once a ReLU neuron outputs 0 for all inputs, it's "dead"
   - Can't recover because gradient is always 0
   - Use Leaky ReLU or careful initialization
```

**The Fix:**

```python
# Check gradient health during training
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-7:
                print(f"WARNING: {name} has vanishing gradient: {grad_norm}")
            if grad_norm > 1e3:
                print(f"WARNING: {name} has exploding gradient: {grad_norm}")
            if torch.isnan(param.grad).any():
                print(f"CRITICAL: {name} has NaN gradient!")

# Gradient clipping for RNNs
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Part 2: Data Pitfalls

### Pitfall 2.1: Data Leakage (The Silent Killer)

**The Problem:**
Information from the test set leaks into training, giving artificially
high validation scores that don't generalize.

```
DATA LEAKAGE SCENARIOS
======================

Scenario 1: Preprocessing on full dataset

   WRONG:                              RIGHT:

   Full Data                           Train Data    Test Data
      │                                    │            │
      ▼                                    ▼            │
   [Normalize]  <- computes mean/std   [Normalize]     │
      │            from ALL data           │            │
      ▼                                    ▼            ▼
   [Split]                             (fitted)    [Apply same
      │                                             transform]
   Train  Test

   Test set statistics leaked          Only train stats used
   into normalization!


Scenario 2: Time series - future leaking into past

   WRONG:                              RIGHT:

   [Jan][Feb][Mar][Apr][May]           [Jan][Feb][Mar] | [Apr][May]
         ↓      ↓                            ↓               ↓
   Random shuffle, then split          Chronological split only!

   Feb predicting March, but           Train only on past,
   trained on April data!              predict future


Scenario 3: Duplicate/near-duplicate images

   Training set:     Test set:
   [cat_001.jpg]     [cat_001_crop.jpg]  <- Same cat, different crop!

   Model memorizes the specific cat, not "cat-ness"
```

**Concrete Example - The Kaggle Disaster:**

```python
# BAD: Fitting scaler on all data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses ALL data
X_train, X_test = train_test_split(X_scaled)  # Leakage!

# GOOD: Fit only on training data
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Apply same transform
```

---

### Pitfall 2.2: Insufficient Data Augmentation

**The Problem:**
Using augmentation that's too weak, or augmentation that changes the label.

```
AUGMENTATION PITFALLS
=====================

Too weak augmentation:
┌─────────┐     ┌─────────┐
│   🐱    │ →   │   🐱    │   (just tiny rotation)
└─────────┘     └─────────┘
                Almost identical - not helping!

Label-changing augmentation:
┌─────────┐     ┌─────────┐
│    6    │ →   │    9    │   (vertical flip on digits)
└─────────┘     └─────────┘
                Wrong! 6 became 9!

Domain-inappropriate augmentation:
┌─────────┐     ┌─────────┐
│ Medical │ →   │ Medical │   (color jitter on X-rays)
│  X-ray  │     │  X-ray  │
└─────────┘     └─────────┘
                Destroys diagnostic information!
```

**The Fix - Task-Appropriate Augmentation:**

```python
# For natural images - aggressive augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# For medical images - conservative augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Small rotation only
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    # NO color jitter - colors are diagnostic!
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# For documents/text images - no spatial distortion
train_transform = transforms.Compose([
    # NO rotation - text orientation matters!
    transforms.ColorJitter(brightness=0.2),  # Lighting only
    transforms.ToTensor(),
])
```

---

### Pitfall 2.3: Class Imbalance Ignorance

**The Problem:**
When one class dominates, model predicts majority class for everything.

```
CLASS IMBALANCE DISASTER
========================

Dataset:
   Class A (normal): 9,500 samples (95%)
   Class B (fraud):    500 samples (5%)

What happens:

   Naive model predicts "A" for everything:

   Accuracy = 95%!  ← Looks great!

   But:
   Recall on B = 0%  ← Catastrophic for fraud detection!


CONFUSION MATRIX REVEALS THE TRUTH:

                  Predicted
                  A      B
              ┌───────┬───────┐
   Actual  A  │ 9,500 │   0   │   ← All correct (trivial)
              ├───────┼───────┤
          B   │  500  │   0   │   ← All wrong! (missed all fraud)
              └───────┴───────┘
```

**The Fixes:**

```python
# Option 1: Class weights in loss function
class_counts = [9500, 500]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()  # Normalize
criterion = nn.CrossEntropyLoss(weight=weights)

# Option 2: Oversampling minority class
from torch.utils.data import WeightedRandomSampler

sample_weights = [1.0 if label == 0 else 19.0 for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
loader = DataLoader(dataset, sampler=sampler)

# Option 3: Focal loss (down-weight easy examples)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

---

## Part 3: Training Pitfalls

### Pitfall 3.1: Wrong Learning Rate

**The Problem:**
Learning rate is THE most important hyperparameter. Get it wrong, and
nothing else matters.

```
LEARNING RATE EFFECTS
=====================

Too High (lr=0.1):           Just Right (lr=0.001):      Too Low (lr=0.00001):

Loss                         Loss                        Loss
  │\  /\  /\                   │\                          │\
  │ \/  \/  \ diverges!        │ \                         │ \________________
  │          \                 │  \____                    │
  │           \                │       \____               │  (will converge
  │                            │            \_____         │   eventually...
  └────────────>               └────────────────────>      └────────────────────>
        Epochs                       Epochs                      Epochs

   EXPLODING                      OPTIMAL                    WASTING TIME


THE LEARNING RATE FINDER:

Loss
  │
  │\_
  │  \_
  │    \_____
  │          \       <- Optimal range
  │           \_____
  │                 \
  │                  \____
  │                       \_/ <- Too high (loss spikes)
  └──────────────────────────────>
  1e-7  1e-5  1e-3  1e-1  1e0
            Learning Rate

Use ~1/10 of the rate where loss starts increasing
```

**Concrete Implementation:**

```python
# Learning rate finder
def find_lr(model, train_loader, optimizer, criterion,
            start_lr=1e-7, end_lr=1, num_iter=100):

    lrs, losses = [], []
    lr = start_lr
    mult = (end_lr / start_lr) ** (1 / num_iter)

    for i, (x, y) in enumerate(train_loader):
        if i >= num_iter:
            break

        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        lrs.append(lr)
        losses.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        lr *= mult

    # Plot and find optimal
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()

# Rule of thumb starting points:
# Adam: 1e-3 to 3e-4
# SGD with momentum: 0.01 to 0.1
# Fine-tuning pretrained: 1e-5 to 1e-4
```

---

### Pitfall 3.2: Not Overfitting a Small Batch First

**The Problem:**
Debugging on the full dataset wastes hours. Bugs should be caught in minutes.

```
THE DEBUGGING HIERARCHY
=======================

                    Hours wasted
                         │
Full dataset  ───────────┼─────────────────────  (days)
                         │
1000 samples  ───────────┼───────────────  (hours)
                         │
100 samples   ───────────┼────────  (minutes)
                         │
10 samples    ───────────┼───  (seconds)  <- START HERE!
                         │


THE SANITY CHECK PROTOCOL:

Step 1: Overfit 10 samples

   Loss should → 0, Accuracy should → 100%

   If NOT:
   - Bug in model architecture
   - Bug in loss function
   - Bug in data pipeline
   - Learning rate too low/high

Step 2: Overfit 100 samples

   Should still reach ~100% training accuracy

   If NOT:
   - Model capacity too low
   - Harder optimization problem

Step 3: Run on full dataset

   Now monitor for overfitting vs underfitting
```

**Concrete Protocol:**

```python
def sanity_check(model, dataset, criterion, lr=1e-3, n_samples=10, epochs=100):
    """Must achieve ~100% accuracy on tiny subset"""

    # Create tiny subset
    subset = Subset(dataset, range(n_samples))
    loader = DataLoader(subset, batch_size=n_samples)  # One batch

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            acc = (out.argmax(1) == y).float().mean()
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")

    final_acc = (out.argmax(1) == y).float().mean()
    if final_acc < 0.99:
        print("WARNING: Failed to overfit small batch!")
        print("Check: model architecture, loss function, data pipeline")
    else:
        print("Sanity check passed!")
```

---

### Pitfall 3.3: Forgetting model.eval() and model.train()

**The Problem:**
BatchNorm and Dropout behave differently during training vs inference.
Forgetting to switch modes gives wrong results.

```
TRAIN vs EVAL MODE
==================

                     Training Mode              Eval Mode
                     (model.train())            (model.eval())

BatchNorm:           Uses batch statistics      Uses running statistics
                     (mean/var of current       (accumulated during
                      batch)                     training)

Dropout:             Randomly zeros neurons     Does nothing
                     (active)                   (all neurons active)


THE BUG:

# Training loop
model.train()
for x, y in train_loader:
    ...  # training

# Forgot model.eval()!
test_acc = evaluate(model, test_loader)  # WRONG RESULTS!

# What happens:
# - Dropout still dropping neurons (predictions too uncertain)
# - BatchNorm using test batch stats (noisy, inconsistent)


THE SYMPTOMS:

1. Test accuracy varies between runs (dropout randomness)
2. Different accuracy for different batch sizes (BatchNorm stats)
3. Training metrics much better than test metrics
```

**The Fix:**

```python
# ALWAYS use context manager or explicit mode setting

def evaluate(model, loader):
    model.eval()  # CRITICAL!
    with torch.no_grad():  # Also important - saves memory, faster
        correct = 0
        total = 0
        for x, y in loader:
            outputs = model(x)
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    model.train()  # Switch back!
    return correct / total

# Or use context manager pattern
class eval_mode:
    def __init__(self, model):
        self.model = model
        self.was_training = model.training

    def __enter__(self):
        self.model.eval()

    def __exit__(self, *args):
        if self.was_training:
            self.model.train()

# Usage
with eval_mode(model):
    predictions = model(test_data)
```

---

### Pitfall 3.4: Incorrect Loss Function for the Task

**The Problem:**
Using the wrong loss function for your task leads to poor optimization.

```
LOSS FUNCTION MATCHING
======================

Task                    WRONG Loss              RIGHT Loss
────                    ──────────              ──────────
Binary classification   MSE                     BCEWithLogitsLoss
Multi-class classif.    MSE                     CrossEntropyLoss
Regression              CrossEntropy            MSE or L1
Multi-label classif.    CrossEntropyLoss        BCEWithLogitsLoss (per label)
Semantic segmentation   CrossEntropy            CrossEntropy (pixel-wise)
                        (without ignore_index)  (with ignore_index for void)


COMMON MISTAKES:

1. Softmax + CrossEntropyLoss = DOUBLE SOFTMAX!

   # WRONG
   output = F.softmax(logits, dim=1)
   loss = F.cross_entropy(output, target)  # CE includes softmax!

   # RIGHT
   loss = F.cross_entropy(logits, target)  # Pass raw logits


2. Sigmoid + BCELoss = WRONG INPUT RANGE

   # WRONG (if logits can be negative)
   loss = F.binary_cross_entropy(logits, target)  # Expects 0-1!

   # RIGHT
   loss = F.binary_cross_entropy_with_logits(logits, target)


3. Wrong reduction mode

   # Returns single scalar - can't weight samples
   loss = F.cross_entropy(pred, target, reduction='mean')

   # Returns per-sample loss - can apply sample weights
   loss = F.cross_entropy(pred, target, reduction='none')
   weighted_loss = (loss * sample_weights).mean()
```

---

### Pitfall 3.5: Numerical Instability

**The Problem:**
Operations like exp, log, and division can produce NaN or Inf values.

```
NUMERICAL INSTABILITY SOURCES
=============================

1. LOG OF ZERO OR NEGATIVE

   y_pred = [0.0, 0.3, 0.7]
   log_pred = log(y_pred)  # log(0) = -inf!

   Fix: Add small epsilon
   log_pred = log(y_pred + 1e-8)


2. EXP OVERFLOW

   logits = [100, 200, 300]
   probs = exp(logits)  # exp(300) = inf!

   Fix: Use log-sum-exp trick
   # Don't do: softmax = exp(x) / sum(exp(x))
   # Do: softmax = exp(x - max(x)) / sum(exp(x - max(x)))


3. DIVISION BY ZERO

   normalized = x / x.std()  # std could be 0!

   Fix: Add epsilon to denominator
   normalized = x / (x.std() + 1e-8)


4. LARGE GRADIENT × SMALL LEARNING RATE ACCUMULATION

   After many updates with fp32:
   weight = weight - lr * gradient

   If weight >> lr * gradient, update might round to zero!

   Fix: Use mixed precision training properly
```

**Concrete Debugging:**

```python
def check_for_nan(model, loss, step):
    """Call during training to catch NaN early"""

    if torch.isnan(loss):
        print(f"NaN loss at step {step}!")

        # Check which parameters have NaN
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"  NaN in parameter: {name}")
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"  NaN in gradient: {name}")

        raise ValueError("NaN detected!")

# Safer implementations
def safe_log(x, eps=1e-8):
    return torch.log(x.clamp(min=eps))

def safe_divide(a, b, eps=1e-8):
    return a / (b + eps)

# Use built-in numerically stable versions
loss = F.cross_entropy(logits, targets)  # Stable
# NOT: loss = -(F.softmax(logits) * targets).sum()  # Unstable
```

---

## Part 4: Architecture Pitfalls

### Pitfall 4.1: Not Using Pretrained Models When You Should

**The Problem:**
Training from scratch when pretrained models exist wastes time and data.

```
PRETRAINED vs FROM SCRATCH
==========================

              From Scratch              Transfer Learning
              ─────────────              ─────────────────
Data needed:  100K+ images              100-1000 images
Training:     Days/weeks                Hours
Performance:  Often worse               Usually better
GPU cost:     $$$                       $


WHEN TO USE PRETRAINED:

✓ Natural images (ImageNet pretraining works great)
✓ Limited data (<10K samples)
✓ Standard objects/scenes
✓ Time/compute constraints

WHEN TO TRAIN FROM SCRATCH:

✗ Very different domain (satellite, medical, microscopy)
✗ Massive dataset available (>1M samples)
✗ Very different input (non-RGB, spectrograms, 3D)
✗ Extreme resolution requirements
```

**The Right Approach:**

```python
import torchvision.models as models

# DON'T: Train from scratch with little data
model = models.resnet50(pretrained=False)  # Random weights

# DO: Use pretrained and fine-tune
model = models.resnet50(pretrained=True)

# Freeze early layers (optional, for very small datasets)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Use lower learning rate for pretrained layers
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},  # New layer
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Fine-tune
    # Earlier layers stay frozen
])
```

---

### Pitfall 4.2: Wrong Receptive Field

**The Problem:**
Your network can't "see" enough of the input to make good decisions.

```
RECEPTIVE FIELD INTUITION
=========================

Receptive field = how much input each output neuron "sees"

Too small:                          Appropriate:

┌───────────────────────┐           ┌───────────────────────┐
│        🐱             │           │        🐱             │
│  ┌───┐                │           │  ┌─────────────┐      │
│  │ ? │ <- sees only   │           │  │             │      │
│  └───┘    whisker!    │           │  │   sees      │      │
│                       │           │  │   whole     │      │
│                       │           │  │   face!     │      │
│                       │           │  └─────────────┘      │
└───────────────────────┘           └───────────────────────┘

Can't recognize cat!                 Can recognize cat!


RECEPTIVE FIELD GROWTH:

3×3 conv → 3×3 conv → 3×3 conv → ... → 3×3 conv
   │          │          │                 │
   3          5          7               2n+1

Each 3×3 layer adds 2 to receptive field.
For 224×224 image, need ~100 layers to cover it!

SOLUTIONS:
1. Use pooling (doubles effective RF per pool)
2. Use dilated convolutions
3. Use larger kernel sizes in early layers
4. Use attention (global RF in one layer!)
```

**Checking Receptive Field:**

```python
def compute_receptive_field(layers):
    """Compute RF for a sequence of conv/pool layers"""
    rf = 1
    stride = 1
    for layer in layers:
        k = layer['kernel']
        s = layer['stride']
        d = layer.get('dilation', 1)

        rf = rf + (k - 1) * d * stride
        stride = stride * s

    return rf

# Example: VGG-style network
layers = [
    {'kernel': 3, 'stride': 1},  # conv
    {'kernel': 3, 'stride': 1},  # conv
    {'kernel': 2, 'stride': 2},  # pool
    {'kernel': 3, 'stride': 1},  # conv
    {'kernel': 3, 'stride': 1},  # conv
    {'kernel': 2, 'stride': 2},  # pool
]
print(f"Receptive field: {compute_receptive_field(layers)}")
# Output: Receptive field: 26 (still small!)
```

---

### Pitfall 4.3: Overcomplicated Architecture

**The Problem:**
Adding complexity that doesn't help and may hurt.

```
COMPLEXITY vs PERFORMANCE
=========================

Performance
    │
    │           ___________
    │          /           <- Plateau
    │         /
    │        /
    │       /
    │      /
    │_____/
    │
    └───────────────────────> Complexity

    Adding more complexity past the plateau:
    - Harder to train
    - Slower inference
    - More bugs
    - Same performance


SIGNS YOU'VE OVERCOMPLICATED:

1. Training is unstable (loss spikes)
2. Many hyperparameters to tune
3. Long training time
4. Can't explain why each component helps
5. Ablation shows components don't help
```

**The Simple Baseline Approach:**

```python
# STEP 1: Start stupidly simple
class SimpleBaseline(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# STEP 2: Add ONE thing at a time, measure improvement
class SlightlyBetter(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# STEP 3: Only add complexity if it MEASURABLY helps
# Each addition should give >1% improvement

# DON'T: Start with "state of the art" architecture
# DO: Start simple, add complexity only when needed
```

---

## Part 5: Evaluation Pitfalls

### Pitfall 5.1: Using Accuracy When It's Misleading

**The Problem:**
Accuracy hides important information, especially for imbalanced data.

```
METRIC SELECTION GUIDE
======================

Scenario                          Better Metrics
────────                          ──────────────
Balanced classification           Accuracy is fine
Imbalanced classification         F1, Precision, Recall, AUC-ROC
Cost-asymmetric (fraud, medical)  Weighted metrics, business cost
Multi-label                       Hamming loss, micro/macro F1
Detection                         mAP, IoU
Segmentation                      mIoU, Dice coefficient
Generation                        FID, IS, human eval


THE ACCURACY TRAP:

Spam detection: 1% spam, 99% not spam

   Model: "Nothing is spam"
   Accuracy: 99%  <- Sounds great!

   But:
   Precision: undefined (0/0)
   Recall: 0%  <- Catches NO spam!


CONFUSION MATRIX ANALYSIS:

                    Predicted
                    Spam    Not Spam
              ┌─────────┬───────────┐
    Actual    │   TP    │    FN     │
    Spam      │         │           │
              ├─────────┼───────────┤
    Not       │   FP    │    TN     │
    Spam      │         │           │
              └─────────┴───────────┘

Precision = TP / (TP + FP)   "Of predicted spam, how many are spam?"
Recall    = TP / (TP + FN)   "Of actual spam, how many did we catch?"
F1        = 2 * P * R / (P + R)   "Harmonic mean"
```

**Complete Evaluation:**

```python
from sklearn.metrics import classification_report, confusion_matrix

def full_evaluation(y_true, y_pred, y_prob=None):
    """Comprehensive classification evaluation"""

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        print(f"\n=== AUC-ROC: {roc_auc_score(y_true, y_prob):.4f} ===")

        # Calibration
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        print("\n=== Calibration ===")
        for pt, pp in zip(prob_true, prob_pred):
            print(f"  Predicted {pp:.2f} -> Actual {pt:.2f}")
```

---

### Pitfall 5.2: Testing on Training Data (Accidentally)

**The Problem:**
Various ways the test set gets contaminated by training information.

```
TEST CONTAMINATION SCENARIOS
============================

1. DIRECT OVERLAP

   train_data = load_all_data()
   test_data = load_all_data()  # Oops, same data!

   # Or: shuffled the whole dataset before splitting


2. PREPROCESSING CONTAMINATION

   scaler.fit(all_data)  # Fit on all
   train = scaler.transform(train_data)
   test = scaler.transform(test_data)  # Test stats leaked!


3. FEATURE SELECTION CONTAMINATION

   # Using test data to select features
   important_features = feature_selection(all_data)
   train_filtered = train_data[important_features]


4. HYPERPARAMETER TUNING ON TEST SET

   for lr in [0.1, 0.01, 0.001]:
       train(model, train_data)
       acc = evaluate(model, test_data)  # Should use VALIDATION!
       if acc > best:
           best_lr = lr

   # Test set is now validation set - you need a NEW test set


THE PROPER SPLIT:

   ┌─────────────────────────────────────────────────┐
   │                   ALL DATA                      │
   └─────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │   TRAIN     │ │ VALIDATION  │ │    TEST     │
   │   (60%)     │ │   (20%)     │ │   (20%)     │
   └─────────────┘ └─────────────┘ └─────────────┘
          │               │               │
          ▼               ▼               ▼
     Training        Hyperparameter    FINAL eval
                      tuning           (once only!)
                      early stopping
                      model selection
```

---

### Pitfall 5.3: Not Doing Proper Error Analysis

**The Problem:**
Looking only at aggregate metrics, not understanding WHY the model fails.

```
ERROR ANALYSIS PROCESS
======================

Step 1: Look at INDIVIDUAL failures

   Instead of: "Accuracy is 85%"
   Ask: "What do the 15% failures look like?"

Step 2: Categorize failures

   ┌────────────────────────────────────────────┐
   │ Failure Type           Count    Percentage │
   ├────────────────────────────────────────────┤
   │ Ambiguous labels         45        30%     │
   │ Occlusion                30        20%     │
   │ Unusual viewpoint        25        17%     │
   │ Small object             20        13%     │
   │ Low quality image        15        10%     │
   │ True model error         15        10%     │
   └────────────────────────────────────────────┘

   Only 10% are true model errors!
   Fix the data, and accuracy could hit 95%!

Step 3: Prioritize fixes

   Ambiguous labels (30%) -> Clean annotations
   Occlusion (20%) -> Augmentation or architecture change
   Unusual viewpoint (17%) -> More training data
```

**Concrete Error Analysis:**

```python
def error_analysis(model, dataset, num_samples=100):
    """Visualize and categorize errors"""

    model.eval()
    errors = []

    for i in range(len(dataset)):
        x, y = dataset[i]
        with torch.no_grad():
            pred = model(x.unsqueeze(0)).argmax().item()

        if pred != y:
            errors.append({
                'index': i,
                'true': y,
                'predicted': pred,
                'confidence': F.softmax(model(x.unsqueeze(0)), dim=1).max().item(),
                'image': x
            })

    print(f"Total errors: {len(errors)} / {len(dataset)}")

    # Confusion pairs
    from collections import Counter
    pairs = Counter([(e['true'], e['predicted']) for e in errors])
    print("\nMost common confusions:")
    for (true, pred), count in pairs.most_common(10):
        print(f"  {true} -> {pred}: {count} times")

    # Show worst errors (highest confidence wrong predictions)
    errors.sort(key=lambda e: -e['confidence'])
    print("\nHighest confidence errors (most concerning):")
    for e in errors[:10]:
        print(f"  True: {e['true']}, Pred: {e['predicted']}, Conf: {e['confidence']:.2%}")

    return errors
```

---

## Part 6: Production Pitfalls

### Pitfall 6.1: Training/Inference Mismatch

**The Problem:**
Model behaves differently in production than during training.

```
TRAINING/INFERENCE DISCREPANCIES
================================

1. DIFFERENT PREPROCESSING

   Training:                    Production:
   RGB order                    BGR order (OpenCV default!)
   Normalized [0,1]            Normalized [-1,1]
   Resized to 224×224          Resized to 256×256

   RESULT: Complete garbage predictions!


2. DIFFERENT DATA DISTRIBUTION

   Training:                    Production:
   Professional photos          User phone photos
   Centered objects             Objects at edges
   Good lighting                Low light, motion blur

   RESULT: Accuracy drops from 95% to 70%


3. BATCH SIZE DIFFERENCES (with BatchNorm)

   Training:                    Production:
   Batch size 64                Batch size 1

   BatchNorm statistics from 64 samples vs 1 sample
   RESULT: Noisy predictions for single samples

   FIX: Use model.eval() mode!
```

**Production Checklist:**

```python
class ProductionModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()  # CRITICAL!

        # Store EXACT preprocessing used during training
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # EXACT same normalization as training!
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):
        """Handles single image prediction"""

        # Ensure consistent input format
        if isinstance(image, np.ndarray):
            # OpenCV uses BGR, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Apply EXACT same preprocessing
        x = self.transform(image)
        x = x.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(x)
            prob = F.softmax(output, dim=1)

        return prob.squeeze().numpy()
```

---

### Pitfall 6.2: Not Monitoring Model Performance

**The Problem:**
Model degrades over time due to data drift, but nobody notices.

```
MODEL DEGRADATION OVER TIME
===========================

Accuracy
   │
   │ ────────────────
   │                  \
   │                   \
   │                    \──────  <- Data drift!
   │                           \
   │                            \
   └─────────────────────────────────> Time
        Deploy           6 months later


CAUSES OF DRIFT:

1. Concept drift: The relationship between inputs and outputs changes
   Example: Fashion trends change, old "stylish" is now "outdated"

2. Data drift: Input distribution changes
   Example: New camera models produce different image characteristics

3. Label drift: What users consider correct changes
   Example: User expectations evolve


WHAT TO MONITOR:

┌────────────────────────────────────────────────────┐
│ Metric                   Alert Threshold           │
├────────────────────────────────────────────────────┤
│ Prediction confidence    Mean drops below 0.8      │
│ Class distribution       >10% change from training │
│ Input feature stats      >2 std from training mean │
│ Latency                  >100ms p95                │
│ Error rate               >5% increase              │
└────────────────────────────────────────────────────┘
```

---

### Pitfall 6.3: Ignoring Latency and Memory Constraints

**The Problem:**
Model works great offline but is too slow/large for production.

```
MODEL EFFICIENCY TRADEOFFS
==========================

                ResNet-152    ResNet-50    MobileNetV2    SqueezeNet
                ──────────    ─────────    ───────────    ──────────
Accuracy:         78.3%        76.1%         71.8%          57.5%
Parameters:       60M          25M           3.4M           1.2M
Inference (ms):   85           45            25             12
Size (MB):        230          98            14             5


OPTIMIZATION TECHNIQUES:

1. QUANTIZATION (float32 -> int8)

   Size reduction: 4×
   Speed improvement: 2-4×
   Accuracy drop: 0.5-2%

2. PRUNING (remove unimportant weights)

   Size reduction: 2-10×
   Speed improvement: 1.5-3×
   Accuracy drop: 0-1%

3. KNOWLEDGE DISTILLATION

   Train small "student" to mimic large "teacher"
   Can get 90% of teacher performance with 10% of size

4. ARCHITECTURE SEARCH

   Use efficient architectures designed for mobile
   MobileNet, EfficientNet, ShuffleNet
```

**Practical Optimization:**

```python
import torch.quantization

# Post-training quantization
def quantize_model(model):
    model.eval()

    # Fuse conv+bn+relu
    model_fused = torch.quantization.fuse_modules(
        model, [['conv', 'bn', 'relu']]
    )

    # Quantize
    model_quantized = torch.quantization.quantize_dynamic(
        model_fused,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )

    return model_quantized

# Compare sizes
original_size = sum(p.numel() * 4 for p in model.parameters())  # float32
quantized_size = sum(p.numel() for p in quantized_model.parameters())  # int8

print(f"Original: {original_size / 1e6:.1f} MB")
print(f"Quantized: {quantized_size / 1e6:.1f} MB")
print(f"Reduction: {original_size / quantized_size:.1f}x")
```

---

## Part 7: Mental Models for Debugging

### The Debugging Decision Tree

```
MODEL NOT WORKING?
==================

Start here
    │
    ▼
Is loss decreasing? ──No──► Learning rate issue
    │                       - Too high: loss oscillates/explodes
    │                       - Too low: loss barely moves
    │                       - Try lr finder
    Yes
    │
    ▼
Does training acc improve? ──No──► Model capacity issue
    │                              - Too small: can't fit training data
    │                              - Bug in architecture
    │                              - Try overfitting single batch
    Yes
    │
    ▼
Does validation acc improve? ──No──► Overfitting
    │                                - Add regularization
    │                                - Get more data
    │                                - Reduce model size
    │                                - Use pretrained model
    Yes
    │
    ▼
Is val acc good enough? ──No──► Underfitting or data problem
    │                           - Model too simple
    │                           - Not enough training
    │                           - Bad data quality
    │                           - Wrong problem framing
    Yes
    │
    ▼
Does test acc match val acc? ──No──► Validation set not representative
    │                                - Resplit data
    │                                - Check for data leakage
    │                                - Use cross-validation
    Yes
    │
    ▼
Does production match test? ──No──► Distribution shift
    │                               - Train/prod preprocessing mismatch
    │                               - Prod data is different
    │                               - Need domain adaptation
    Yes
    │
    ▼
SUCCESS!
```

### Common Symptoms and Causes

```
SYMPTOM → DIAGNOSIS TABLE
=========================

Symptom                          Likely Causes
───────                          ────────────
Loss = NaN                       • Exploding gradients
                                 • log(0) or division by zero
                                 • Learning rate too high

Loss doesn't decrease            • Learning rate too low
                                 • Bug in loss function
                                 • Dead ReLUs

Loss decreases then explodes     • Learning rate too high
                                 • Numerical instability
                                 • Bad batch (outliers)

Train acc high, val acc low      • Overfitting
                                 • Data leakage
                                 • Train/val mismatch

Both train and val acc low       • Underfitting
                                 • Model too simple
                                 • Bug in model
                                 • Impossible task

Predictions all same class       • Class imbalance
                                 • Last layer bias issue
                                 • Vanishing gradients

Predictions are random           • Model not training
                                 • Data labels shuffled
                                 • Wrong data fed to model

Val acc varies across runs       • Forgot model.eval()
                                 • Small val set
                                 • High variance model
```

---

## Summary: The Practitioner's Checklist

```
BEFORE TRAINING
===============
□ Verified no data leakage between splits
□ Checked class balance, planned mitigation if needed
□ Preprocessing fit ONLY on training data
□ Appropriate augmentation for domain
□ Verified data loading (visualize samples with labels)
□ Started with pretrained model if applicable
□ Sanity checked: can overfit small batch

DURING TRAINING
===============
□ Monitoring BOTH train and val loss
□ Using appropriate learning rate (lr finder)
□ Gradient clipping for RNNs
□ model.train() during training
□ model.eval() during validation
□ Logging metrics to track progress
□ Saving checkpoints

AFTER TRAINING
==============
□ Evaluated on held-out TEST set (only once!)
□ Used appropriate metrics for task
□ Performed error analysis on failures
□ Checked calibration of probabilities
□ Verified no train/val/test overlap

FOR PRODUCTION
==============
□ model.eval() always
□ Preprocessing EXACTLY matches training
□ Quantized/optimized for latency
□ Monitoring for data drift
□ Logging predictions for debugging
□ Fallback strategy for edge cases
```

---

## Final Wisdom

```
THE THREE LAWS OF DEEP LEARNING DEBUGGING
=========================================

1. IT'S PROBABLY A DATA PROBLEM

   Before blaming the model, check:
   - Are labels correct?
   - Is preprocessing consistent?
   - Is there enough data?
   - Is the data representative?

2. START SIMPLE, ADD COMPLEXITY

   Can't debug what you don't understand.
   Get a simple baseline working first.
   Add one thing at a time.
   Measure impact of each addition.

3. TRUST BUT VERIFY

   Don't assume code is correct.
   Visualize intermediate outputs.
   Check tensor shapes.
   Print summary statistics.
   Unit test critical components.


THE MASTER'S MINDSET:
=====================

      "The model is not wrong,
       it's doing exactly what you told it.
       Find out what you told it wrong."
```

---

*This guide covers the most common pitfalls encountered in deep learning practice.
When stuck, return to basics: verify data, simplify the model, and check assumptions.*
