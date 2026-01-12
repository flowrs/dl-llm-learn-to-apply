# Week 3-4: Convolutional Neural Networks & Training

> Combined guide merging CSE 493G1/CS231n course materials with Stanford CS231n notes

## Overview

This document covers:
- Why CNNs for images (spatial structure, parameter sharing)
- Convolutional layers, pooling, and architectures
- Training techniques: initialization, normalization, regularization
- Optimizers and learning rate schedules
- CNN visualization and understanding

---

## Part 1: Why Convolutional Neural Networks?

### The Problem with Fully-Connected Networks

```
    Fully-Connected on Images: Parameter Explosion
    ═══════════════════════════════════════════════

    Input: 200×200×3 image = 120,000 pixels
    First hidden layer with 1000 neurons:
    Parameters = 120,000 × 1000 = 120 MILLION (just layer 1!)

    Problems:
    1. Too many parameters → overfitting
    2. No spatial structure exploited
    3. Same edge detector learned at every position
```

### CNN Key Insights

**1. Local Connectivity**: Neurons connect to small local regions

**2. Parameter Sharing**: Same filter applied across entire image

**3. Translation Equivariance**: Detecting feature at one location helps detect it elsewhere

```
    CNN vs Fully-Connected
    ══════════════════════

    Fully-Connected:              CNN:
    Every pixel to every neuron   Local regions, shared weights

    ┌─────────────────┐           ┌─────────────────┐
    │●●●●●●●●●●●●●●●●●│           │    ┌───┐        │
    │●●●●●●●●●●●●●●●●●│           │    │ F │ slides │
    │●●●●●●●●●●●●●●●●●│           │    │ i │ across │
    │●●●●●●●●●●●●●●●●●│           │    │ l │ image  │
    └────────┬────────┘           │    │ t │        │
             │                    │    │ e │        │
         all connect              │    │ r │        │
             │                    │    └───┘        │
    ┌────────▼────────┐           └─────────────────┘
    │    Hidden       │           Same filter everywhere!
    └─────────────────┘
```

---

## Part 2: Convolutional Layer

### The Convolution Operation

A filter (kernel) slides across the input, computing dot products:

```
    Convolution Operation
    ═════════════════════

    Input (5×5)              Filter (3×3)           Output
    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │ 0 │ 1 │    │ 1 │ 0 │-1 │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┤
    │ 0 │ 1 │ 2 │ 3 │ 2 │    │ 1 │ 0 │-1 │         ┌───┬───┬───┐
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┤         │-1 │-2 │ 1 │
    │ 1 │ 1 │ 1 │ 1 │ 1 │    │ 1 │ 0 │-1 │   ──▶   ├───┼───┼───┤
    ├───┼───┼───┼───┼───┤    └───┴───┴───┘         │ 0 │ 2 │ 2 │
    │ 2 │ 2 │ 0 │ 0 │ 1 │                          ├───┼───┼───┤
    ├───┼───┼───┼───┼───┤    Edge detector!        │-1 │ 1 │ 3 │
    │ 1 │ 0 │ 1 │ 2 │ 1 │    (vertical edges)      └───┴───┴───┘
    └───┴───┴───┴───┴───┘
```

### Output Size Formula

$$O = \frac{W - F + 2P}{S} + 1$$

Where:
- W = Input size
- F = Filter size
- P = Padding
- S = Stride

```
    Hyperparameters Effect
    ══════════════════════

    Input: 7×7, Filter: 3×3

    Stride=1, Pad=0:          Stride=2, Pad=0:          Stride=1, Pad=1:
    Output = (7-3+0)/1+1      Output = (7-3+0)/2+1      Output = (7-3+2)/1+1
           = 5×5                     = 3×3                     = 7×7
                                                        (preserves size!)
```

### 3D Volumes

Convolutions work on 3D volumes (Width × Height × Depth):

```
    3D Convolution
    ══════════════

    Input Volume           Filter              Output (one slice)
    [32×32×3]             [5×5×3]             [28×28]

    ┌─────────────┐       ┌─────┐             ┌─────────┐
    │  R  G  B    │       │     │             │         │
    │  ┌──┬──┬──┐ │   ×   │  ×  │   =         │  one    │
    │  │  │  │  │ │       │     │             │  number │
    │  └──┴──┴──┘ │       └─────┘             └─────────┘
    └─────────────┘

    Filter extends through FULL depth of input!
    One filter → one activation map
    K filters → K activation maps (output depth = K)
```

### Parameter Sharing

```
    Parameter Savings
    ═════════════════

    Without Sharing (hypothetical):
    55×55×96 neurons, each with 11×11×3+1 weights
    = 290,400 × 364 = 105,705,600 parameters

    With Sharing:
    96 filters, each with 11×11×3+1 weights
    = 96 × 364 = 34,944 parameters

    Reduction: ~3000x fewer parameters!
```

---

## Part 3: Pooling Layer

### Purpose

Reduce spatial dimensions while retaining important information:

```
    Max Pooling (2×2, stride 2)
    ═══════════════════════════

    Input (4×4)                    Output (2×2)
    ┌───┬───┬───┬───┐              ┌───┬───┐
    │ 1 │ 3 │ 2 │ 1 │              │   │   │
    ├───┼───┼───┼───┤              │ 4 │ 6 │
    │ 4 │ 2 │ 6 │ 4 │     ──▶      ├───┼───┤
    ├───┼───┼───┼───┤              │ 8 │ 4 │
    │ 3 │ 1 │ 2 │ 1 │              │   │   │
    ├───┼───┼───┼───┤              └───┴───┘
    │ 8 │ 5 │ 4 │ 2 │
    └───┴───┴───┴───┘         Take max in each region
```

### Pooling Types

| Type | Operation | Use Case |
|------|-----------|----------|
| Max Pooling | Maximum value | Most common, preserves strong activations |
| Average Pooling | Mean value | Smoother, sometimes at end |
| Global Average Pooling | Mean over entire feature map | Replace FC at end |

---

## Part 4: CNN Architectures

### Standard Pattern

```
    Typical CNN Architecture
    ════════════════════════

    INPUT → [[CONV → RELU]*N → POOL?]*M → [FC → RELU]*K → FC

    Example for CIFAR-10:
    ┌─────────────────────────────────────────────────────────┐
    │ INPUT    CONV    RELU    POOL    FC      FC     OUTPUT  │
    │ 32×32×3  32×32×32        16×16×32       512     10      │
    │    │       │       │       │       │      │       │     │
    │    ▼       ▼       ▼       ▼       ▼      ▼       ▼     │
    │   ┌─┐    ┌───┐   ┌───┐   ┌──┐   ┌───┐  ┌───┐   ┌───┐   │
    │   │ │    │   │   │   │   │  │   │   │  │   │   │   │   │
    │   └─┘    └───┘   └───┘   └──┘   └───┘  └───┘   └───┘   │
    └─────────────────────────────────────────────────────────┘
```

### Classic Architectures

```
    Architecture Evolution
    ══════════════════════

    LeNet-5 (1998):     5 layers, 60K params
    │
    ▼
    AlexNet (2012):     8 layers, 60M params    ← Deep Learning revolution!
    │                   11×11 filters, ReLU, Dropout
    ▼
    VGGNet (2014):      19 layers, 138M params
    │                   Only 3×3 filters (simpler but deeper)
    ▼
    GoogLeNet (2014):   22 layers, 4M params
    │                   Inception modules (parallel paths)
    ▼
    ResNet (2015):      152 layers, 60M params
                        Skip connections (residual learning)
```

### VGGNet Pattern

```
    VGG Philosophy: Simplicity Through Depth
    ════════════════════════════════════════

    Only 3×3 convolutions!

    Why 3×3?
    - Two 3×3 layers = one 5×5 receptive field
    - Three 3×3 layers = one 7×7 receptive field
    - But with more non-linearities and fewer parameters!

    Receptive Field:
    ┌───────────────┐
    │ ┌───────────┐ │     Two 3×3 layers:
    │ │ ┌───────┐ │ │     - Params: 2×(3×3×C×C) = 18C²
    │ │ │ 3×3   │ │ │
    │ │ │       │ │ │     One 5×5 layer:
    │ │ └───────┘ │ │     - Params: 5×5×C×C = 25C²
    │ └───────────┘ │
    └───────────────┘     Fewer params + more non-linearity!
```

### ResNet Skip Connections

```
    Residual Block
    ══════════════

    Standard Block:              Residual Block:

        x                            x ─────────────┐
        │                            │              │
        ▼                            ▼              │
    ┌───────┐                    ┌───────┐         │
    │ CONV  │                    │ CONV  │         │
    └───────┘                    └───────┘         │
        │                            │              │
        ▼                            ▼              │
    ┌───────┐                    ┌───────┐         │
    │ CONV  │                    │ CONV  │         │
    └───────┘                    └───────┘         │
        │                            │              │
        ▼                            ▼              │
        H(x)                     F(x) + x ◄────────┘

                                Skip Connection!
                                Learns residual: F(x) = H(x) - x
                                If optimal H(x) = x, just learn F(x) = 0
```

---

## Part 5: Training Techniques

### Data Preprocessing

```python
# Zero-center and normalize
X -= np.mean(X, axis=0)  # Subtract mean
X /= np.std(X, axis=0)   # Divide by std

# For images: often just subtract mean (per-channel or global)
X -= np.array([123.68, 116.779, 103.939])  # ImageNet means
```

### Weight Initialization

**Bad**: All zeros (symmetry problem)
**Bad**: Too small/large (vanishing/exploding gradients)

```python
# Xavier initialization (for tanh/sigmoid)
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)

# He initialization (for ReLU) - recommended
W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

### Batch Normalization

Normalize activations to reduce internal covariate shift:

```
    Batch Normalization
    ═══════════════════

    For each mini-batch:

    1. Compute mean:      μ = (1/m) Σ xᵢ
    2. Compute variance:  σ² = (1/m) Σ (xᵢ - μ)²
    3. Normalize:         x̂ = (x - μ) / √(σ² + ε)
    4. Scale and shift:   y = γx̂ + β    (learnable!)

    Benefits:
    ✓ Higher learning rates
    ✓ Less sensitive to initialization
    ✓ Acts as regularization
    ✓ Faster convergence
```

### Dropout

Randomly zero neurons during training:

```
    Dropout (p=0.5)
    ═══════════════

    Training:                    Test:
    ┌─●─●─●─●─●─┐                ┌─●─●─●─●─●─┐
    │ │ │ │ │ │ │                │ │ │ │ │ │ │
    │ ○─●─○─●─○ │  randomly      │ ●─●─●─●─● │  use all,
    │ │ │ │ │ │ │  drop 50%      │ │ │ │ │ │ │  scale by p
    └─●─○─●─○─●─┘                └─●─●─●─●─●─┘

    Inverted dropout (preferred):
    - Training: scale by 1/p when dropping
    - Test: use weights unchanged
```

```python
# Inverted dropout implementation
def dropout_forward(x, p=0.5, training=True):
    if training:
        mask = (np.random.rand(*x.shape) < p) / p
        return x * mask
    return x
```

### Data Augmentation

```
    Common Augmentations
    ════════════════════

    Original    Flip       Crop       Color       Rotate
    ┌─────┐    ┌─────┐    ┌───┐      ┌─────┐    ┌─────┐
    │ 🐱  │    │  🐱 │    │🐱 │      │ 🐱  │    │  🐱 │
    │     │ ▶  │     │    │   │      │tint │    │   ↺ │
    └─────┘    └─────┘    └───┘      └─────┘    └─────┘

    Advanced:
    - Mixup: blend two images and labels
    - Cutout: random erasing
    - AutoAugment: learned augmentation policies
```

---

## Part 6: Optimization

### Optimizer Comparison

```
    SGD vs Momentum vs Adam
    ═══════════════════════

    SGD:           Follow gradient directly
                   θ = θ - α∇L

    Momentum:      Build up "velocity"
                   v = βv + ∇L
                   θ = θ - αv

    Adam:          Adaptive learning rates + momentum
                   Combines RMSprop + Momentum

    Loss Surface Visualization:
    ┌─────────────────────────────────────────┐
    │                    SGD                  │
    │         ──→──→──→──→──→●               │
    │        /                                │
    │       /     Momentum                    │
    │      ──────→────────→●                 │
    │     /                                   │
    │    /        Adam                        │
    │   ─────→────→●                         │
    │  start                    minimum       │
    └─────────────────────────────────────────┘
```

### Learning Rate Schedules

```python
# Step decay
if epoch % 30 == 0:
    lr *= 0.1

# Exponential decay
lr = lr_0 * decay_rate ** (epoch / decay_steps)

# Cosine annealing
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / T))

# Warmup (for large batches)
if epoch < warmup_epochs:
    lr = lr_max * epoch / warmup_epochs
```

```
    Learning Rate Schedules
    ═══════════════════════

    Step Decay:              Cosine Annealing:        Warmup + Decay:
    LR                       LR                       LR
    │▔▔▔▔▔┐                  │▔▔▔\                    │    /▔▔▔▔▔\
    │     └───┐              │    \                   │   /       \
    │         └───┐          │     \                  │  /         \
    │             └───       │      \_____            │ /           \___
    └─────────────────       └─────────────           └─────────────────
         epochs                   epochs                   epochs
```

### Hyperparameter Search

**Random search > Grid search**:
- More efficient for high-dimensional spaces
- Better coverage of important dimensions

```python
# Log-scale for learning rate
lr = 10 ** np.random.uniform(-5, -2)  # 1e-5 to 1e-2

# Coarse-to-fine search
# 1. Broad search: few epochs, many configs
# 2. Narrow search: more epochs, promising regions
```

---

## Part 7: Understanding CNNs

### What Do CNNs Learn?

```
    Feature Hierarchy
    ═════════════════

    Layer 1: Edge detectors
    ┌───┬───┬───┬───┐
    │ / │ \ │ ─ │ │ │  Gabor-like filters
    └───┴───┴───┴───┘

    Layer 2-3: Textures and patterns
    ┌───┬───┬───┬───┐
    │░░░│▓▓▓│╱╲╱│○○○│  Grid patterns, circles
    └───┴───┴───┴───┘

    Layer 4-5: Parts and objects
    ┌───┬───┬───┬───┐
    │👁│👂│🐾│🚗│  Eyes, wheels, faces
    └───┴───┴───┴───┘
```

### Visualization Techniques

**1. Filter Visualization**: Show learned weights

**2. Activation Maps**: What fires for an input

**3. Saliency Maps**: Gradient of output w.r.t. input

**4. Grad-CAM**: Class-specific attention

```
    Grad-CAM Visualization
    ══════════════════════

    Input Image          Class: "Cat"          Class: "Dog"
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │   🐱 🐕     │      │   ████      │      │        ████│
    │             │  →   │   ████      │      │        ████│
    │             │      │             │      │             │
    └─────────────┘      └─────────────┘      └─────────────┘
                         highlights cat       highlights dog
```

---

## Part 8: Practical Guidelines

### Architecture Guidelines

| Component | Recommendation |
|-----------|----------------|
| Filter size | 3×3 (sometimes 1×1, 5×5) |
| Stride | 1 for conv, 2 for downsampling |
| Padding | Same padding (preserve size) |
| Pooling | 2×2 max pooling, stride 2 |
| Channels | Double after each pooling (64→128→256) |
| FC layers | Minimize or replace with global avg pool |

### Training Guidelines

| Parameter | Starting Point |
|-----------|----------------|
| Batch size | 32-256 (larger with BatchNorm) |
| Learning rate | 1e-3 (Adam), 1e-1 (SGD+momentum) |
| Weight decay | 1e-4 to 1e-5 |
| Dropout | 0.5 (FC layers) |
| Epochs | Until validation plateaus |

### Debugging Checklist

```
    Training Diagnosis
    ══════════════════

    Symptom: Loss not decreasing
    → Check learning rate (try 10x higher/lower)
    → Check for bugs (gradient check)
    → Check data (are labels correct?)

    Symptom: Training loss decreases, val loss increases
    → Overfitting! Add regularization
    → Data augmentation
    → Reduce model capacity

    Symptom: Both losses plateau early
    → Underfitting. Increase capacity
    → Train longer
    → Better hyperparameters
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **Why CNNs** | Local connectivity, parameter sharing, translation equivariance |
| **Conv Layer** | Filter × input, output size = (W-F+2P)/S+1 |
| **Pooling** | Spatial downsampling, max pooling most common |
| **Architectures** | VGG (3×3 only), ResNet (skip connections) |
| **BatchNorm** | Normalize activations, enables higher LR |
| **Dropout** | Regularization via random neuron dropping |
| **Optimizers** | Adam default, SGD+momentum for best results |
| **LR Schedule** | Start high, decay (step/cosine/warmup) |

---

## References

**Course Materials:**
- [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [CS231n: Neural Networks 2 (Training)](https://cs231n.github.io/neural-networks-2/)
- [CS231n: Neural Networks 3 (Optimization)](https://cs231n.github.io/neural-networks-3/)

**Key Papers:**
- AlexNet: Krizhevsky et al., 2012
- VGGNet: Simonyan & Zisserman, 2014
- GoogLeNet: Szegedy et al., 2014
- ResNet: He et al., 2015
- Batch Normalization: Ioffe & Szegedy, 2015
