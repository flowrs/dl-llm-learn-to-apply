# Module 3: Convolutional Neural Networks

## Learning Objectives

By the end of this module, you will understand:
- Why CNNs are designed for image data
- The convolution operation and its properties
- Pooling layers for spatial reduction
- Classic CNN architectures (LeNet to ResNet)
- Transfer learning with pre-trained models

---

## 3.1 Motivation: Why Not Fully-Connected?

### The Problem with FC Layers for Images

Consider a modest 200×200 RGB image:
- Input size: 200 × 200 × 3 = 120,000 pixels
- One FC layer with 1000 neurons: 120,000 × 1000 = **120 million parameters**

Problems:
1. **Too many parameters** → overfitting, memory issues
2. **No spatial structure** → treats adjacent pixels same as distant ones
3. **Not translation invariant** → cat in corner ≠ cat in center

### CNNs: Key Ideas

1. **Local connectivity**: Neurons connect to small local regions
2. **Parameter sharing**: Same filter applied across entire image
3. **Translation equivariance**: Features detected anywhere in image

---

## 3.2 The Convolution Operation

### What is Convolution?

A **filter** (kernel) slides across the image, computing dot products:

```
Input (5×5):               Filter (3×3):           Output (3×3):
┌─────────────────┐        ┌───────────┐          ┌───────────┐
│ 1  2  3  4  5   │        │ 1  0  -1  │          │ ?  ?  ?   │
│ 6  7  8  9  10  │   *    │ 1  0  -1  │    =     │ ?  ?  ?   │
│ 11 12 13 14 15  │        │ 1  0  -1  │          │ ?  ?  ?   │
│ 16 17 18 19 20  │        └───────────┘          └───────────┘
│ 21 22 23 24 25  │
└─────────────────┘
```

**One position calculation**:
```
Output[0,0] = 1×1 + 2×0 + 3×(-1) +
              6×1 + 7×0 + 8×(-1) +
              11×1 + 12×0 + 13×(-1)
            = 1 - 3 + 6 - 8 + 11 - 13 = -6
```

### Filter as Feature Detector

Different filters detect different features:

```
Edge detector        Blur              Sharpen
  (vertical)
┌───────────┐     ┌───────────┐     ┌───────────┐
│ -1  0  1  │     │ 1/9 1/9 1/9│     │  0  -1  0 │
│ -1  0  1  │     │ 1/9 1/9 1/9│     │ -1   5 -1 │
│ -1  0  1  │     │ 1/9 1/9 1/9│     │  0  -1  0 │
└───────────┘     └───────────┘     └───────────┘
```

In CNNs, **filter values are learned** from data.

### Convolution Hyperparameters

**Filter size (F)**: Typically 3×3, 5×5, 7×7 (odd numbers for center pixel)

**Stride (S)**: Step size when sliding filter
- Stride 1: Moves 1 pixel at a time (default)
- Stride 2: Moves 2 pixels (reduces output size by ~half)

**Padding (P)**: Zeros added around border
- "Valid": No padding, output shrinks
- "Same": Pad so output size = input size

### Output Size Formula

```
Output size = (Input size - Filter size + 2×Padding) / Stride + 1
            = (W - F + 2P) / S + 1
```

**Example**: Input 32×32, Filter 5×5, Stride 1, Padding 2
```
Output = (32 - 5 + 4) / 1 + 1 = 32×32 (same size)
```

### Multiple Filters = Multiple Feature Maps

One filter produces one **feature map** (activation map).
Multiple filters produce a volume:

```
Input:          Filters:              Output:
[H × W × 3]  ×  [K × F × F × 3]  =   [H' × W' × K]
                 (K filters)          (K feature maps)

32×32×3     ×   64 filters 3×3   =   32×32×64
```

### Parameter Count

For a conv layer with:
- K filters of size F×F
- Input depth D

```
Parameters = K × (F × F × D + 1)    (weights + bias per filter)

Example: 64 filters, 3×3, input depth 3
Parameters = 64 × (3×3×3 + 1) = 64 × 28 = 1,792
```

Compare to FC: 64 neurons on 32×32×3 input = 196,672 parameters!

---

## 3.3 Pooling Layers

### Purpose

**Pooling** reduces spatial dimensions:
- Reduces computation
- Provides translation invariance
- Controls overfitting

### Max Pooling

Take maximum value in each region:

```
Input (4×4):                Max Pool 2×2, Stride 2:
┌─────────────┐             ┌───────┐
│ 1  3 │ 2  1 │             │ 4 │ 3 │
│ 4  2 │ 1  3 │    →        │───┼───│
├─────────────┤             │ 7 │ 4 │
│ 7  1 │ 3  2 │             └───────┘
│ 2  5 │ 4  1 │
└─────────────┘
```

**No learnable parameters** in pooling.

### Average Pooling

Take average value in each region. Sometimes used in final layers (global average pooling).

### Common Setting

- **2×2 pooling with stride 2**: Halves each spatial dimension
- Input: [H, W, D] → Output: [H/2, W/2, D]
- Discards 75% of activations

---

## 3.4 CNN Architecture Patterns

### Basic Pattern

```
[CONV → RELU] × N → POOL → ... → FC → OUTPUT
```

More detailed:
```
INPUT → [[CONV → RELU]*N → POOL?]*M → [FC → RELU]*K → FC
```

### Layer Progression

As we go deeper:
- **Spatial dimensions decrease** (pooling)
- **Depth increases** (more filters)
- **Features become more abstract**

```
Layer   Output Size   What it learns
──────────────────────────────────────
Input   224×224×3     Pixels
Conv1   112×112×64    Edges, colors
Conv2   56×56×128     Textures, patterns
Conv3   28×28×256     Parts (eyes, wheels)
Conv4   14×14×512     Objects
Conv5   7×7×512       Scenes, concepts
FC      1×1×1000      Class scores
```

---

## 3.5 Classic Architectures

### LeNet-5 (1998)

The original CNN for digit recognition:

```
INPUT (32×32×1)
  ↓
CONV 5×5, 6 filters → 28×28×6
  ↓
POOL 2×2 → 14×14×6
  ↓
CONV 5×5, 16 filters → 10×10×16
  ↓
POOL 2×2 → 5×5×16
  ↓
FC → 120
  ↓
FC → 84
  ↓
FC → 10 (digits)
```

~60K parameters

### AlexNet (2012)

Won ImageNet 2012, launched deep learning revolution:

```
INPUT (227×227×3)
  ↓
CONV 11×11, 96 filters, stride 4 → 55×55×96
  ↓
MAX POOL 3×3, stride 2 → 27×27×96
  ↓
CONV 5×5, 256 filters → 27×27×256
  ↓
MAX POOL 3×3, stride 2 → 13×13×256
  ↓
CONV 3×3, 384 filters → 13×13×384
  ↓
CONV 3×3, 384 filters → 13×13×384
  ↓
CONV 3×3, 256 filters → 13×13×256
  ↓
MAX POOL 3×3, stride 2 → 6×6×256
  ↓
FC → 4096
  ↓
FC → 4096
  ↓
FC → 1000 (ImageNet classes)
```

~60M parameters

**Innovations**: ReLU, dropout, data augmentation, GPU training

### VGGNet (2014)

Key insight: **Small 3×3 filters everywhere**

```
Two 3×3 convs have same receptive field as one 5×5
Three 3×3 convs have same receptive field as one 7×7

But: Fewer parameters + more non-linearities
```

VGG-16 structure:
```
2 × [Conv3-64] → Pool
2 × [Conv3-128] → Pool
3 × [Conv3-256] → Pool
3 × [Conv3-512] → Pool
3 × [Conv3-512] → Pool
FC → 4096 → 4096 → 1000
```

~138M parameters (most in FC layers)

### GoogLeNet/Inception (2014)

**Inception module**: Multiple filter sizes in parallel

```
        Input
          │
    ┌─────┼─────┐
    ↓     ↓     ↓
  1×1   3×3   5×5   Pool
    ↓     ↓     ↓     ↓
    └─────┴─────┴─────┘
          │
     Concatenate
```

Uses 1×1 convolutions to reduce depth before expensive operations.

~4M parameters (very efficient!)

### ResNet (2015)

Key innovation: **Residual connections** (skip connections)

```
    Input x
       │
    ┌──┴──┐
    │     │
  Conv    │
    │     │
  ReLU    │
    │     │
  Conv    │
    │     │
    └──┬──┘
       + ←── Skip connection adds input
       │
     ReLU
       │
    Output
```

**Why it works**:
- Easier to learn identity mapping (just set weights to 0)
- Gradients flow directly through skip connections
- Enables training of very deep networks (100+ layers)

```python
def residual_block(x, out_channels):
    identity = x

    out = conv(x, out_channels)
    out = batch_norm(out)
    out = relu(out)
    out = conv(out, out_channels)
    out = batch_norm(out)

    out = out + identity  # Skip connection
    out = relu(out)

    return out
```

ResNet variants: ResNet-18, 34, 50, 101, 152

---

## 3.6 Modern Practices

### Batch Normalization in CNNs

Applied after convolution, before activation:

```
Conv → BatchNorm → ReLU
```

Normalizes across batch and spatial dimensions:
```python
# For input shape [N, C, H, W]
# Normalize over N, H, W for each channel C
mean = x.mean(axis=(0, 2, 3))  # [C]
var = x.var(axis=(0, 2, 3))    # [C]
```

### 1×1 Convolutions

**Not** pointless! 1×1 convolutions:
- Change depth (channel compression/expansion)
- Add non-linearity
- Mix information across channels

```
Input: [H, W, 256]
1×1 Conv with 64 filters
Output: [H, W, 64]   (reduced depth)
```

### Global Average Pooling

Replace FC layers with global average pooling:

```
Traditional:          Modern:
7×7×512               7×7×512
   ↓                     ↓
Flatten             GlobalAvgPool
25,088                  512
   ↓                     ↓
FC 4096               FC 1000
   ↓
FC 1000
```

Fewer parameters, less overfitting.

---

## 3.7 Transfer Learning

### The Idea

Pre-trained CNN features are **universal**:
- Early layers: edges, colors (useful for any image task)
- Later layers: more task-specific

### Approaches

**1. Feature Extraction** (small dataset):
```python
# Freeze pre-trained weights
model = load_pretrained_resnet()
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(512, num_my_classes)

# Train only new layer
train(model)
```

**2. Fine-tuning** (medium dataset):
```python
# Start with pre-trained weights
model = load_pretrained_resnet()

# Replace final layer
model.fc = nn.Linear(512, num_my_classes)

# Train entire network with small learning rate
optimizer = Adam(model.parameters(), lr=1e-5)
train(model)
```

**3. Fine-tune later layers** (medium dataset):
```python
# Freeze early layers
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Train later layers + new head
# ...
```

### When to Use What

| Your Data | Strategy |
|-----------|----------|
| Small, similar to ImageNet | Feature extraction |
| Small, different from ImageNet | Feature extraction + data augmentation |
| Medium, similar | Fine-tune all layers |
| Medium, different | Fine-tune later layers |
| Large | Train from scratch or fine-tune |

---

## 3.8 Understanding CNNs

### Visualizing Filters

First layer filters are interpretable (edge detectors):

```
Learned filters often look like:
┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
│ - │ 0 │ + │  │ + │ + │ + │  │ + │ 0 │ - │
│ - │ 0 │ + │  │ 0 │ 0 │ 0 │  │ + │ 0 │ - │
│ - │ 0 │ + │  │ - │ - │ - │  │ + │ 0 │ - │
└───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘
Vertical edge  Horizontal     Diagonal
```

### Feature Map Visualization

Show which regions activate a filter:
- High activation = feature present
- Creates "heat maps" over image

### Receptive Field

The region of input that affects a single output neuron.

```
After 1 conv (3×3): receptive field = 3×3
After 2 convs (3×3): receptive field = 5×5
After 3 convs (3×3): receptive field = 7×7
```

Stack small filters to get large receptive fields efficiently.

---

## 3.9 Summary

### Key Concepts

1. **Convolution** applies learnable filters to detect local features
2. **Parameter sharing** makes CNNs efficient and translation equivariant
3. **Pooling** reduces spatial dimensions and adds invariance
4. **Architecture patterns**: CONV-RELU-POOL blocks, increasing depth
5. **ResNet's skip connections** enable very deep networks
6. **Transfer learning** leverages pre-trained features

### Glossary Terms Covered

- Convolutional Neural Network (CNN)
- Convolution
- Filter (Kernel)
- Feature Map
- Pooling
- Stride
- Padding
- Residual Connection (Skip Connection)
- ResNet

### What's Next

Module 4 covers **Sequence Models** (RNNs, LSTMs) for processing sequential data like text and time series.

---

## Exercises

1. **Output size**: Input 64×64×3, Conv 5×5×32, stride 1, padding "same". What's the output shape?

2. **Parameters**: How many parameters in a Conv layer with 128 filters of 3×3 on input depth 64?

3. **Receptive field**: After 4 layers of 3×3 convolutions (stride 1), what's the receptive field?

4. **Code**: Implement a simple CNN in PyTorch for CIFAR-10.

---

## References

- CS231n: Convolutional Neural Networks
- He et al., "Deep Residual Learning for Image Recognition"
- Simonyan & Zisserman, "Very Deep Convolutional Networks" (VGGNet)
