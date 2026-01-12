# Module 3: Convolutional Neural Networks

## Learning Objectives

By the end of this module, you will understand:
- Why CNNs are specifically designed for image data
- The convolution operation and its mathematical properties
- Pooling layers for spatial reduction and invariance
- Classic CNN architectures from LeNet to modern designs
- Transfer learning with pre-trained models
- Visualization and interpretation techniques

---

## 3.1 Motivation: Why Not Fully-Connected Networks?

### The Problem with FC Layers for Images

Consider processing a modest 200×200 RGB image with a fully-connected layer:

```
Input Image: 200×200×3 pixels = 120,000 values

FC Layer with 1000 hidden neurons:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input (120,000 values)                                         │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───────────────┬─┐             │
│  │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │   ...120,000   │ │             │
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───────────────┴─┘             │
│    │ │ │ │ │ │ │ │ │ │ │ │ │ │ │                 │              │
│    │ │ │ │ │ │ │ │ │ │ │ │ │ │ │   Every input   │              │
│    │ │ │ │ │ │ │ │ │ │ │ │ │ │ │   connects to   │              │
│    │ │ │ │ │ │ │ │ │ │ │ │ │ │ │   every output  │              │
│    ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓                 ↓              │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─────────────────┬─┐                 │
│  │ │ │ │ │ │ │ │ │ │ │ │ │    ...1000      │ │                 │
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─────────────────┴─┘                 │
│  Hidden layer (1000 neurons)                                    │
│                                                                 │
│  Parameters: 120,000 × 1000 = 120,000,000 weights               │
│            + 1,000 biases = 120,001,000 total                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Problems with this approach:**

1. **Too Many Parameters**
   - 120 million parameters for ONE layer
   - Memory: 120M × 4 bytes = 480 MB just for weights
   - Massive overfitting risk with limited training data
   - ImageNet has ~1.2M images; model has 120M parameters in first layer alone

2. **No Spatial Structure Understanding**
   ```
   FC layer treats these the same:

   Adjacent pixels:           Distant pixels:
   ┌───┬───┐                  ┌───┐     ┌───┐
   │ A │ B │ ← Connected      │ A │ ... │ B │ ← Also connected
   └───┴───┘    equally       └───┘     └───┘    equally!

   But adjacent pixels are semantically related!
   A cat's eye pixel relates more to nearby eye pixels
   than to pixels from the background.
   ```

3. **No Translation Invariance**
   ```
   Same cat, different positions → completely different FC activations:

   ┌─────────────────┐    ┌─────────────────┐
   │  ╭───╮          │    │          ╭───╮  │
   │  │cat│          │    │          │cat│  │
   │  ╰───╯          │    │          ╰───╯  │
   │                 │    │                 │
   └─────────────────┘    └─────────────────┘

   FC input vector:        FC input vector:
   [cat, 0, 0, 0, ...]     [0, 0, 0, ..., cat]

   These look completely different to FC layer!
   Model must learn "cat in position 1", "cat in position 2", etc.
   ```

### The CNN Solution: Three Key Ideas

**1. Local Connectivity (Sparse Connections)**
```
Instead of connecting to ALL inputs:

FC Layer:                        Conv Layer:
┌─────────────────────┐          ┌─────────────────────┐
│ [all 120,000 pixels]│          │ [3×3 local region]  │
│         │           │          │       │             │
│         ↓           │          │       ↓             │
│    [neuron]         │          │  [neuron]           │
│                     │          │                     │
│ Parameters: 120,000 │          │ Parameters: 9       │
└─────────────────────┘          └─────────────────────┘
```

**2. Parameter Sharing (Same Filter Everywhere)**
```
The same 3×3 filter slides across the entire image:

Image:
┌─────────────────────────────────────┐
│  ┌─────┐                            │
│  │ 3×3 │ Same 9 parameters detect   │
│  │filter│ the same feature          │
│  └──┬──┘ everywhere in the image    │
│     │                               │
│     │    ┌─────┐                    │
│     └───→│ 3×3 │                    │
│          │filter│                   │
│          └──┬──┘                    │
│             │    ┌─────┐            │
│             └───→│ 3×3 │            │
│                  │filter│           │
│                  └─────┘            │
└─────────────────────────────────────┘

One filter: 9 parameters can process ANY size image!
```

**3. Translation Equivariance**
```
If input shifts, output shifts correspondingly:

Input shifts right → Feature map shifts right

┌─────────┐         ┌─────────┐
│ ▓▓▓     │  edge   │   ▓     │
│ ▓▓▓     │ detect  │   ▓     │
│ ▓▓▓     │ ──────→ │   ▓     │
│         │         │         │
└─────────┘         └─────────┘

┌─────────┐         ┌─────────┐
│   ▓▓▓   │  same   │     ▓   │
│   ▓▓▓   │ filter  │     ▓   │
│   ▓▓▓   │ ──────→ │     ▓   │
│         │         │         │
└─────────┘         └─────────┘

The feature is detected regardless of position!
```

### Mathematical Comparison

| Property | FC Layer | Conv Layer |
|----------|----------|------------|
| Parameters for 200×200×3 input | 120M per neuron | 27 per 3×3×3 filter |
| Spatial awareness | None | Explicit |
| Translation | Must relearn | Equivariant |
| Input size | Fixed | Flexible |

---

## 3.2 The Convolution Operation

### Mathematical Definition

Convolution is a mathematical operation that combines two functions:

```
Continuous: (f * g)(t) = ∫ f(τ)g(t - τ)dτ

Discrete 2D (for images):
                M    N
(I * K)[i,j] = Σ    Σ   I[i+m, j+n] × K[m, n]
               m=-M n=-N

Where:
  I = Input image (2D matrix)
  K = Kernel/Filter (small 2D matrix)
  * = Convolution operator
```

### Visual Step-by-Step

Let's convolve a 5×5 input with a 3×3 filter:

```
Step 1: Position filter at top-left
┌─────────────────────┐        ┌───────────┐
│[1] [2] [3]  4   5   │        │ 1   0  -1 │
│[6] [7] [8]  9   10  │   *    │ 1   0  -1 │
│[11][12][13] 14  15  │        │ 1   0  -1 │
│ 16  17  18  19  20  │        └───────────┘
│ 21  22  23  24  25  │
└─────────────────────┘

Calculation:
(1×1) + (2×0) + (3×-1) +
(6×1) + (7×0) + (8×-1) +
(11×1) + (12×0) + (13×-1)
= 1 + 0 - 3 + 6 + 0 - 8 + 11 + 0 - 13
= -6

Output[0,0] = -6
```

```
Step 2: Slide filter right by 1 (stride=1)
┌─────────────────────┐
│ 1  [2] [3] [4]  5   │
│ 6  [7] [8] [9]  10  │
│ 11 [12][13][14] 15  │
│ 16  17  18  19  20  │
│ 21  22  23  24  25  │
└─────────────────────┘

Calculation:
(2×1) + (3×0) + (4×-1) +
(7×1) + (8×0) + (9×-1) +
(12×1) + (13×0) + (14×-1)
= 2 - 4 + 7 - 9 + 12 - 14
= -6

Output[0,1] = -6
```

```
Complete convolution produces 3×3 output:
┌─────────────────────┐
│  5×5 Input          │        ┌───────────┐
│                     │   *    │ 3×3 Filter│
│                     │        └───────────┘
│                     │              ↓
│                     │        ┌───────────┐
└─────────────────────┘        │-6 -6 -6   │
                               │-6 -6 -6   │
                               │-6 -6 -6   │
                               └───────────┘
                                3×3 Output
```

### Filters as Feature Detectors

Different filter values detect different image features:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        EDGE DETECTION FILTERS                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Vertical Edge          Horizontal Edge        Diagonal Edge             │
│  ┌─────────────┐        ┌─────────────┐       ┌─────────────┐           │
│  │ -1   0   1  │        │ -1  -1  -1  │       │  0   1   1  │           │
│  │ -1   0   1  │        │  0   0   0  │       │ -1   0   1  │           │
│  │ -1   0   1  │        │  1   1   1  │       │ -1  -1   0  │           │
│  └─────────────┘        └─────────────┘       └─────────────┘           │
│                                                                          │
│  Detects:               Detects:              Detects:                   │
│  │                      ───                   ╱                          │
│  │                      ───                    ╱                         │
│  │                      ───                     ╱                        │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                         SOBEL OPERATORS                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sobel X (vertical)     Sobel Y (horizontal)                             │
│  ┌─────────────┐        ┌─────────────┐                                 │
│  │ -1   0   1  │        │ -1  -2  -1  │                                 │
│  │ -2   0   2  │        │  0   0   0  │                                 │
│  │ -1   0   1  │        │  1   2   1  │                                 │
│  └─────────────┘        └─────────────┘                                 │
│                                                                          │
│  Gradient magnitude = √(Gx² + Gy²)                                      │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                      OTHER USEFUL FILTERS                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Box Blur (3×3)         Gaussian Blur         Sharpen                    │
│  ┌─────────────┐        ┌─────────────┐       ┌─────────────┐           │
│  │ 1/9 1/9 1/9 │        │ 1  2  1     │       │  0  -1   0  │           │
│  │ 1/9 1/9 1/9 │        │ 2  4  2     │ ÷16   │ -1   5  -1  │           │
│  │ 1/9 1/9 1/9 │        │ 1  2  1     │       │  0  -1   0  │           │
│  └─────────────┘        └─────────────┘       └─────────────┘           │
│                                                                          │
│  Laplacian (edge)       Emboss                Identity                   │
│  ┌─────────────┐        ┌─────────────┐       ┌─────────────┐           │
│  │  0  -1   0  │        │ -2  -1   0  │       │  0   0   0  │           │
│  │ -1   4  -1  │        │ -1   1   1  │       │  0   1   0  │           │
│  │  0  -1   0  │        │  0   1   2  │       │  0   0   0  │           │
│  └─────────────┘        └─────────────┘       └─────────────┘           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key insight: In CNNs, filter values are LEARNED from data, not hand-designed!**

### Convolution Hyperparameters

#### Filter Size (F)

```
Common filter sizes and their properties:

1×1 Filter:
┌───┐
│ w │  - Changes depth only (channel mixing)
└───┘  - No spatial feature detection
       - Used for dimensionality reduction

3×3 Filter:                    5×5 Filter:
┌───┬───┬───┐                  ┌───┬───┬───┬───┬───┐
│   │   │   │                  │   │   │   │   │   │
├───┼───┼───┤ Most common!     ├───┼───┼───┼───┼───┤
│   │ C │   │ Good balance     │   │   │   │   │   │
├───┼───┼───┤ of receptive     ├───┼───┼───┼───┼───┤
│   │   │   │ field & params   │   │   │ C │   │   │
└───┴───┴───┘                  ├───┼───┼───┼───┼───┤
9 parameters                   │   │   │   │   │   │
                               ├───┼───┼───┼───┼───┤
                               │   │   │   │   │   │
                               └───┴───┴───┴───┴───┘
                               25 parameters

7×7 Filter:
Only used in first layer of some architectures (AlexNet, ResNet)
49 parameters - larger receptive field, more computation

Why odd sizes? To have a center pixel for symmetry
```

#### Stride (S)

```
Stride = 1 (default):                    Stride = 2:
Filter moves 1 pixel at a time           Filter moves 2 pixels at a time

Step 0    Step 1    Step 2               Step 0    Step 1
┌─────┐─────────────────┐               ┌─────┐───────────┐
│[▓▓▓]│ ▓ ▓ ▓ ▓ ▓      │               │[▓▓▓]│ ▓ ▓ ▓ ▓  │
│[▓▓▓]│ ▓ ▓ ▓ ▓ ▓      │               │[▓▓▓]│ ▓ ▓ ▓ ▓  │
│[▓▓▓]│ ▓ ▓ ▓ ▓ ▓      │               │[▓▓▓]│ ▓ ▓ ▓ ▓  │
│  ▓  │ ▓ ▓ ▓ ▓ ▓      │               │  ▓  │ ▓ ▓ ▓ ▓  │
│  ▓  │ ▓ ▓ ▓ ▓ ▓      │               │  ▓  │ ▓ ▓ ▓ ▓  │
└─────────────────────-┘               └──────────────────┘

┌─────────────────────┐                 ┌───────────────────┐
│  ▓ │[▓▓▓]│ ▓ ▓ ▓    │                 │  ▓   ▓ │[▓▓▓]│ ▓  │
│  ▓ │[▓▓▓]│ ▓ ▓ ▓    │                 │  ▓   ▓ │[▓▓▓]│ ▓  │
│  ▓ │[▓▓▓]│ ▓ ▓ ▓    │                 │  ▓   ▓ │[▓▓▓]│ ▓  │
│  ▓   ▓   ▓ ▓ ▓ ▓    │                 │  ▓   ▓   ▓   ▓ ▓  │
│  ▓   ▓   ▓ ▓ ▓ ▓    │                 │  ▓   ▓   ▓   ▓ ▓  │
└─────────────────────┘                 └───────────────────┘

Output size: 5 positions                 Output size: 2 positions
Stride 2 reduces output by ~half
```

#### Padding (P)

```
No Padding (Valid):
┌─────────────┐
│ ▓ ▓ ▓ ▓ ▓   │    5×5 input
│ ▓ ▓ ▓ ▓ ▓   │    3×3 filter
│ ▓ ▓ ▓ ▓ ▓   │    ───────────
│ ▓ ▓ ▓ ▓ ▓   │    3×3 output
│ ▓ ▓ ▓ ▓ ▓   │
└─────────────┘    Output shrinks!

With Padding=1 (Same):
┌───────────────────┐
│ 0 0 0 0 0 0 0     │    5×5 input + 1 padding
│ 0 ▓ ▓ ▓ ▓ ▓ 0     │    3×3 filter
│ 0 ▓ ▓ ▓ ▓ ▓ 0     │    ───────────
│ 0 ▓ ▓ ▓ ▓ ▓ 0     │    5×5 output
│ 0 ▓ ▓ ▓ ▓ ▓ 0     │
│ 0 ▓ ▓ ▓ ▓ ▓ 0     │    Same size!
│ 0 0 0 0 0 0 0     │
└───────────────────┘

Zero padding: Most common, adds zeros around border
Reflect padding: Mirrors edge pixels (reduces edge artifacts)
Replicate padding: Copies edge pixels
```

### Output Size Formula

```
Output Size = floor((W - F + 2P) / S) + 1

Where:
  W = Input width (or height)
  F = Filter size
  P = Padding
  S = Stride

Examples:
─────────────────────────────────────────────────────────────────
Input   Filter  Stride  Padding    Output
─────────────────────────────────────────────────────────────────
32×32   3×3     1       0          (32-3+0)/1+1 = 30×30
32×32   3×3     1       1          (32-3+2)/1+1 = 32×32  (same)
32×32   5×5     1       2          (32-5+4)/1+1 = 32×32  (same)
32×32   3×3     2       0          (32-3+0)/2+1 = 15×15  (downsample)
224×224 7×7     2       3          (224-7+6)/2+1 = 112×112
─────────────────────────────────────────────────────────────────

For "same" padding (output = input):
P = (F - 1) / 2  (only works for stride=1 and odd F)
```

### Multi-Channel Convolution (Color Images)

For RGB images, filters must have depth 3:

```
Input Volume: [H × W × 3]     Filter: [F × F × 3]     Output: [H' × W' × 1]

       R Channel    G Channel    B Channel
       ┌─────────┐ ┌─────────┐ ┌─────────┐
       │         │ │         │ │         │
       │         │ │         │ │         │
       │         │ │         │ │         │
       └─────────┘ └─────────┘ └─────────┘
            ×           ×           ×
       ┌─────┐     ┌─────┐     ┌─────┐
       │ R   │     │ G   │     │ B   │
       │filter│    │filter│    │filter│
       └─────┘     └─────┘     └─────┘
            ↓           ↓           ↓
            └───────────┼───────────┘
                        ↓
                     [SUM]
                        ↓
                  Single value

One 3×3×3 filter has 3×3×3 = 27 weights + 1 bias = 28 parameters
```

### Multiple Filters = Multiple Feature Maps

```
K filters produce K feature maps (output channels):

Input:           K Filters:              Output:
[H × W × D]  ×  [K × F × F × D]  =   [H' × W' × K]

┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│             │  │ Filter 1 (edge)│  │ ┌─────────────┐     │
│             │  │ Filter 2 (blob)│  │ │ Feature Map 1│     │
│    Input    │  │ Filter 3 (tex) │  │ │ Feature Map 2│     │
│  32×32×3    │  │     ...        │  │ │ Feature Map 3│     │
│             │  │ Filter K       │  │ │     ...      │     │
│             │  └─────────────────┘  │ │ Feature Map K│     │
└─────────────┘        64 filters    │ └─────────────┘     │
                         3×3         │    32×32×64         │
                                     └─────────────────────┘
```

### Parameter Count

```
Conv Layer Parameters:

Parameters = K × (F × F × D + 1)
             ↑        ↑      ↑
        # filters  filter   bias
                    size

Example 1: First conv layer
─────────────────────────────
Input: 224×224×3 (RGB image)
Filters: 64 of size 3×3
Parameters: 64 × (3×3×3 + 1) = 64 × 28 = 1,792

Example 2: Middle conv layer
─────────────────────────────
Input: 28×28×256
Filters: 512 of size 3×3
Parameters: 512 × (3×3×256 + 1) = 512 × 2305 = 1,180,160

Comparison with FC:
─────────────────────────────
Same input 28×28×256 = 200,704 values
FC layer with 512 neurons: 200,704 × 512 = 102,760,448 parameters
Conv layer with 512 3×3 filters: 1,180,160 parameters

Conv is 87× more efficient!
```

### PyTorch Convolution Implementation

```python
import torch
import torch.nn as nn

# Define a convolutional layer
conv = nn.Conv2d(
    in_channels=3,      # Input depth (e.g., RGB)
    out_channels=64,    # Number of filters
    kernel_size=3,      # Filter size (3×3)
    stride=1,           # Step size
    padding=1,          # Zero padding
    bias=True           # Include bias term
)

# Input shape: [batch_size, channels, height, width]
x = torch.randn(32, 3, 224, 224)  # Batch of 32 RGB 224×224 images

# Forward pass
y = conv(x)
print(y.shape)  # torch.Size([32, 64, 224, 224])

# Examine parameters
print(conv.weight.shape)  # torch.Size([64, 3, 3, 3]) - 64 filters, 3×3×3 each
print(conv.bias.shape)    # torch.Size([64]) - one bias per filter
print(f"Total parameters: {conv.weight.numel() + conv.bias.numel()}")  # 1792
```

---

## 3.3 Pooling Layers

### Purpose of Pooling

Pooling provides three key benefits:

```
1. SPATIAL DIMENSION REDUCTION
   ─────────────────────────────
   Input: 224×224×64  →  Pool 2×2  →  Output: 112×112×64

   Reduces computation in subsequent layers by 4×

2. TRANSLATION INVARIANCE
   ─────────────────────────────
   ┌───────────────┐         ┌───────────────┐
   │   ┌─┐         │         │     ┌─┐       │
   │   │◯│         │         │     │◯│       │
   │   └─┘         │         │     └─┘       │
   │               │         │               │
   └───────────────┘         └───────────────┘
   Eye slightly left         Eye slightly right

   After pooling, both produce similar output
   → Small shifts don't change the representation

3. OVERFITTING CONTROL
   ─────────────────────────────
   Fewer spatial positions = fewer parameters in FC layers
   Also acts as regularization (discards spatial precision)
```

### Max Pooling

The most common pooling operation - takes maximum value in each window:

```
Max Pooling 2×2 with Stride 2:

Input (4×4):                    Output (2×2):
┌─────┬─────┬─────┬─────┐       ┌─────┬─────┐
│  1  │  3  │  2  │  1  │       │     │     │
├─────┼─────┼─────┼─────┤       │  4  │  3  │
│  4  │  2  │  1  │  3  │       │     │     │
├═════╪═════╪═════╪═════┤       ├─────┼─────┤
│  7  │  1  │  3  │  2  │       │     │     │
├─────┼─────┼─────┼─────┤       │  7  │  4  │
│  2  │  5  │  4  │  1  │       │     │     │
└─────┴─────┴─────┴─────┘       └─────┴─────┘

Top-left region:  max(1,3,4,2) = 4
Top-right region: max(2,1,1,3) = 3
Bot-left region:  max(7,1,2,5) = 7
Bot-right region: max(3,2,4,1) = 4
```

```
Properties of Max Pooling:
────────────────────────────────────────────────────────
✓ NO learnable parameters
✓ Preserves strongest activations (features)
✓ Provides local translation invariance
✓ Halves spatial dimensions (with 2×2, stride 2)
✗ Discards 75% of activations (3 of 4 values)
✗ Loses precise spatial information
```

### Average Pooling

Takes average value instead of maximum:

```
Average Pooling 2×2 with Stride 2:

Input (4×4):                    Output (2×2):
┌─────┬─────┬─────┬─────┐       ┌─────┬─────┐
│  1  │  3  │  2  │  2  │       │     │     │
├─────┼─────┼─────┼─────┤       │ 2.5 │ 2.0 │
│  4  │  2  │  2  │  2  │       │     │     │
├═════╪═════╪═════╪═════┤       ├─────┼─────┤
│  8  │  0  │  4  │  0  │       │     │     │
├─────┼─────┼─────┼─────┤       │ 3.0 │ 2.0 │
│  0  │  4  │  0  │  4  │       │     │     │
└─────┴─────┴─────┴─────┘       └─────┴─────┘

Top-left: (1+3+4+2)/4 = 2.5
Top-right: (2+2+2+2)/4 = 2.0
Bot-left: (8+0+0+4)/4 = 3.0
Bot-right: (4+0+0+4)/4 = 2.0
```

### Global Average Pooling (GAP)

Averages entire feature map to single value:

```
Global Average Pooling:

Input: [7 × 7 × 512]          Output: [1 × 1 × 512]
┌─────────────────┐           ┌─────┐
│ Feature Map 1   │──average──│ v₁  │
│ (7×7 values)    │           │     │
├─────────────────┤           ├─────┤
│ Feature Map 2   │──average──│ v₂  │
│ (7×7 values)    │           │     │
├─────────────────┤           ├─────┤
│      ...        │           │ ... │
├─────────────────┤           ├─────┤
│ Feature Map 512 │──average──│v₅₁₂ │
│ (7×7 values)    │           │     │
└─────────────────┘           └─────┘

Each 7×7 feature map → single average value
Output: 512-dimensional vector (one value per channel)

Benefits:
✓ Drastically reduces parameters before FC
✓ No spatial parameters to learn
✓ More robust to input size variations
```

### Pooling Summary

```
                    Max Pool    Avg Pool    Global Avg Pool
────────────────────────────────────────────────────────────
Operation           Maximum     Average     Avg over entire map
Typical size        2×2         2×2         H×W (adaptive)
Stride              2           2           N/A
Parameters          0           0           0
Output size         H/2, W/2    H/2, W/2    1×1
Main use            General     Sometimes   Final before FC
────────────────────────────────────────────────────────────
```

### PyTorch Pooling Layers

```python
import torch.nn as nn

# Max pooling
max_pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

# Average pooling
avg_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)

# Global average pooling (adaptive to any input size)
gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

# Usage
x = torch.randn(32, 512, 14, 14)
print(max_pool(x).shape)   # torch.Size([32, 512, 7, 7])
print(avg_pool(x).shape)   # torch.Size([32, 512, 7, 7])
print(gap(x).shape)        # torch.Size([32, 512, 1, 1])
```

---

## 3.4 CNN Architecture Patterns

### The Basic CNN Pattern

```
Classic CNN Architecture:

Input Image
    ↓
┌───────────────────────────────────────────────────────────────┐
│                     FEATURE EXTRACTION                         │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  CONV BLOCK 1: [CONV → BN → ReLU] × N → MaxPool        │  │
│  │  Learns: Edges, colors, simple textures                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  CONV BLOCK 2: [CONV → BN → ReLU] × N → MaxPool        │  │
│  │  Learns: Textures, patterns, simple shapes              │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  CONV BLOCK 3: [CONV → BN → ReLU] × N → MaxPool        │  │
│  │  Learns: Parts, object components                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  CONV BLOCK 4-5: More layers for complex features       │  │
│  │  Learns: Objects, scenes, semantic concepts             │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────┐
│                      CLASSIFICATION HEAD                       │
│                                                                │
│   Flatten or Global Average Pool                               │
│              ↓                                                 │
│   FC → ReLU → Dropout → FC → Softmax                          │
│                                                                │
└───────────────────────────────────────────────────────────────┘
    ↓
Output: Class Probabilities
```

### Layer Progression Pattern

As we go deeper in the network:

```
Layer Depth →

Spatial Size:     DECREASES ↘
                  224 → 112 → 56 → 28 → 14 → 7 → 1

Channel Count:    INCREASES ↗
                  3 → 64 → 128 → 256 → 512 → 512

Feature Abstraction: INCREASES ↗
                  pixels → edges → textures → parts → objects → concepts

Receptive Field:  INCREASES ↗
                  3×3 → 7×7 → 15×15 → 31×31 → ...

Visual representation:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   [224×224×3] → [112×112×64] → [56×56×128] → [28×28×256]          │
│                                                                     │
│   ████████████    ████████      ████        ██                     │
│   ████████████    ████████      ████        ██                     │
│   ████████████    ████████      ████                               │
│   ████████████    ████████                                         │
│   ████████████                                                     │
│   (tall & thin)   (shorter,    (even        (short,               │
│                    wider)       shorter)     wide)                 │
│                                                                     │
│   → [14×14×512] → [7×7×512] → [1×1×512] → [1000]                   │
│                                                                     │
│   █              █              █           ████...                 │
│                                                                     │
│   (compact but deep channels)              (class scores)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What Each Layer Learns

```
HIERARCHICAL FEATURE LEARNING
═══════════════════════════════

Layer 1: Edge and Color Detectors
┌──────────────────────────────────────────┐
│  │  │  ─  ─  ╱  ╲  ●  ●  [R] [G] [B]    │
│  Vertical, horizontal, diagonal edges    │
│  Color blobs, gradients                  │
└──────────────────────────────────────────┘
              ↓
Layer 2: Texture and Corner Detectors
┌──────────────────────────────────────────┐
│  ┼  ╬  ▓▓  ░░  ⌐  ⌐  ╭╮  ╰╯            │
│  Corners, crosses, simple textures       │
│  Combinations of edges                   │
└──────────────────────────────────────────┘
              ↓
Layer 3: Part Detectors
┌──────────────────────────────────────────┐
│  👁  👄  🦶  ⚙  🔵  📐                  │
│  Eyes, wheels, geometric shapes          │
│  Object parts, repeated patterns         │
└──────────────────────────────────────────┘
              ↓
Layer 4-5: Object/Scene Detectors
┌──────────────────────────────────────────┐
│  🐱  🚗  🏠  👤  🌳                      │
│  Full objects, faces, animals            │
│  Scene categories, complex concepts      │
└──────────────────────────────────────────┘
```

---

## 3.5 Classic CNN Architectures

### LeNet-5 (1998) - The Pioneer

```
LeNet-5: Digit Recognition Pioneer
Designed by Yann LeCun for handwritten digit recognition

┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  INPUT: 32×32×1 (grayscale digit image)                           │
│  │                                                                 │
│  ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ CONV1: 6 filters, 5×5, stride 1                         │      │
│  │ Output: 28×28×6                                          │      │
│  │ Params: 6 × (5×5×1 + 1) = 156                           │      │
│  └─────────────────────────────────────────────────────────┘      │
│  │                                                                 │
│  ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ POOL1: 2×2 average pooling, stride 2                    │      │
│  │ Output: 14×14×6                                          │      │
│  └─────────────────────────────────────────────────────────┘      │
│  │                                                                 │
│  ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ CONV2: 16 filters, 5×5, stride 1                        │      │
│  │ Output: 10×10×16                                         │      │
│  │ Params: 16 × (5×5×6 + 1) = 2,416                        │      │
│  └─────────────────────────────────────────────────────────┘      │
│  │                                                                 │
│  ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ POOL2: 2×2 average pooling, stride 2                    │      │
│  │ Output: 5×5×16                                           │      │
│  └─────────────────────────────────────────────────────────┘      │
│  │                                                                 │
│  ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ FC1: 120 neurons                                         │      │
│  │ Params: 5×5×16 × 120 + 120 = 48,120                     │      │
│  └─────────────────────────────────────────────────────────┘      │
│  │                                                                 │
│  ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ FC2: 84 neurons                                          │      │
│  │ Params: 120 × 84 + 84 = 10,164                          │      │
│  └─────────────────────────────────────────────────────────┘      │
│  │                                                                 │
│  ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ OUTPUT: 10 neurons (digits 0-9)                         │      │
│  │ Params: 84 × 10 + 10 = 850                              │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                    │
│  Total Parameters: ~61,000                                         │
│  Innovations: Local connectivity, weight sharing, pooling          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### AlexNet (2012) - The Deep Learning Revolution

```
AlexNet: ImageNet 2012 Winner
The network that launched the deep learning revolution

┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  INPUT: 227×227×3 (RGB ImageNet image)                                │
│  │                                                                     │
│  ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ CONV1: 96 filters, 11×11, stride 4                            │    │
│  │ Output: 55×55×96                                               │    │
│  │ + ReLU + Local Response Norm + Max Pool 3×3 stride 2          │    │
│  │ After pool: 27×27×96                                           │    │
│  └───────────────────────────────────────────────────────────────┘    │
│  │                                                                     │
│  ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ CONV2: 256 filters, 5×5, padding 2                            │    │
│  │ Output: 27×27×256                                              │    │
│  │ + ReLU + LRN + Max Pool 3×3 stride 2                          │    │
│  │ After pool: 13×13×256                                          │    │
│  └───────────────────────────────────────────────────────────────┘    │
│  │                                                                     │
│  ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ CONV3: 384 filters, 3×3, padding 1                            │    │
│  │ Output: 13×13×384 + ReLU                                       │    │
│  └───────────────────────────────────────────────────────────────┘    │
│  │                                                                     │
│  ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ CONV4: 384 filters, 3×3, padding 1                            │    │
│  │ Output: 13×13×384 + ReLU                                       │    │
│  └───────────────────────────────────────────────────────────────┘    │
│  │                                                                     │
│  ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ CONV5: 256 filters, 3×3, padding 1                            │    │
│  │ Output: 13×13×256 + ReLU + Max Pool 3×3 stride 2              │    │
│  │ After pool: 6×6×256                                            │    │
│  └───────────────────────────────────────────────────────────────┘    │
│  │                                                                     │
│  ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ FC6: 4096 neurons + ReLU + Dropout(0.5)                       │    │
│  │ FC7: 4096 neurons + ReLU + Dropout(0.5)                       │    │
│  │ FC8: 1000 neurons (ImageNet classes) + Softmax                │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  Total Parameters: ~62 million                                         │
│                                                                        │
│  Key Innovations:                                                      │
│  ✓ ReLU activation (faster training than sigmoid/tanh)                │
│  ✓ Dropout (regularization, prevents overfitting)                     │
│  ✓ Data augmentation (translations, reflections, color jitter)        │
│  ✓ GPU training (split across 2 GPUs)                                 │
│  ✓ Large scale (60M params, trained on 1.2M images)                   │
│                                                                        │
│  ImageNet Results:                                                     │
│  Top-5 error: 15.3% (previous best: 26.2%) — 10.9% improvement!       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### VGGNet (2014) - Simplicity and Depth

```
VGGNet: Small Filters, Deep Networks
Key insight: Use 3×3 filters everywhere

┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  WHY 3×3 FILTERS?                                                     │
│  ════════════════                                                     │
│                                                                        │
│  Two 3×3 convs = one 5×5 conv (same receptive field)                  │
│  ┌───┬───┬───┐   ┌───┬───┬───┐     ┌───┬───┬───┬───┬───┐             │
│  │ 3×3 conv │ + │ 3×3 conv │  =  │    5×5 conv    │             │
│  └───────────┘   └───────────┘     └─────────────────┘             │
│  18 params       18 params         25 params                          │
│  2 non-linearities                 1 non-linearity                    │
│                                                                        │
│  Three 3×3 convs = one 7×7 conv                                       │
│  27 params + 3 ReLUs  vs  49 params + 1 ReLU                          │
│                                                                        │
│  Benefits:                                                             │
│  ✓ Fewer parameters (more efficient)                                  │
│  ✓ More non-linearities (more expressive)                             │
│  ✓ Easier to train                                                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

VGG-16 Architecture:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  INPUT: 224×224×3                                                      │
│  │                                                                     │
│  ├──► Block 1: 2×[Conv3-64] → MaxPool    →  112×112×64               │
│  │                                                                     │
│  ├──► Block 2: 2×[Conv3-128] → MaxPool   →  56×56×128                │
│  │                                                                     │
│  ├──► Block 3: 3×[Conv3-256] → MaxPool   →  28×28×256                │
│  │                                                                     │
│  ├──► Block 4: 3×[Conv3-512] → MaxPool   →  14×14×512                │
│  │                                                                     │
│  ├──► Block 5: 3×[Conv3-512] → MaxPool   →  7×7×512                  │
│  │                                                                     │
│  ├──► FC: 4096 → 4096 → 1000                                          │
│  │                                                                     │
│  Total: 138 million parameters                                         │
│  (Most in FC layers: 7×7×512 × 4096 = 102M)                           │
│                                                                        │
│  VGG-19: Same but blocks 3-5 have 4 convs instead of 3                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### GoogLeNet/Inception (2014) - Efficiency Through Parallelism

```
Inception Module: Multiple Filter Sizes in Parallel
Key insight: Let the network decide which filter size is best

┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│                        NAIVE INCEPTION MODULE                          │
│                                                                        │
│                              Input                                     │
│                                │                                       │
│            ┌───────┬───────────┼───────────┬───────┐                  │
│            │       │           │           │       │                  │
│            ▼       ▼           ▼           ▼       │                  │
│         ┌─────┐ ┌─────┐    ┌─────┐    ┌─────┐    │                  │
│         │ 1×1 │ │ 3×3 │    │ 5×5 │    │Pool │    │                  │
│         │conv │ │conv │    │conv │    │3×3  │    │                  │
│         └──┬──┘ └──┬──┘    └──┬──┘    └──┬──┘    │                  │
│            │       │           │           │       │                  │
│            └───────┴───────────┴───────────┘       │                  │
│                        │                           │                  │
│                   Concatenate                      │                  │
│                        │                           │                  │
│                      Output                        │                  │
│                                                                        │
│  Problem: 5×5 convs are expensive!                                    │
│  192 input channels × 32 5×5 filters = 153,600 parameters             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│                    INCEPTION MODULE WITH REDUCTION                     │
│                                                                        │
│                              Input                                     │
│                                │                                       │
│       ┌────────┬───────────────┼───────────────┬────────┐             │
│       │        │               │               │        │             │
│       ▼        ▼               ▼               ▼        │             │
│    ┌─────┐  ┌─────┐         ┌─────┐        ┌─────┐     │             │
│    │ 1×1 │  │ 1×1 │         │ 1×1 │        │Pool │     │             │
│    │conv │  │conv │         │conv │        │3×3  │     │             │
│    └──┬──┘  └──┬──┘         └──┬──┘        └──┬──┘     │             │
│       │        │ ←reduce       │ ←reduce      │        │             │
│       │        ▼               ▼              ▼        │             │
│       │     ┌─────┐         ┌─────┐       ┌─────┐     │             │
│       │     │ 3×3 │         │ 5×5 │       │ 1×1 │     │             │
│       │     │conv │         │conv │       │conv │     │             │
│       │     └──┬──┘         └──┬──┘       └──┬──┘     │             │
│       │        │               │              │        │             │
│       └────────┴───────────────┴──────────────┘        │             │
│                        │                               │             │
│                   Concatenate                          │             │
│                        │                               │             │
│                      Output                            │             │
│                                                                        │
│  1×1 convolutions reduce depth BEFORE expensive operations            │
│  Much fewer parameters!                                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

GoogLeNet Full Architecture:
- 22 layers deep
- 9 Inception modules
- No FC layers (uses Global Average Pooling)
- Only ~5 million parameters (vs 60M for AlexNet, 138M for VGG)
- Uses auxiliary classifiers for training (inject gradients mid-network)
```

### ResNet (2015) - The Residual Revolution

```
THE DEGRADATION PROBLEM
═══════════════════════

Deeper networks should be better... but they're not!

                     Training Error vs Depth
    Error ▲
          │     20-layer         56-layer
          │      ┌────┐
          │      │    │          ┌────┐
          │      │    │          │    │
          │      │    │          │    │
          │      │    │          │    │
          │      └────┘          │    │
          │                      │    │
          │                      └────┘
          └──────────────────────────────────▶
                                              Depth

56-layer network has HIGHER training error than 20-layer!
This is NOT overfitting (training error is higher)

The problem: Very deep networks are hard to optimize
Deep networks should at least be able to copy shallower networks
(set extra layers to identity) but SGD can't find this solution
```

```
THE RESIDUAL SOLUTION
═════════════════════

Instead of learning H(x), learn F(x) = H(x) - x

    Traditional Block:              Residual Block:

         x                              x
         │                              │
         ▼                         ┌────┴────┐
    ┌─────────┐                    │         │
    │  Conv   │                    ▼         │
    ├─────────┤               ┌─────────┐    │
    │  ReLU   │               │  Conv   │    │
    ├─────────┤               ├─────────┤    │
    │  Conv   │               │  ReLU   │    │
    └────┬────┘               ├─────────┤    │
         │                    │  Conv   │    │
         ▼                    └────┬────┘    │
       H(x)                        │         │
                                   └────┬────┘
                                        │
                                      [+] ← element-wise addition
                                        │
                                        ▼
                                      ReLU
                                        │
                                        ▼
                                   H(x) = F(x) + x

    Learn: H(x)                  Learn: F(x) = H(x) - x
                                 Output: H(x) = F(x) + x
```

```
WHY RESIDUALS WORK
══════════════════

1. EASY TO LEARN IDENTITY
   ───────────────────────
   To make H(x) = x, just set F(x) = 0
   This means: set all weights in the residual branch to zero

   Traditional: Must learn W such that W·x = x (non-trivial!)
   Residual:    Just set W = 0, and x passes through skip connection

2. GRADIENT FLOW
   ───────────────────────
   Backpropagation through skip connection:

   ∂Loss/∂x = ∂Loss/∂H × (∂F/∂x + 1)
                          ↑       ↑
                       residual  skip (always 1!)
                       branch

   The "+1" from skip connection ensures gradients flow!
   Even if ∂F/∂x → 0, gradients can still propagate through skip

3. ENSEMBLE INTERPRETATION
   ───────────────────────
   ResNet can be viewed as ensemble of many paths:

   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │  Input ─┬─► Block1 ─┬─► Block2 ─┬─► Block3 ─┬─► Output │
   │         │           │           │           │          │
   │         └───────────┴───────────┴───────────┘          │
   │                                                         │
   │  n blocks = 2^n possible paths from input to output    │
   │  Network learns which paths are useful                  │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
```

```
RESNET ARCHITECTURES
════════════════════

Basic Block (ResNet-18, 34):           Bottleneck Block (ResNet-50+):
┌─────────────────────────┐            ┌─────────────────────────┐
│                         │            │                         │
│         x               │            │         x               │
│         │               │            │         │               │
│    ┌────┴────┐          │            │    ┌────┴────┐          │
│    │         │          │            │    │         │          │
│    ▼         │          │            │    ▼         │          │
│ ┌──────┐    │          │            │ ┌──────┐    │          │
│ │Conv3×3│    │          │            │ │Conv1×1│    │ ← reduce │
│ └──┬───┘    │          │            │ └──┬───┘    │   channels│
│    ▼         │          │            │    ▼         │          │
│ BatchNorm    │          │            │ BatchNorm    │          │
│    ▼         │          │            │    ▼         │          │
│   ReLU       │          │            │   ReLU       │          │
│    ▼         │          │            │    ▼         │          │
│ ┌──────┐    │          │            │ ┌──────┐    │          │
│ │Conv3×3│    │          │            │ │Conv3×3│    │ ← spatial│
│ └──┬───┘    │          │            │ └──┬───┘    │          │
│    ▼         │          │            │    ▼         │          │
│ BatchNorm    │          │            │ BatchNorm    │          │
│    │         │          │            │    ▼         │          │
│    └────┬────┘          │            │   ReLU       │          │
│         │               │            │    ▼         │          │
│        [+]              │            │ ┌──────┐    │          │
│         ▼               │            │ │Conv1×1│    │ ← expand │
│       ReLU              │            │ └──┬───┘    │   channels│
│         │               │            │    ▼         │          │
│       output            │            │ BatchNorm    │          │
│                         │            │    │         │          │
└─────────────────────────┘            │    └────┬────┘          │
                                       │         │               │
2 conv layers                          │        [+]              │
                                       │         ▼               │
                                       │       ReLU              │
                                       │         │               │
                                       │       output            │
                                       │                         │
                                       └─────────────────────────┘

                                       3 conv layers (1×1, 3×3, 1×1)
                                       More efficient for deep networks

ResNet Variants:
──────────────────────────────────────────────────────────────
Model        Layers    Params    Top-1 Error    Block Type
──────────────────────────────────────────────────────────────
ResNet-18    18        11.7M     30.2%          Basic
ResNet-34    34        21.8M     26.7%          Basic
ResNet-50    50        25.6M     24.0%          Bottleneck
ResNet-101   101       44.5M     22.4%          Bottleneck
ResNet-152   152       60.2M     21.7%          Bottleneck
──────────────────────────────────────────────────────────────
```

### PyTorch ResNet Implementation

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18/34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # For matching dimensions in skip connection

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample identity if dimensions don't match
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # SKIP CONNECTION - the key innovation!
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1×1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3×3 conv for spatial processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1×1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

### Architecture Comparison Summary

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE EVOLUTION SUMMARY                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Model       Year  Depth  Params   Top-5  Key Innovation                   │
│  ─────────────────────────────────────────────────────────────────────────│
│  LeNet-5     1998    5     60K      N/A   CNNs work!                       │
│  AlexNet     2012    8     60M    15.3%   ReLU, dropout, GPU, scale        │
│  VGG-16      2014   16    138M    7.3%    Small 3×3 filters, depth         │
│  GoogLeNet   2014   22      5M    6.7%    Inception modules, 1×1 convs     │
│  ResNet-50   2015   50     26M    5.3%    Skip connections                 │
│  ResNet-152  2015  152     60M    4.5%    Very deep with residuals         │
│                                                                            │
│  Trend: More depth + better training methods → lower error                 │
│                                                                            │
│  Error Rate Evolution on ImageNet:                                         │
│  ────────────────────────────────────────────────────────────────         │
│  30% ─┬──────────────────────────────────────────────────────────         │
│       │  █                                                                 │
│  25% ─┤  █ AlexNet (2012)                                                 │
│       │  █                                                                 │
│  20% ─┤  █                                                                 │
│       │  █                                                                 │
│  15% ─┤  █───█                                                            │
│       │      █ VGG (2014)                                                 │
│  10% ─┤      █                                                            │
│       │      █───█ GoogLeNet                                              │
│   5% ─┤          █───█ ResNet                                             │
│       │              █───█ ResNeXt, SENet...                              │
│   0% ─┴──────────────────────────────────────────────────────────         │
│        2012   2013   2014   2015   2016   2017                            │
│                                                                            │
│  Human performance: ~5% top-5 error                                        │
│  ResNet-152 (2015): 4.5% — first to surpass human level!                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.6 Modern Practices

### Batch Normalization in CNNs

```
BatchNorm Placement in Conv Blocks:

Recommended order:
┌─────────────────────────────────┐
│  Conv → BatchNorm → ReLU        │  Standard (original paper)
└─────────────────────────────────┘

Alternative (sometimes used):
┌─────────────────────────────────┐
│  Conv → ReLU → BatchNorm        │  Less common
└─────────────────────────────────┘

BatchNorm for Convolutions:
───────────────────────────────────────────────────────────────
For input shape [N, C, H, W]:
- Compute mean and variance per channel across N, H, W
- Each channel has its own γ and β parameters

                       ┌──────────────────────────┐
Channel 0:             │     Batch dimension      │
                       │   ┌───┬───┬───┬───┐     │
                       │   │ 0 │ 1 │...│N-1│     │
                       │   └───┴───┴───┴───┘     │
                       │          ↓               │
                       │   Compute μ₀, σ₀²       │
                       │   across all N images    │
                       │   and all H×W positions  │
                       └──────────────────────────┘

Parameters: 2 × C (γ and β per channel)
For 512 channels: only 1024 learnable parameters
```

### 1×1 Convolutions

```
1×1 Convolutions: More Useful Than They Look!
═════════════════════════════════════════════

What they do:
─────────────
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input: [H × W × C_in]        1×1 Conv         [H × W × C_out]  │
│                                                                 │
│  ┌──────────────────┐     ┌─────────────┐     ┌──────────────┐ │
│  │                  │     │             │     │              │ │
│  │  █ █ █ █ █ █ █   │     │   K × 1×1   │     │  █ █ █ █ █   │ │
│  │  C_in channels   │  ×  │   filters   │  =  │  K channels  │ │
│  │                  │     │             │     │              │ │
│  └──────────────────┘     └─────────────┘     └──────────────┘ │
│                                                                 │
│  Each 1×1×C_in filter produces one output channel               │
│  K filters → K output channels                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Use cases:
─────────

1. DIMENSIONALITY REDUCTION (Channel compression)
   Input: 256 channels → 1×1 Conv with 64 filters → 64 channels
   Reduces computation in subsequent layers

2. DIMENSIONALITY EXPANSION (Channel expansion)
   Input: 64 channels → 1×1 Conv with 256 filters → 256 channels
   Used in bottleneck blocks after spatial convolution

3. CROSS-CHANNEL INTERACTION
   Mixes information across channels at each spatial position
   Like a mini fully-connected layer applied at every pixel

4. ADDING NON-LINEARITY
   1×1 Conv + ReLU adds another non-linear transformation
   Increases network capacity without increasing receptive field

Example: Bottleneck dimension reduction
───────────────────────────────────────
Input: [14 × 14 × 256]
Without 1×1: 256 × 3×3 × 256 = 589,824 params for 3×3 conv
With 1×1:    256 × 1×1 × 64 = 16,384 (reduce)
             64 × 3×3 × 64 = 36,864 (3×3 conv)
             64 × 1×1 × 256 = 16,384 (expand)
             Total: 69,632 params — 8.5× fewer!
```

### Global Average Pooling

```
Global Average Pooling: Replacing FC Layers
═══════════════════════════════════════════

Traditional (VGG-style):
─────────────────────────
Feature maps → Flatten → FC → FC → Output

[7 × 7 × 512]     25,088    4096    1000
                     ↓        ↓
              25,088 × 4096 = 102M params (just first FC!)

Modern (ResNet-style):
─────────────────────────
Feature maps → Global Average Pool → FC → Output

[7 × 7 × 512]        512             1000
    ↓                  ↓
Average each       512 × 1000 = 512K params
7×7 feature map
to single value

Visualization:
──────────────
                Feature Maps [7 × 7 × 512]
    ┌─────────────────────────────────────────────────────┐
    │ ┌─────┐ ┌─────┐ ┌─────┐           ┌─────┐         │
    │ │█████│ │▓▓▓▓▓│ │░░░░░│    ...    │▒▒▒▒▒│         │
    │ │█████│ │▓▓▓▓▓│ │░░░░░│           │▒▒▒▒▒│         │
    │ │█████│ │▓▓▓▓▓│ │░░░░░│           │▒▒▒▒▒│         │
    │ └──┬──┘ └──┬──┘ └──┬──┘           └──┬──┘         │
    │    │       │       │                 │             │
    │   avg     avg     avg              avg             │
    │    │       │       │                 │             │
    │    ▼       ▼       ▼                 ▼             │
    │  [0.7]   [0.3]   [0.5]    ...     [0.9]           │
    └─────────────────────────────────────────────────────┘
                        ↓
              512-dimensional vector
                        ↓
                    FC layer
                        ↓
                   1000 classes

Benefits:
─────────
✓ Massive parameter reduction (100M → 0.5M)
✓ More robust to input size variations
✓ Acts as structural regularizer
✓ Each feature map can be interpreted as detecting a concept
```

---

## 3.7 Transfer Learning

### The Power of Pre-trained Features

```
WHY TRANSFER LEARNING WORKS
═══════════════════════════

Pre-trained CNN features are remarkably universal:

Layer 1-2 (Early):           Layer 3-4 (Middle):         Layer 5+ (Late):
Gabor filters, colors        Textures, patterns          Object parts
┌───────────────────┐        ┌───────────────────┐       ┌───────────────────┐
│  │ ─ ╱ ● ○        │        │ ▓▓▓ ░░░ ███       │       │  👁 🦶 ⚙         │
│                   │        │ textures          │       │  eyes, wheels     │
│  VERY UNIVERSAL   │        │  FAIRLY UNIVERSAL │       │  TASK-SPECIFIC    │
│  Same for almost  │        │  Similar across   │       │  Depends on       │
│  all vision tasks │        │  many tasks       │       │  target domain    │
└───────────────────┘        └───────────────────┘       └───────────────────┘

Transfer learning leverages: Years of ImageNet training + millions of images
→ Better starting point than random initialization
→ Often works even when target task is quite different
```

### Transfer Learning Strategies

```
STRATEGY 1: FEATURE EXTRACTION (Small dataset, similar domain)
═══════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   Pre-trained CNN (frozen weights)              New Head       │
│   ┌──────────────────────────────────┐    ┌─────────────────┐ │
│   │  Conv1 → Conv2 → ... → Conv5     │───►│ FC → New classes│ │
│   │         (ALL FROZEN)             │    │   (TRAINABLE)   │ │
│   └──────────────────────────────────┘    └─────────────────┘ │
│                                                                │
│   What to do:                                                  │
│   1. Load pre-trained model (e.g., ResNet-50 on ImageNet)     │
│   2. Remove final classification layer                         │
│   3. Freeze all other layers (requires_grad = False)          │
│   4. Add new FC layer for your classes                         │
│   5. Train ONLY the new layer                                  │
│                                                                │
│   Pros: Fast, works with tiny datasets (100s of images)       │
│   Cons: Limited adaptation to new domain                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Code:
─────
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(2048, num_classes)  # Only this trains

# Train
optimizer = Adam(model.fc.parameters(), lr=1e-3)  # Only FC params
```

```
STRATEGY 2: FINE-TUNING ALL LAYERS (Medium dataset)
═══════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   Pre-trained CNN (trainable)                   New Head       │
│   ┌──────────────────────────────────┐    ┌─────────────────┐ │
│   │  Conv1 → Conv2 → ... → Conv5     │───►│ FC → New classes│ │
│   │       (ALL TRAINABLE)            │    │   (TRAINABLE)   │ │
│   └──────────────────────────────────┘    └─────────────────┘ │
│                                                                │
│   What to do:                                                  │
│   1. Load pre-trained model                                    │
│   2. Replace final layer for your classes                      │
│   3. Train entire network with SMALL learning rate             │
│   4. Use smaller LR for early layers (discriminative LR)       │
│                                                                │
│   Pros: Better adaptation, higher accuracy                     │
│   Cons: Needs more data, risk of overfitting                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Code:
─────
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)

# Different learning rates for different parts
optimizer = Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

```
STRATEGY 3: FINE-TUNE LATER LAYERS ONLY (Medium dataset, different domain)
══════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   Pre-trained CNN                                New Head      │
│   ┌──────────────────┬───────────────────┐ ┌─────────────────┐│
│   │ Conv1 → Conv2    │ Conv3 → Conv4 → 5 │►│ FC → New classes││
│   │   (FROZEN)       │   (TRAINABLE)     │ │   (TRAINABLE)   ││
│   └──────────────────┴───────────────────┘ └─────────────────┘│
│                                                                │
│   Rationale:                                                   │
│   - Early layers: Universal features (edges, colors)          │
│   - Keep them as-is, they're already good                      │
│   - Later layers: Task-specific, need adaptation               │
│   - Train them for your specific task                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Code:
─────
model = models.resnet50(pretrained=True)

# Freeze early layers
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Later layers + new head remain trainable
model.fc = nn.Linear(2048, num_classes)

# Train
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

### Transfer Learning Decision Guide

```
DECISION MATRIX FOR TRANSFER LEARNING
═════════════════════════════════════

                        Target Domain Similar    Target Domain Different
                        to Source (ImageNet)     from Source
                   ┌────────────────────────┬────────────────────────────┐
                   │                        │                            │
   Small Dataset   │  FEATURE EXTRACTION    │  FEATURE EXTRACTION +      │
   (< 1K images)   │                        │  DATA AUGMENTATION         │
                   │  Freeze all, train FC  │  Freeze all, augment lots  │
                   │                        │  May need domain-specific  │
                   │                        │  pre-training              │
                   ├────────────────────────┼────────────────────────────┤
                   │                        │                            │
   Medium Dataset  │  FINE-TUNE ALL         │  FINE-TUNE LATER LAYERS    │
   (1K-100K)       │                        │                            │
                   │  Small LR everywhere   │  Freeze early, train late  │
                   │  Best results          │  Early features still      │
                   │                        │  useful (edges, etc.)      │
                   ├────────────────────────┼────────────────────────────┤
                   │                        │                            │
   Large Dataset   │  FINE-TUNE ALL or      │  TRAIN FROM SCRATCH or     │
   (> 100K)        │  TRAIN FROM SCRATCH    │  FINE-TUNE ALL             │
                   │                        │                            │
                   │  Pre-training may not  │  Enough data to learn      │
                   │  help much             │  domain-specific features  │
                   │                        │                            │
                   └────────────────────────┴────────────────────────────┘

Common Pre-trained Models (Available in PyTorch torchvision):
─────────────────────────────────────────────────────────────
Model           Params    Top-1    Use Case
────────────────────────────────────────────────────────────
ResNet-18       11.7M     30.2%    Mobile, edge devices
ResNet-50       25.6M     23.9%    General purpose, balanced
ResNet-152      60.2M     21.7%    Maximum accuracy
EfficientNet-B0  5.3M     23.7%    Efficient, mobile
EfficientNet-B7  66M      15.6%    State-of-art accuracy
ViT-B/16        86M       18.6%    Transformers for vision
────────────────────────────────────────────────────────────
```

---

## 3.8 Understanding CNNs

### Visualizing First Layer Filters

```
FIRST LAYER FILTERS ARE INTERPRETABLE
═════════════════════════════════════

After training on natural images, first layer filters learn:

┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  Color Detectors:                 Edge/Gabor Detectors:                   │
│  ┌─────────────────────────┐      ┌─────────────────────────────────────┐│
│  │                         │      │                                     ││
│  │  [R] [G] [B] [Y] [C]    │      │  │  ─  ╱  ╲  ┼  ╬  various         ││
│  │  Red Green Blue Yellow  │      │  orientations and frequencies       ││
│  │  Cyan detectors         │      │                                     ││
│  │                         │      │                                     ││
│  └─────────────────────────┘      └─────────────────────────────────────┘│
│                                                                           │
│  Example 11×11 AlexNet first layer filters (96 filters):                  │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                              │
│  │ ║ │ ═ │ ╱ │ ╲ │ R │ G │ B │ Y │ ╬ │...│                              │
│  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤                              │
│  │ ║ │ ═ │ ╱ │ ╲ │ ▓ │ ░ │ █ │ ▒ │ ╬ │...│                              │
│  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤                              │
│  │...│...│...│...│...│...│...│...│...│...│                              │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                              │
│                                                                           │
│  These match neuroscience findings about visual cortex!                   │
│  Early visual cortex has edge-detecting neurons (Hubel & Wiesel, 1962)   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Feature Map Visualization

```
VISUALIZING INTERMEDIATE ACTIVATIONS
════════════════════════════════════

Input Image                        Feature Maps at Different Depths
    ┌─────────┐
    │  🐱     │
    │  cat    │
    │  image  │
    └────┬────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ Layer 1: Responds to edges                                         │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                   │
│ │ │││ │ │ ─── │ │ ╱╱╱ │ │ ╲╲╲ │  Edge maps                        │
│ │ │││ │ │ ─── │ │ ╱╱╱ │ │ ╲╲╲ │  show where                       │
│ │ │││ │ │ ─── │ │ ╱╱╱ │ │ ╲╲╲ │  edges are                        │
│ └─────┘ └─────┘ └─────┘ └─────┘                                   │
│ vertical horizontal diagonal                                       │
├────────────────────────────────────────────────────────────────────┤
│ Layer 2-3: Responds to textures and patterns                       │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                   │
│ │ ░░░ │ │ ▓▓▓ │ │ ○○○ │ │ ███ │  Fur texture,                     │
│ │ ░░░ │ │ ▓▓▓ │ │ ○○○ │ │ ███ │  stripes,                         │
│ │ ░░░ │ │ ▓▓▓ │ │ ○○○ │ │ ███ │  spots                            │
│ └─────┘ └─────┘ └─────┘ └─────┘                                   │
├────────────────────────────────────────────────────────────────────┤
│ Layer 4-5: Responds to semantic parts/objects                      │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                   │
│ │ 👁   │ │ 🐾   │ │ 👂   │ │ 🐱   │  Eyes, paws,                    │
│ │ eye │ │ paw │ │ ear │ │ face│  ears, whole                      │
│ │     │ │     │ │     │ │     │  cat face                         │
│ └─────┘ └─────┘ └─────┘ └─────┘                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Receptive Field

```
RECEPTIVE FIELD: How much input a neuron "sees"
═══════════════════════════════════════════════

After each layer, neurons have larger receptive fields:

Layer 0: Input image
┌───────────────────────────────────────────┐
│  Each pixel = 1×1 receptive field         │
└───────────────────────────────────────────┘

Layer 1: After 3×3 conv
┌───────────────────────────────────────────┐
│  Each neuron sees 3×3 input region        │
│                                           │
│  ┌─────┐                                  │
│  │ █ █ █ │                                │
│  │ █ ○ █ │ ← This neuron sees             │
│  │ █ █ █ │   9 input pixels               │
│  └─────┘                                  │
└───────────────────────────────────────────┘

Layer 2: After another 3×3 conv
┌───────────────────────────────────────────┐
│  Each neuron sees 5×5 input region        │
│                                           │
│  ┌───────────┐                            │
│  │ █ █ █ █ █ │                            │
│  │ █ █ █ █ █ │                            │
│  │ █ █ ○ █ █ │ ← This neuron sees         │
│  │ █ █ █ █ █ │   25 input pixels          │
│  │ █ █ █ █ █ │                            │
│  └───────────┘                            │
└───────────────────────────────────────────┘

Receptive Field Formula:
────────────────────────
For n layers of k×k convolutions with stride 1:
RF = n × (k - 1) + 1

Examples:
1 layer 3×3:  RF = 1×2 + 1 = 3
2 layers 3×3: RF = 2×2 + 1 = 5
3 layers 3×3: RF = 3×2 + 1 = 7
5 layers 3×3: RF = 5×2 + 1 = 11

With pooling (stride 2), receptive field grows faster:
Each pooling approximately doubles the RF
```

---

## 3.9 Summary

### Key Concepts

```
CONVOLUTIONAL NEURAL NETWORKS: KEY TAKEAWAYS
════════════════════════════════════════════

1. MOTIVATION
   - FC layers have too many parameters for images
   - CNNs exploit spatial structure through local connectivity

2. CONVOLUTION OPERATION
   - Filter slides across image computing dot products
   - Learns feature detectors (edges → textures → parts → objects)
   - Key hyperparameters: filter size, stride, padding

3. POOLING
   - Reduces spatial dimensions (typically 2×)
   - Provides translation invariance
   - Max pooling most common; GAP for final layer

4. ARCHITECTURES
   - LeNet (1998): Proved CNNs work
   - AlexNet (2012): Started the revolution (ReLU, dropout, GPUs)
   - VGG (2014): Small 3×3 filters everywhere
   - Inception (2014): Parallel filter sizes, 1×1 convolutions
   - ResNet (2015): Skip connections enable very deep networks

5. TRANSFER LEARNING
   - Pre-trained features are universal
   - Feature extraction: Freeze backbone, train new head
   - Fine-tuning: Train entire network with small LR
   - Almost always better than training from scratch

6. UNDERSTANDING CNNs
   - Early layers: Edges, colors (interpretable)
   - Later layers: Parts, objects (semantic)
   - Receptive field grows with depth
```

### Glossary Terms

| Term | Definition |
|------|------------|
| **Convolution** | Operation where filter slides across input computing weighted sums |
| **Filter/Kernel** | Small learnable weight matrix that detects features |
| **Feature Map** | Output of applying one filter across input |
| **Stride** | Step size when sliding filter |
| **Padding** | Zeros added around input border |
| **Pooling** | Downsampling operation (max or average) |
| **Receptive Field** | Region of input that affects one output neuron |
| **Skip Connection** | Direct connection bypassing layers (ResNet) |
| **Transfer Learning** | Using pre-trained weights for new tasks |
| **Feature Extraction** | Using frozen pre-trained features |
| **Fine-tuning** | Training pre-trained model on new task |

---

## 3.10 Exercises

1. **Output Size Calculation**: Given input 64×64×3, Conv 5×5×32, stride 1, padding "same", what's the output shape?

2. **Parameter Count**: Calculate parameters for a Conv layer with 128 filters of 3×3 on input with 64 channels.

3. **Receptive Field**: After 4 consecutive 3×3 conv layers (stride 1, no pooling), what's the receptive field?

4. **Architecture Design**: Design a CNN for 128×128×3 input with 5 classes. Show layer dimensions.

5. **Transfer Learning**: You have 500 images of 10 flower species. Describe your transfer learning strategy.

6. **Code Exercise**: Implement ResNet-18 from scratch in PyTorch.

---

## References

### Foundational Papers

- **LeNet-5**: LeCun et al., ["Gradient-Based Learning Applied to Document Recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) (1998)
- **AlexNet**: Krizhevsky et al., ["ImageNet Classification with Deep Convolutional Neural Networks"](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (2012)
- **VGGNet**: Simonyan & Zisserman, ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) (2014)
- **GoogLeNet/Inception**: Szegedy et al., ["Going Deeper with Convolutions"](https://arxiv.org/abs/1409.4842) (2014)
- **ResNet**: He et al., ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) (2015)
- **Batch Normalization**: Ioffe & Szegedy, ["Batch Normalization: Accelerating Deep Network Training"](https://arxiv.org/abs/1502.03167) (2015)

### Courses and Lectures

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  - [Lecture 5: Convolutional Neural Networks](http://cs231n.stanford.edu/slides/2023/lecture_5.pdf)
  - [Lecture 9: CNN Architectures](http://cs231n.stanford.edu/slides/2023/lecture_9.pdf)
- [CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
  - CNNs for text classification (Lecture 11)

### Books and Resources

- **Deep Learning** by Goodfellow, Bengio, Courville
  - [Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- **Neural Networks and Deep Learning** by Michael Nielsen
  - [Chapter 6: CNNs](http://neuralnetworksanddeeplearning.com/chap6.html)
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Distill.pub: Feature Visualization](https://distill.pub/2017/feature-visualization/)

### Interactive Resources

- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Interactive CNN visualization
- [Setosa.io Convolution Demo](https://setosa.io/ev/image-kernels/) - Interactive kernel visualization
- [TensorFlow Playground CNN](https://playground.tensorflow.org/) - Neural network playground

### Visualization Papers

- Zeiler & Fergus, ["Visualizing and Understanding Convolutional Networks"](https://arxiv.org/abs/1311.2901) (2013)
- Yosinski et al., ["How transferable are features in deep neural networks?"](https://arxiv.org/abs/1411.1792) (2014)

---

*Module 3 complete. Next: Module 4 covers Sequence Models (RNNs, LSTMs) for sequential data.*
