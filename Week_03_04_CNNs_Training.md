# Week 3-4: Convolutional Neural Networks & Training Techniques
## From Novice to Practitioner: CNNs, Batch Norm, Dropout, and Optimization

---

## Table of Contents
1. [Why CNNs?](#why-cnns)
2. [Convolution Operation](#convolution-operation)
3. [CNN Architecture](#cnn-architecture)
4. [Pooling Layers](#pooling-layers)
5. [Data Preprocessing](#data-preprocessing)
6. [Weight Initialization](#weight-initialization)
7. [Batch Normalization](#batch-normalization)
8. [Dropout](#dropout)
9. [Advanced Optimizers](#advanced-optimizers)
10. [Learning Rate Schedules](#learning-rate-schedules)
11. [Coding Exercises](#coding-exercises)
12. [Business Applications](#business-applications)

---

## Why CNNs?

### The Problem with Fully Connected Networks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 WHY FULLY CONNECTED NETWORKS FAIL FOR IMAGES                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FULLY CONNECTED APPROACH:                                                │
│                                                                             │
│   Image (224x224x3)              Hidden Layer (4096)                       │
│   ┌──────────────────┐           ┌─────────────────┐                       │
│   │                  │           │                 │                       │
│   │   150,528 pixels │──────────►│  4,096 neurons  │                       │
│   │                  │           │                 │                       │
│   └──────────────────┘           └─────────────────┘                       │
│                                                                             │
│   Parameters: 150,528 × 4,096 = 616 MILLION just for one layer!            │
│                                                                             │
│   PROBLEMS:                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  1. TOO MANY PARAMETERS                                             │  │
│   │     • Memory: 616M × 4 bytes = 2.5 GB per layer                     │  │
│   │     • Overfitting: More params than training examples               │  │
│   │                                                                     │  │
│   │  2. IGNORES SPATIAL STRUCTURE                                       │  │
│   │     • Adjacent pixels are related (edges, textures)                 │  │
│   │     • Fully connected treats each pixel independently               │  │
│   │                                                                     │  │
│   │  3. NOT TRANSLATION INVARIANT                                       │  │
│   │     • A cat in top-left looks different from cat in bottom-right   │  │
│   │     • Need separate weights for each position                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   CNN SOLUTION:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  1. LOCAL CONNECTIVITY: Each neuron sees only a small region       │  │
│   │  2. PARAMETER SHARING: Same filter slides across entire image      │  │
│   │  3. TRANSLATION INVARIANCE: Same pattern detected anywhere         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   A 3x3 filter: 3 × 3 × 3 = 27 parameters (instead of 150,528!)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CNN Intuition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CNN INTUITION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   HIERARCHICAL FEATURE LEARNING:                                           │
│                                                                             │
│   Input        Layer 1         Layer 2          Layer 3         Output     │
│   Image        (Edges)        (Textures)       (Parts)         (Objects)   │
│                                                                             │
│   ┌─────┐     ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐     │
│   │     │     │ / \ │         │░░░░░│         │     │         │     │     │
│   │  🐱 │ ──► │ | - │   ──►   │▓▓▓▓▓│   ──►   │ 👁 👁│   ──►   │ CAT │     │
│   │     │     │ _ / │         │▒▒▒▒▒│         │  👃 │         │     │     │
│   └─────┘     └─────┘         └─────┘         └─────┘         └─────┘     │
│                                                                             │
│   Learns:     Learns:         Learns:         Learns:         Classifies:  │
│   Raw         Simple          Complex         Object          Full         │
│   Pixels      Edges           Textures        Parts           Objects      │
│                                                                             │
│                                                                             │
│   WHAT EACH LAYER "SEES":                                                  │
│                                                                             │
│   Layer 1 Filters (learned edge detectors):                                │
│   ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                              │
│   │-1 0 1 │  │-1-1-1 │  │ 1-1 1 │  │ 0 1 0 │                              │
│   │-1 0 1 │  │ 0 0 0 │  │-1 1-1 │  │ 1 0 1 │                              │
│   │-1 0 1 │  │ 1 1 1 │  │ 1-1 1 │  │ 0 1 0 │                              │
│   └───────┘  └───────┘  └───────┘  └───────┘                              │
│   Vertical   Horizontal  Diagonal   Corner                                 │
│   Edges      Edges       Edges      Detector                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Convolution Operation

### How Convolution Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONVOLUTION OPERATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input Image (5×5)              Filter (3×3)            Output (3×3)       │
│                                                                             │
│   ┌───┬───┬───┬───┬───┐         ┌───┬───┬───┐           ┌───┬───┬───┐     │
│   │ 1 │ 2 │ 3 │ 0 │ 1 │         │ 1 │ 0 │-1 │           │ ? │   │   │     │
│   ├───┼───┼───┼───┼───┤         ├───┼───┼───┤           ├───┼───┼───┤     │
│   │ 0 │ 1 │ 2 │ 3 │ 0 │         │ 1 │ 0 │-1 │           │   │   │   │     │
│   ├───┼───┼───┼───┼───┤    *    ├───┼───┼───┤     =     ├───┼───┼───┤     │
│   │ 3 │ 0 │ 1 │ 2 │ 1 │         │ 1 │ 0 │-1 │           │   │   │   │     │
│   ├───┼───┼───┼───┼───┤         └───┴───┴───┘           └───┴───┴───┘     │
│   │ 2 │ 1 │ 0 │ 1 │ 2 │                                                    │
│   ├───┼───┼───┼───┼───┤                                                    │
│   │ 1 │ 0 │ 2 │ 3 │ 0 │                                                    │
│   └───┴───┴───┴───┴───┘                                                    │
│                                                                             │
│   STEP BY STEP:                                                            │
│                                                                             │
│   Position 1 (top-left):                                                   │
│   ┌───────────────────┐                                                    │
│   │ 1×1 + 2×0 + 3×(-1)│ = 1 + 0 - 3 = -2                                  │
│   │ 0×1 + 1×0 + 2×(-1)│ = 0 + 0 - 2 = -2                                  │
│   │ 3×1 + 0×0 + 1×(-1)│ = 3 + 0 - 1 = 2                                   │
│   │       SUM         │ = -2 - 2 + 2 = -2                                  │
│   └───────────────────┘                                                    │
│                                                                             │
│   Position 2 (slide right by 1):                                           │
│   ┌───────────────────┐                                                    │
│   │ 2×1 + 3×0 + 0×(-1)│ = 2                                               │
│   │ 1×1 + 2×0 + 3×(-1)│ = -2                                              │
│   │ 0×1 + 1×0 + 2×(-1)│ = -2                                              │
│   │       SUM         │ = -2                                               │
│   └───────────────────┘                                                    │
│                                                                             │
│   Continue sliding across entire image...                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Convolution with Multiple Channels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CONVOLUTION WITH RGB IMAGES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: 32×32×3 (RGB image)                                               │
│   Filter: 3×3×3 (must match input channels)                                │
│   Output: 30×30×1 (one activation map)                                     │
│                                                                             │
│                                                                             │
│   INPUT IMAGE                     FILTER                    OUTPUT         │
│   (32×32×3)                       (3×3×3)                  (30×30×1)       │
│                                                                             │
│   ┌────────────┐                  ┌────────┐               ┌────────┐      │
│   │  R channel │                  │ R filt │               │        │      │
│   │    32×32   │  ┌────────┐      │  3×3   │  ┌────────┐   │  One   │      │
│   ├────────────┤  │  G ch  │      ├────────┤  │ G filt │   │ output │      │
│   │  G channel │  │  32×32 │  *   │  3×3   │  │  3×3   │ = │ 30×30  │      │
│   ├────────────┤  ├────────┤      ├────────┤  ├────────┤   │        │      │
│   │  B channel │  │  B ch  │      │  3×3   │  │ B filt │   │        │      │
│   │    32×32   │  │  32×32 │      │  3×3   │  │  3×3   │   │        │      │
│   └────────────┘  └────────┘      └────────┘  └────────┘   └────────┘      │
│                                                                             │
│   Computation at each position:                                            │
│   - Multiply R_input × R_filter → sum                                      │
│   - Multiply G_input × G_filter → sum                                      │
│   - Multiply B_input × B_filter → sum                                      │
│   - Add all three + bias = one output value                                │
│                                                                             │
│   MULTIPLE FILTERS = MULTIPLE OUTPUT CHANNELS:                             │
│                                                                             │
│   ┌────────────┐      ┌───────────────────┐      ┌────────────────┐       │
│   │            │      │ Filter 1 (3×3×3)  │      │ Output ch 1    │       │
│   │   Input    │      │ Filter 2 (3×3×3)  │      │ Output ch 2    │       │
│   │  32×32×3   │  *   │ Filter 3 (3×3×3)  │  =   │ Output ch 3    │       │
│   │            │      │    ...            │      │    ...         │       │
│   │            │      │ Filter 64 (3×3×3) │      │ Output ch 64   │       │
│   └────────────┘      └───────────────────┘      └────────────────┘       │
│                                                                             │
│   Parameters: 64 filters × 3×3×3 = 64 × 27 = 1,728 parameters             │
│   (Much less than 150,528 × 4,096 = 616 million for fully connected!)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Padding and Stride

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PADDING AND STRIDE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   OUTPUT SIZE FORMULA:                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │              (W - F + 2P)                                           │  │
│   │   Output  =  ─────────────  + 1                                     │  │
│   │                   S                                                 │  │
│   │                                                                     │  │
│   │   W = input size, F = filter size, P = padding, S = stride         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   STRIDE = 1 (default):                                                    │
│   ┌───┬───┬───┬───┬───┐                                                   │
│   │ ▓ │ ▓ │ ▓ │   │   │  Step 1: Filter at position 0                    │
│   ├───┼───┼───┼───┼───┤                                                   │
│   │ ▓ │ ▓ │ ▓ │   │   │                                                   │
│   ├───┼───┼───┼───┼───┤                                                   │
│   │ ▓ │ ▓ │ ▓ │   │   │                                                   │
│   └───┴───┴───┴───┴───┘                                                   │
│                                                                             │
│   ┌───┬───┬───┬───┬───┐                                                   │
│   │   │ ▓ │ ▓ │ ▓ │   │  Step 2: Move right by 1                         │
│   ├───┼───┼───┼───┼───┤                                                   │
│   │   │ ▓ │ ▓ │ ▓ │   │                                                   │
│   ├───┼───┼───┼───┼───┤                                                   │
│   │   │ ▓ │ ▓ │ ▓ │   │                                                   │
│   └───┴───┴───┴───┴───┘                                                   │
│                                                                             │
│   STRIDE = 2 (downsampling):                                               │
│   ┌───┬───┬───┬───┬───┐         ┌───┬───┬───┬───┬───┐                    │
│   │ ▓ │ ▓ │ ▓ │   │   │   ──►   │   │   │ ▓ │ ▓ │ ▓ │                    │
│   ├───┼───┼───┼───┼───┤         ├───┼───┼───┼───┼───┤                    │
│   │ ▓ │ ▓ │ ▓ │   │   │         │   │   │ ▓ │ ▓ │ ▓ │                    │
│   ├───┼───┼───┼───┼───┤         ├───┼───┼───┼───┼───┤                    │
│   │ ▓ │ ▓ │ ▓ │   │   │         │   │   │ ▓ │ ▓ │ ▓ │                    │
│   └───┴───┴───┴───┴───┘         └───┴───┴───┴───┴───┘                    │
│   Step 1: pos 0                 Step 2: Skip 1, pos 2                     │
│                                                                             │
│   PADDING (preserves spatial size):                                        │
│   ┌───┬───┬───┬───┬───┬───┬───┐                                           │
│   │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │  ← Zero padding                          │
│   ├───┼───┼───┼───┼───┼───┼───┤                                           │
│   │ 0 │ ▓ │ ▓ │ ▓ │   │   │ 0 │                                           │
│   ├───┼───┼───┼───┼───┼───┼───┤                                           │
│   │ 0 │ ▓ │ ▓ │ ▓ │   │   │ 0 │  Original 5×5 + padding 1                │
│   ├───┼───┼───┼───┼───┼───┼───┤  = 7×7                                    │
│   │ 0 │ ▓ │ ▓ │ ▓ │   │   │ 0 │                                           │
│   ├───┼───┼───┼───┼───┼───┼───┤  Output: (7-3+0)/1 + 1 = 5×5             │
│   │ 0 │   │   │   │   │   │ 0 │  Same as input!                          │
│   ├───┼───┼───┼───┼───┼───┼───┤                                           │
│   │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │                                           │
│   └───┴───┴───┴───┴───┴───┴───┘                                           │
│                                                                             │
│   COMMON SETTINGS:                                                         │
│   • F=3, P=1, S=1: Output same size as input                              │
│   • F=3, P=0, S=2: Output half size (downsampling)                        │
│   • F=1, P=0, S=1: 1×1 convolution (channel mixing)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CNN Architecture

### Standard Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CNN ARCHITECTURE PATTERN                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GENERAL PATTERN:                                                         │
│   INPUT → [[CONV → ReLU] × N → POOL?] × M → [FC → ReLU] × K → FC          │
│                                                                             │
│                                                                             │
│   EXAMPLE: Simple CNN for CIFAR-10                                         │
│                                                                             │
│   Layer          Output Size      Parameters        Description            │
│   ─────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   INPUT          32×32×3          0                 RGB image              │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  CONV1       32×32×32         (3×3×3)×32 = 864    32 filters, 3×3    │  │
│   │  ReLU        32×32×32         0                                      │  │
│   │  POOL        16×16×32         0                   2×2 max pool       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  CONV2       16×16×64         (3×3×32)×64=18,432  64 filters, 3×3   │  │
│   │  ReLU        16×16×64         0                                      │  │
│   │  POOL        8×8×64           0                   2×2 max pool       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  CONV3       8×8×128          (3×3×64)×128=73,728 128 filters       │  │
│   │  ReLU        8×8×128          0                                      │  │
│   │  POOL        4×4×128          0                   2×2 max pool       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  FLATTEN     2048             0                   4×4×128 = 2048     │  │
│   │  FC1         512              2048×512=1,048,576                     │  │
│   │  ReLU        512              0                                      │  │
│   │  FC2         10               512×10=5,120        Output classes     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TOTAL PARAMETERS: ~1.15 million                                          │
│                                                                             │
│                                                                             │
│   SPATIAL SIZE PROGRESSION:                                                │
│                                                                             │
│   32×32 ──CONV──► 32×32 ──POOL──► 16×16 ──CONV──► 16×16 ──POOL──► 8×8    │
│     │                                                               │      │
│     └── Channels: 3 ────► 32 ────► 32 ────► 64 ────► 64 ─────► 128 ─┘      │
│                                                                             │
│   As spatial size DECREASES, channels INCREASE (preserves information)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Famous Architectures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FAMOUS CNN ARCHITECTURES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LENET-5 (1998) - Yann LeCun                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  INPUT(32×32) → CONV → POOL → CONV → POOL → FC → FC → OUTPUT(10)   │  │
│   │  • First successful CNN for digit recognition                       │  │
│   │  • ~60K parameters                                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ALEXNET (2012) - Krizhevsky et al.                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  INPUT(227×227) → CONV(96,11×11,s4) → POOL → CONV(256,5×5) → ...   │  │
│   │  • Won ImageNet 2012 (started deep learning revolution)            │  │
│   │  • 60M parameters, trained on 2 GPUs                                │  │
│   │  • Introduced ReLU, Dropout, Data Augmentation                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   VGGNet (2014) - Simonyan & Zisserman                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  KEY INSIGHT: Use only 3×3 filters, stack them deep                │  │
│   │                                                                     │  │
│   │  Two 3×3 filters = One 5×5 filter (same receptive field)          │  │
│   │  But: 2×(3×3) = 18 params vs 5×5 = 25 params (fewer!)             │  │
│   │       + extra non-linearity (ReLU between them)                    │  │
│   │                                                                     │  │
│   │  VGG-16: 16 layers, 138M parameters                                │  │
│   │  VGG-19: 19 layers                                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   RESNET (2015) - He et al.                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  KEY INSIGHT: Skip connections (residual learning)                 │  │
│   │                                                                     │  │
│   │       x ─────────────────────────┐                                 │  │
│   │       │                          │                                 │  │
│   │       ▼                          │                                 │  │
│   │   ┌───────┐                      │                                 │  │
│   │   │ CONV  │                      │ (identity shortcut)             │  │
│   │   │ ReLU  │                      │                                 │  │
│   │   │ CONV  │                      │                                 │  │
│   │   └───┬───┘                      │                                 │  │
│   │       │                          │                                 │  │
│   │       └──────── + ◄──────────────┘                                 │  │
│   │                 │                                                   │  │
│   │                 ▼                                                   │  │
│   │              ReLU                                                   │  │
│   │                                                                     │  │
│   │  F(x) + x instead of just F(x)                                     │  │
│   │  • Easier to learn identity (just set F(x)=0)                      │  │
│   │  • Enables training 100+ layer networks                            │  │
│   │  • ResNet-152: 152 layers, 60M params                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pooling Layers

### Max Pooling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MAX POOLING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PURPOSE: Reduce spatial dimensions while keeping important features      │
│                                                                             │
│   2×2 Max Pooling with Stride 2:                                           │
│                                                                             │
│   Input (4×4)                                    Output (2×2)               │
│   ┌───┬───┬───┬───┐                             ┌───┬───┐                  │
│   │ 1 │ 3 │ 2 │ 1 │                             │ 4 │ 6 │                  │
│   ├───┼───┼───┼───┤          MAX                ├───┼───┤                  │
│   │ 4 │ 2 │ 6 │ 4 │    ─────────────►           │ 8 │ 9 │                  │
│   ├───┼───┼───┼───┤                             └───┴───┘                  │
│   │ 1 │ 8 │ 3 │ 2 │                                                        │
│   ├───┼───┼───┼───┤                                                        │
│   │ 7 │ 2 │ 9 │ 5 │                                                        │
│   └───┴───┴───┴───┘                                                        │
│                                                                             │
│   Region 1:          Region 2:          Region 3:          Region 4:       │
│   ┌───┬───┐          ┌───┬───┐          ┌───┬───┐          ┌───┬───┐      │
│   │ 1 │ 3 │          │ 2 │ 1 │          │ 1 │ 8 │          │ 3 │ 2 │      │
│   │ 4 │ 2 │          │ 6 │ 4 │          │ 7 │ 2 │          │ 9 │ 5 │      │
│   └───┴───┘          └───┴───┘          └───┴───┘          └───┴───┘      │
│   max = 4            max = 6            max = 8            max = 9         │
│                                                                             │
│                                                                             │
│   BENEFITS:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  1. REDUCES COMPUTATION: 4× fewer activations to process           │  │
│   │  2. TRANSLATION INVARIANCE: Small shifts don't change output       │  │
│   │  3. INCREASES RECEPTIVE FIELD: Each neuron "sees" larger area      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   BACKPROPAGATION:                                                         │
│   - Gradient flows only to the max element                                 │
│   - Other elements in the pooling region get zero gradient                 │
│                                                                             │
│   Forward:  [1, 3, 4, 2] → max = 4                                        │
│   Backward: grad_out = 1.5                                                 │
│             grad_in = [0, 0, 1.5, 0]  (only max element gets gradient)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing

### Standard Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA PREPROCESSING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1: MEAN SUBTRACTION (most important!)                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   # Compute mean on TRAINING data only                             │  │
│   │   mean = np.mean(X_train, axis=0)  # Per-pixel mean                │  │
│   │                                                                     │  │
│   │   # Apply to all sets                                              │  │
│   │   X_train -= mean                                                   │  │
│   │   X_val -= mean                                                     │  │
│   │   X_test -= mean                                                    │  │
│   │                                                                     │  │
│   │   WHY: Centers data around zero, helps optimization                │  │
│   │                                                                     │  │
│   │   Before:                      After:                              │  │
│   │   Values: [0, 255]             Values: [-128, 127]                 │  │
│   │   Mean: ~128                   Mean: ~0                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   STEP 2: NORMALIZATION (optional but helpful)                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   # Scale to unit variance                                         │  │
│   │   std = np.std(X_train, axis=0)                                    │  │
│   │   X_train /= (std + 1e-8)                                          │  │
│   │   X_val /= (std + 1e-8)                                            │  │
│   │   X_test /= (std + 1e-8)                                           │  │
│   │                                                                     │  │
│   │   WHY: Equal importance to all features                            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   CRITICAL MISTAKE TO AVOID:                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ✗ WRONG: Compute statistics on entire dataset                    │  │
│   │   mean = np.mean(ALL_DATA)  # LEAKS TEST INFO!                     │  │
│   │                                                                     │  │
│   │   ✓ RIGHT: Compute only on training set, apply to all             │  │
│   │   mean = np.mean(X_train)                                          │  │
│   │   X_test -= mean  # Use TRAINING mean on test                      │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   VISUALIZATION OF EFFECT:                                                 │
│                                                                             │
│   Raw Data:                 Mean-Subtracted:         Normalized:           │
│        ▲                         ▲                       ▲                 │
│   200  │    ●●●                  │   ●●●                 │   ●●●           │
│        │  ●●●                    │ ●●●                   │  ●●●            │
│   100  │●●●                    0─┼─●●●                 0─┼──●●●            │
│        │                         │  ●●●                  │   ●●●           │
│      0─┼────────►              ──┼────────►            ──┼────────►        │
│        0    255               -128  127                -3   3              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Weight Initialization

### Why Initialization Matters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WEIGHT INITIALIZATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BAD INITIALIZATION EXAMPLES:                                             │
│                                                                             │
│   1. ALL ZEROS:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   W = np.zeros((input_dim, output_dim))                            │  │
│   │                                                                     │  │
│   │   PROBLEM: All neurons compute same thing!                         │  │
│   │   - Forward: All outputs identical                                 │  │
│   │   - Backward: All gradients identical                              │  │
│   │   - Update: All weights update identically                         │  │
│   │   → Network can never learn different features                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   2. TOO LARGE:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   W = np.random.randn(input_dim, output_dim) * 1.0                 │  │
│   │                                                                     │  │
│   │   PROBLEM: Activations explode!                                    │  │
│   │                                                                     │  │
│   │   Layer 1: outputs ~N(0, 100)                                      │  │
│   │   Layer 2: outputs ~N(0, 10000)                                    │  │
│   │   Layer 3: outputs ~N(0, 1000000)  → OVERFLOW!                     │  │
│   │                                                                     │  │
│   │   Activations: │████████████████████████████████████░░░░░│         │  │
│   │                0                                    ∞              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   3. TOO SMALL:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   W = np.random.randn(input_dim, output_dim) * 0.001               │  │
│   │                                                                     │  │
│   │   PROBLEM: Activations vanish!                                     │  │
│   │                                                                     │  │
│   │   Layer 1: outputs ~N(0, 0.001)                                    │  │
│   │   Layer 2: outputs ~N(0, 0.000001)                                 │  │
│   │   Layer 3: outputs ~N(0, 0.000000001)  → All zeros!                │  │
│   │                                                                     │  │
│   │   Gradients also vanish → No learning                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                                                                             │
│   GOOD INITIALIZATION:                                                     │
│                                                                             │
│   XAVIER INITIALIZATION (for tanh/sigmoid):                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)           │  │
│   │                                                                     │  │
│   │   Variance of output ≈ Variance of input                           │  │
│   │   Maintains signal through many layers                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   HE INITIALIZATION (for ReLU):                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)           │  │
│   │                                                                     │  │
│   │   ReLU kills half the activations (negative → 0)                   │  │
│   │   So we need 2× larger variance to compensate                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ACTIVATION STATISTICS WITH GOOD INIT:                                    │
│                                                                             │
│   Layer 1        Layer 5        Layer 10       Layer 20                    │
│     ▲              ▲              ▲              ▲                         │
│     │  ██          │  ██          │  ██          │  ██                     │
│     │ ████         │ ████         │ ████         │ ████                    │
│     │██████        │██████        │██████        │██████                   │
│     └──────►       └──────►       └──────►       └──────►                  │
│                                                                             │
│   All layers maintain similar activation distribution!                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Batch Normalization

### The Problem and Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BATCH NORMALIZATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   THE PROBLEM: Internal Covariate Shift                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   As training progresses, the distribution of layer inputs changes │  │
│   │   because earlier layer parameters change.                         │  │
│   │                                                                     │  │
│   │   Layer 2 has to constantly adapt to new input distributions!      │  │
│   │                                                                     │  │
│   │   Epoch 1:     Epoch 10:    Epoch 100:                             │  │
│   │     ▲            ▲            ▲                                    │  │
│   │    ███          ███            ███                                 │  │
│   │   █████       █████          █████                                 │  │
│   │   ─────►      ────────►      ──────────►                           │  │
│   │   -5  5       -2  8          0  15                                 │  │
│   │                                                                     │  │
│   │   Distribution keeps shifting! Hard to train.                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   THE SOLUTION: Normalize activations within each mini-batch               │
│                                                                             │
│   BATCH NORM ALGORITHM:                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Given: Mini-batch of activations x = {x₁, x₂, ..., xₘ}          │  │
│   │                                                                     │  │
│   │   Step 1: Compute batch mean                                       │  │
│   │   μ = (1/m) Σᵢ xᵢ                                                  │  │
│   │                                                                     │  │
│   │   Step 2: Compute batch variance                                   │  │
│   │   σ² = (1/m) Σᵢ (xᵢ - μ)²                                         │  │
│   │                                                                     │  │
│   │   Step 3: Normalize                                                │  │
│   │   x̂ᵢ = (xᵢ - μ) / √(σ² + ε)                                       │  │
│   │                                                                     │  │
│   │   Step 4: Scale and shift (learnable parameters!)                  │  │
│   │   yᵢ = γ x̂ᵢ + β                                                   │  │
│   │                                                                     │  │
│   │   γ and β are learned during training                              │  │
│   │   Network can learn to undo normalization if needed!               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WHERE TO PUT BATCH NORM:                                                 │
│                                                                             │
│   Option 1 (original):     Option 2 (more common):                        │
│   FC → BN → ReLU           FC → ReLU → BN                                 │
│                                                                             │
│   For CNNs: Normalize per channel (compute μ,σ across H,W,N)              │
│                                                                             │
│   TRAINING vs. INFERENCE:                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   TRAINING:                                                        │  │
│   │   - Use batch statistics (μ_batch, σ_batch)                        │  │
│   │   - Keep running average of μ and σ for inference                  │  │
│   │                                                                     │  │
│   │   INFERENCE:                                                       │  │
│   │   - Use running average (μ_running, σ_running)                     │  │
│   │   - Deterministic output (no batch dependency)                     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   BENEFITS:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  ✓ Higher learning rates (more stable training)                    │  │
│   │  ✓ Less sensitive to initialization                                │  │
│   │  ✓ Regularization effect (each sample sees different batch stats) │  │
│   │  ✓ Enables training very deep networks                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dropout

### Dropout Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DROPOUT                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IDEA: Randomly "drop" neurons during training                            │
│                                                                             │
│   TRAINING (p=0.5):                                                        │
│                                                                             │
│   Without Dropout:              With Dropout:                              │
│                                                                             │
│   Input     Hidden    Output    Input     Hidden    Output                 │
│     ○─────────○─────────○         ○─────────○─────────○                    │
│     │    ╲    │    ╲    │         │    ╲    ╳    ╲    │                    │
│     ○─────────○─────────○         ○─────────╳─────────○                    │
│     │    ╲    │    ╲    │         │    ╲    │    ╲    │  (╳ = dropped)     │
│     ○─────────○─────────○         ○─────────○─────────○                    │
│     │    ╲    │    ╲    │         │    ╲    ╳    ╲    │                    │
│     ○─────────○─────────○         ○─────────╳─────────○                    │
│                                                                             │
│   All neurons active            Random 50% dropped each forward pass       │
│                                                                             │
│                                                                             │
│   IMPLEMENTATION:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   # Training: drop neurons randomly                                │  │
│   │   mask = np.random.rand(*h.shape) > p  # p = dropout probability  │  │
│   │   h = h * mask                          # Zero out dropped neurons │  │
│   │                                                                     │  │
│   │   # Test: scale outputs to match expected value                    │  │
│   │   h = h * (1 - p)                                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   INVERTED DROPOUT (preferred):                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   # Training: drop and scale                                       │  │
│   │   mask = np.random.rand(*h.shape) > p                              │  │
│   │   h = h * mask / (1 - p)  # Scale up to maintain expected value   │  │
│   │                                                                     │  │
│   │   # Test: do nothing!                                              │  │
│   │   h = h                                                            │  │
│   │                                                                     │  │
│   │   Benefit: Test code is simpler, faster                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                                                                             │
│   WHY DROPOUT WORKS:                                                       │
│                                                                             │
│   1. PREVENTS CO-ADAPTATION:                                               │
│      ┌───────────────────────────────────────────────────────────────┐    │
│      │  Without dropout: Neurons can "rely" on specific partners     │    │
│      │  With dropout: Each neuron must be independently useful       │    │
│      └───────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   2. ENSEMBLE EFFECT:                                                      │
│      ┌───────────────────────────────────────────────────────────────┐    │
│      │  Each forward pass = different subnetwork                     │    │
│      │  2^n possible networks with n neurons                         │    │
│      │  Test time = approximate ensemble averaging                    │    │
│      └───────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   TYPICAL VALUES:                                                          │
│   - p = 0.5 for fully connected layers                                    │
│   - p = 0.2-0.3 for convolutional layers (or don't use)                  │
│   - Never apply to output layer                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Advanced Optimizers

### Optimizer Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ADVANCED OPTIMIZERS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SGD WITH MOMENTUM:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   v = momentum * v - learning_rate * gradient                      │  │
│   │   w = w + v                                                        │  │
│   │                                                                     │  │
│   │   Intuition: Ball rolling down hill with inertia                   │  │
│   │                                                                     │  │
│   │   SGD path:           Momentum path:                               │  │
│   │   ┌─────────────┐     ┌─────────────┐                              │  │
│   │   │    ↓↗↓↗↓    │     │    ↓        │                              │  │
│   │   │   ↗↓↗↓↗↓   │     │     ↘       │                              │  │
│   │   │  ↓↗↓↗↓↗↓↗  │     │      ↘      │                              │  │
│   │   │   minimum   │     │    minimum  │                              │  │
│   │   └─────────────┘     └─────────────┘                              │  │
│   │   Oscillates!         Smooth descent                               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   RMSPROP:                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   cache = decay * cache + (1 - decay) * gradient²                  │  │
│   │   w = w - learning_rate * gradient / (√cache + ε)                  │  │
│   │                                                                     │  │
│   │   Intuition: Adapt learning rate per parameter                     │  │
│   │   - Parameters with large gradients: smaller effective LR         │  │
│   │   - Parameters with small gradients: larger effective LR          │  │
│   │                                                                     │  │
│   │   Typical: decay = 0.99, ε = 1e-8                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ADAM (Recommended Default):                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   # Combine momentum + RMSprop                                     │  │
│   │   m = β₁ * m + (1 - β₁) * gradient        # First moment (mean)   │  │
│   │   v = β₂ * v + (1 - β₂) * gradient²       # Second moment (var)   │  │
│   │                                                                     │  │
│   │   # Bias correction (important early in training)                  │  │
│   │   m_hat = m / (1 - β₁^t)                                           │  │
│   │   v_hat = v / (1 - β₂^t)                                           │  │
│   │                                                                     │  │
│   │   w = w - learning_rate * m_hat / (√v_hat + ε)                     │  │
│   │                                                                     │  │
│   │   Typical: β₁ = 0.9, β₂ = 0.999, ε = 1e-8, lr = 0.001             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   COMPARISON:                                                              │
│                                                                             │
│   Optimizer    Speed    Tuning    Best For                                 │
│   ──────────────────────────────────────────────────────────────────────   │
│   SGD          Slow     Hard      Final fine-tuning, when time available  │
│   Momentum     Medium   Medium    General training                         │
│   RMSprop      Fast     Easy      RNNs, non-stationary problems           │
│   Adam         Fast     Easy      Default choice, works well generally    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Learning Rate Schedules

### Common Schedules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LEARNING RATE SCHEDULES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHY DECAY LEARNING RATE?                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Early training: Large steps to find good region quickly           │  │
│   │  Late training: Small steps to fine-tune within good region        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                                                                             │
│   STEP DECAY:                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   lr │▔▔▔▔▔▔▔▔▔┐                                                   │  │
│   │      │         │                                                    │  │
│   │      │         └────────┐                                           │  │
│   │      │                  │                                           │  │
│   │      │                  └────────┐                                  │  │
│   │      │                           │                                  │  │
│   │      └───────────────────────────┴────► epoch                      │  │
│   │                                                                     │  │
│   │   lr = initial_lr * decay^(epoch // step_size)                     │  │
│   │   Example: decay by 0.1 every 30 epochs                            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   EXPONENTIAL DECAY:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   lr │╲                                                             │  │
│   │      │ ╲                                                            │  │
│   │      │  ╲                                                           │  │
│   │      │   ╲_                                                         │  │
│   │      │     ╲___                                                     │  │
│   │      │         ╲_______                                             │  │
│   │      └─────────────────────────────────► epoch                     │  │
│   │                                                                     │  │
│   │   lr = initial_lr * e^(-decay_rate * epoch)                        │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   COSINE ANNEALING:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   lr │╲                                                             │  │
│   │      │ ╲                                                            │  │
│   │      │  ╲                                                           │  │
│   │      │   ╲                                                          │  │
│   │      │    ╲___                                                      │  │
│   │      │        ╲___                                                  │  │
│   │      │            ╲____                                             │  │
│   │      └──────────────────╲───────────► epoch                        │  │
│   │                                                                     │  │
│   │   lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π * epoch/T))       │  │
│   │                                                                     │  │
│   │   Smooth decay, often works best!                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WARMUP + DECAY:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   lr │    /▔▔╲                                                      │  │
│   │      │   /    ╲                                                     │  │
│   │      │  /      ╲                                                    │  │
│   │      │ /        ╲___                                                │  │
│   │      │/             ╲___                                            │  │
│   │      └──────────────────────────────► epoch                        │  │
│   │      warmup  peak   decay                                          │  │
│   │                                                                     │  │
│   │   First few epochs: linearly increase LR (warmup)                  │  │
│   │   Rest: decay normally                                             │  │
│   │                                                                     │  │
│   │   Helps with large batch sizes / unstable early training           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Coding Exercises

### Exercise 1: Convolutional Layer

```python
#==============================================================================
# EXERCISE 1: IMPLEMENT CONVOLUTIONAL LAYER
#==============================================================================

import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """
    Forward pass for a convolutional layer.

    Args:
        x: Input data of shape (N, C, H, W)
        w: Filter weights of shape (F, C, HH, WW)
        b: Biases of shape (F,)
        conv_param: Dictionary with 'stride' and 'pad'

    Returns:
        out: Output data of shape (N, F, H', W')
        cache: (x, w, b, conv_param) for backward pass
    """
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Calculate output dimensions
    H_out = (H + 2 * pad - HH) // stride + 1
    W_out = (W + 2 * pad - WW) // stride + 1

    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Initialize output
    out = np.zeros((N, F, H_out, W_out))

    #===========================================================================
    # TODO: Implement the forward pass
    # Hint: Use nested loops over N, F, H_out, W_out
    # For each output position, compute dot product with filter
    #===========================================================================

    for n in range(N):  # For each image in batch
        for f in range(F):  # For each filter
            for i in range(H_out):  # For each output row
                for j in range(W_out):  # For each output column
                    # Extract the receptive field
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # Compute convolution (element-wise multiply and sum)
                    out[n, f, i, j] = np.sum(receptive_field * w[f]) + b[f]

    #===========================================================================

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    Backward pass for a convolutional layer.

    Args:
        dout: Upstream gradients of shape (N, F, H', W')
        cache: Tuple of (x, w, b, conv_param)

    Returns:
        dx: Gradient with respect to x
        dw: Gradient with respect to w
        db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Initialize gradients
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    #===========================================================================
    # TODO: Implement the backward pass
    #===========================================================================

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Gradient w.r.t. weights: input * upstream gradient
                    dw[f] += x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]

                    # Gradient w.r.t. input: weights * upstream gradient
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]

    # Gradient w.r.t. bias: sum over all positions
    db = np.sum(dout, axis=(0, 2, 3))

    # Remove padding from dx
    if pad > 0:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_padded

    #===========================================================================

    return dx, dw, db


#==============================================================================
# EXERCISE 2: IMPLEMENT MAX POOLING
#==============================================================================

def max_pool_forward_naive(x, pool_param):
    """
    Forward pass for max pooling layer.

    Args:
        x: Input data of shape (N, C, H, W)
        pool_param: Dictionary with 'pool_height', 'pool_width', 'stride'

    Returns:
        out: Output data of shape (N, C, H', W')
        cache: (x, pool_param) for backward pass
    """
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = (H - pool_h) // stride + 1
    W_out = (W - pool_w) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    #===========================================================================
    # TODO: Implement max pooling forward pass
    #===========================================================================

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_h
                    w_start = j * stride
                    w_end = w_start + pool_w

                    pool_region = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(pool_region)

    #===========================================================================

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    Backward pass for max pooling layer.

    Args:
        dout: Upstream gradients of shape (N, C, H', W')
        cache: Tuple of (x, pool_param)

    Returns:
        dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)

    #===========================================================================
    # TODO: Implement max pooling backward pass
    # Gradient flows only to the position that was the maximum
    #===========================================================================

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_h
                    w_start = j * stride
                    w_end = w_start + pool_w

                    pool_region = x[n, c, h_start:h_end, w_start:w_end]

                    # Find position of max value
                    max_val = np.max(pool_region)
                    mask = (pool_region == max_val)

                    # Route gradient to max position
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]

    #===========================================================================

    return dx


#==============================================================================
# EXERCISE 3: IMPLEMENT BATCH NORMALIZATION
#==============================================================================

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    Args:
        x: Input data of shape (N, D)
        gamma: Scale parameter of shape (D,)
        beta: Shift parameter of shape (D,)
        bn_param: Dictionary with 'mode', 'eps', 'momentum', 'running_mean', 'running_var'

    Returns:
        out: Normalized output of shape (N, D)
        cache: Values needed for backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == 'train':
        #=======================================================================
        # TODO: Implement training-time batch normalization
        #=======================================================================

        # Step 1: Compute batch mean
        mu = np.mean(x, axis=0)

        # Step 2: Compute batch variance
        var = np.var(x, axis=0)

        # Step 3: Normalize
        x_hat = (x - mu) / np.sqrt(var + eps)

        # Step 4: Scale and shift
        out = gamma * x_hat + beta

        # Update running statistics
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        # Cache for backward pass
        cache = (x, x_hat, mu, var, gamma, eps)

        #=======================================================================

    elif mode == 'test':
        #=======================================================================
        # TODO: Implement test-time batch normalization
        # Use running statistics instead of batch statistics
        #=======================================================================

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta

        #=======================================================================

    else:
        raise ValueError(f'Invalid mode "{mode}"')

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Args:
        dout: Upstream gradients of shape (N, D)
        cache: Values from forward pass

    Returns:
        dx: Gradient with respect to x
        dgamma: Gradient with respect to gamma
        dbeta: Gradient with respect to beta
    """
    x, x_hat, mu, var, gamma, eps = cache
    N, D = dout.shape

    #===========================================================================
    # TODO: Implement backward pass
    # This is tricky! Follow the computational graph carefully.
    #===========================================================================

    # Gradient of beta: sum of upstream gradients
    dbeta = np.sum(dout, axis=0)

    # Gradient of gamma: sum of (upstream * normalized input)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Gradient of x_hat
    dx_hat = dout * gamma

    # Gradient through normalization (this is the tricky part)
    std_inv = 1.0 / np.sqrt(var + eps)

    dx = (1.0 / N) * std_inv * (
        N * dx_hat
        - np.sum(dx_hat, axis=0)
        - x_hat * np.sum(dx_hat * x_hat, axis=0)
    )

    #===========================================================================

    return dx, dgamma, dbeta


#==============================================================================
# EXERCISE 4: IMPLEMENT DROPOUT
#==============================================================================

def dropout_forward(x, dropout_param):
    """
    Forward pass for inverted dropout.

    Args:
        x: Input data of any shape
        dropout_param: Dictionary with 'p' (dropout probability) and 'mode'

    Returns:
        out: Output data of same shape as x
        cache: (dropout_param, mask) for backward pass
    """
    p = dropout_param['p']
    mode = dropout_param['mode']

    mask = None
    out = None

    if mode == 'train':
        #=======================================================================
        # TODO: Implement training-time inverted dropout
        # 1. Create random mask (keep with probability 1-p)
        # 2. Apply mask and scale by 1/(1-p)
        #=======================================================================

        mask = (np.random.rand(*x.shape) >= p) / (1 - p)
        out = x * mask

        #=======================================================================

    elif mode == 'test':
        #=======================================================================
        # TODO: Implement test-time dropout (just pass through)
        #=======================================================================

        out = x

        #=======================================================================

    cache = (dropout_param, mask)
    return out, cache


def dropout_backward(dout, cache):
    """
    Backward pass for inverted dropout.

    Args:
        dout: Upstream gradients of any shape
        cache: (dropout_param, mask) from forward pass

    Returns:
        dx: Gradient with respect to x
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        #=======================================================================
        # TODO: Implement backward pass
        # Gradient flows only through kept neurons
        #=======================================================================

        dx = dout * mask

        #=======================================================================

    elif mode == 'test':
        dx = dout

    return dx
```

### Exercise 5: Complete CNN

```python
#==============================================================================
# EXERCISE 5: BUILD A COMPLETE CNN
#==============================================================================

import numpy as np

class ConvNet:
    """
    A convolutional neural network with the following architecture:

    conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data with shape (N, C, H, W).
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Args:
            input_dim: Tuple (C, H, W) giving size of input images
            num_filters: Number of filters in conv layer
            filter_size: Size of filters in conv layer
            hidden_dim: Number of units in fully-connected hidden layer
            num_classes: Number of output classes
            weight_scale: Scale for weight initialization
            reg: L2 regularization strength
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim

        #=======================================================================
        # TODO: Initialize weights and biases
        # W1: Conv filter weights (num_filters, C, filter_size, filter_size)
        # b1: Conv biases (num_filters,)
        # W2: FC weights (num_filters * H/2 * W/2, hidden_dim)
        # b2: FC biases (hidden_dim,)
        # W3: FC weights (hidden_dim, num_classes)
        # b3: FC biases (num_classes,)
        #=======================================================================

        # Conv layer
        self.params['W1'] = weight_scale * np.random.randn(
            num_filters, C, filter_size, filter_size
        ).astype(dtype)
        self.params['b1'] = np.zeros(num_filters, dtype=dtype)

        # After conv (same size due to padding) and max pool (halved)
        H_pool = H // 2
        W_pool = W // 2
        fc_input_dim = num_filters * H_pool * W_pool

        # First FC layer
        self.params['W2'] = weight_scale * np.random.randn(
            fc_input_dim, hidden_dim
        ).astype(dtype)
        self.params['b2'] = np.zeros(hidden_dim, dtype=dtype)

        # Output FC layer
        self.params['W3'] = weight_scale * np.random.randn(
            hidden_dim, num_classes
        ).astype(dtype)
        self.params['b3'] = np.zeros(num_classes, dtype=dtype)

        #=======================================================================

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the network.

        Args:
            X: Array of input data of shape (N, C, H, W)
            y: Array of labels of shape (N,). If None, return scores only.

        Returns:
            If y is None: scores of shape (N, num_classes)
            If y is not None: tuple of (loss, grads)
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Conv params (same padding to preserve size)
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        #=======================================================================
        # FORWARD PASS
        #=======================================================================

        # Conv - ReLU - Pool
        conv_out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
        relu1_out = np.maximum(0, conv_out)
        pool_out, pool_cache = max_pool_forward_naive(relu1_out, pool_param)

        # Flatten for FC layers
        N = X.shape[0]
        flat_out = pool_out.reshape(N, -1)

        # FC - ReLU
        fc1_out = flat_out @ W2 + b2
        relu2_out = np.maximum(0, fc1_out)

        # FC (scores)
        scores = relu2_out @ W3 + b3

        if y is None:
            return scores

        #=======================================================================
        # COMPUTE LOSS
        #=======================================================================

        # Softmax loss
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_log_probs = -np.log(probs[np.arange(N), y])
        data_loss = np.sum(correct_log_probs) / N

        # L2 regularization
        reg_loss = 0.5 * self.reg * (
            np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)
        )

        loss = data_loss + reg_loss

        #=======================================================================
        # BACKWARD PASS
        #=======================================================================

        grads = {}

        # Gradient of softmax loss
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        # Backprop through FC2
        grads['W3'] = relu2_out.T @ dscores + self.reg * W3
        grads['b3'] = np.sum(dscores, axis=0)
        drelu2 = dscores @ W3.T

        # Backprop through ReLU2
        dfc1 = drelu2 * (fc1_out > 0)

        # Backprop through FC1
        grads['W2'] = flat_out.T @ dfc1 + self.reg * W2
        grads['b2'] = np.sum(dfc1, axis=0)
        dflat = dfc1 @ W2.T

        # Reshape for pool backprop
        dpool = dflat.reshape(pool_out.shape)

        # Backprop through pool
        drelu1 = max_pool_backward_naive(dpool, pool_cache)

        # Backprop through ReLU1
        dconv = drelu1 * (conv_out > 0)

        # Backprop through conv
        dx, dW1, db1 = conv_backward_naive(dconv, conv_cache)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1

        #=======================================================================

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              num_epochs=10, batch_size=100, verbose=True):
        """
        Train the network using SGD.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(num_epochs):
            # Shuffle training data
            perm = np.random.permutation(num_train)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(iterations_per_epoch):
                # Get mini-batch
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute loss and gradients
                loss, grads = self.loss(X_batch, y_batch)
                loss_history.append(loss)

                # Update parameters
                for param_name in self.params:
                    self.params[param_name] -= learning_rate * grads[param_name]

            # Decay learning rate
            learning_rate *= learning_rate_decay

            # Check accuracy
            train_acc = np.mean(self.predict(X) == y)
            val_acc = np.mean(self.predict(X_val) == y_val)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}: '
                      f'loss={loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, X):
        """Predict class labels for test data."""
        scores = self.loss(X)
        return np.argmax(scores, axis=1)
```

---

## Business Applications

### Production CNN System

```python
#==============================================================================
# BUSINESS APPLICATION: Image Quality Assessment System
#==============================================================================

import numpy as np
from typing import List, Dict, Tuple
import json

class ImageQualityAssessor:
    """
    Production system for automated image quality assessment.

    Use Cases:
    - E-commerce: Reject low-quality product photos
    - Social media: Flag blurry/dark images before posting
    - Manufacturing: Detect defective products from camera images
    - Healthcare: Ensure medical images meet quality standards
    """

    QUALITY_LEVELS = ['excellent', 'good', 'acceptable', 'poor', 'rejected']

    def __init__(self):
        """Initialize the quality assessment model."""
        self.model = None
        self.preprocessing_stats = None
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.75,
            'acceptable': 0.5,
            'poor': 0.25
        }

    def extract_quality_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract handcrafted features for quality assessment.

        In a real system, these would be learned by the CNN, but
        understanding what makes an image "high quality" is valuable.
        """
        features = []

        # 1. Brightness: Average pixel intensity
        brightness = np.mean(image)
        features.append(brightness / 255.0)

        # 2. Contrast: Standard deviation of pixel intensities
        contrast = np.std(image)
        features.append(contrast / 128.0)

        # 3. Sharpness: Gradient magnitude (simple Laplacian approximation)
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Simple gradient using differences
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        sharpness = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
        features.append(sharpness / 50.0)

        # 4. Color saturation (if color image)
        if len(image.shape) == 3:
            # Simple saturation: max - min across channels
            saturation = np.mean(np.max(image, axis=2) - np.min(image, axis=2))
            features.append(saturation / 128.0)
        else:
            features.append(0.0)

        # 5. Noise estimate: local variance in smooth regions
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        # Compute local variance in 4x4 patches
        h, w = gray.shape
        patch_vars = []
        for i in range(0, h-4, 4):
            for j in range(0, w-4, 4):
                patch = gray[i:i+4, j:j+4]
                patch_vars.append(np.var(patch))

        noise = np.percentile(patch_vars, 10)  # Low percentile = smooth region variance
        features.append(1.0 - min(noise / 100.0, 1.0))  # Invert: high noise = low quality

        return np.array(features)

    def assess_single(self, image: np.ndarray) -> Dict:
        """
        Assess quality of a single image.

        Args:
            image: Input image (H, W, 3) or (H, W)

        Returns:
            Dictionary with quality assessment results
        """
        features = self.extract_quality_features(image)

        # Simple scoring: weighted combination of features
        # In production, this would be a trained CNN
        weights = np.array([0.2, 0.25, 0.3, 0.15, 0.1])
        quality_score = np.sum(features * weights)

        # Determine quality level
        if quality_score >= self.quality_thresholds['excellent']:
            quality_level = 'excellent'
        elif quality_score >= self.quality_thresholds['good']:
            quality_level = 'good'
        elif quality_score >= self.quality_thresholds['acceptable']:
            quality_level = 'acceptable'
        elif quality_score >= self.quality_thresholds['poor']:
            quality_level = 'poor'
        else:
            quality_level = 'rejected'

        # Generate recommendations
        recommendations = []
        if features[0] < 0.3:
            recommendations.append("Image is too dark. Increase lighting or exposure.")
        elif features[0] > 0.85:
            recommendations.append("Image is overexposed. Reduce lighting or exposure.")

        if features[1] < 0.2:
            recommendations.append("Image has low contrast. Consider adjusting levels.")

        if features[2] < 0.3:
            recommendations.append("Image appears blurry. Ensure camera is focused.")

        if features[4] < 0.5:
            recommendations.append("Image has visible noise. Use better lighting or lower ISO.")

        return {
            'quality_score': float(quality_score),
            'quality_level': quality_level,
            'is_acceptable': quality_level in ['excellent', 'good', 'acceptable'],
            'feature_scores': {
                'brightness': float(features[0]),
                'contrast': float(features[1]),
                'sharpness': float(features[2]),
                'color_saturation': float(features[3]),
                'noise_level': float(features[4])
            },
            'recommendations': recommendations
        }

    def assess_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Assess quality of multiple images.

        Args:
            images: List of input images

        Returns:
            List of quality assessment results
        """
        return [self.assess_single(img) for img in images]

    def filter_by_quality(self, images: List[np.ndarray],
                          min_level: str = 'acceptable') -> Tuple[List, List, List]:
        """
        Filter images by quality level.

        Args:
            images: List of input images
            min_level: Minimum acceptable quality level

        Returns:
            Tuple of (accepted_images, accepted_indices, assessment_results)
        """
        level_order = {level: i for i, level in enumerate(self.QUALITY_LEVELS)}
        min_level_idx = level_order[min_level]

        accepted_images = []
        accepted_indices = []
        all_results = []

        for i, image in enumerate(images):
            result = self.assess_single(image)
            all_results.append(result)

            if level_order[result['quality_level']] <= min_level_idx:
                accepted_images.append(image)
                accepted_indices.append(i)

        return accepted_images, accepted_indices, all_results

    def generate_report(self, results: List[Dict]) -> Dict:
        """
        Generate summary report for batch of images.

        Args:
            results: List of assessment results

        Returns:
            Summary report dictionary
        """
        total = len(results)
        if total == 0:
            return {'error': 'No images to analyze'}

        # Count by level
        level_counts = {level: 0 for level in self.QUALITY_LEVELS}
        for r in results:
            level_counts[r['quality_level']] += 1

        # Average scores
        avg_scores = {}
        for feature in ['brightness', 'contrast', 'sharpness', 'color_saturation', 'noise_level']:
            avg_scores[feature] = np.mean([r['feature_scores'][feature] for r in results])

        # Common issues
        all_recommendations = []
        for r in results:
            all_recommendations.extend(r['recommendations'])

        issue_counts = {}
        for rec in all_recommendations:
            issue_counts[rec] = issue_counts.get(rec, 0) + 1

        return {
            'total_images': total,
            'acceptance_rate': sum(level_counts[l] for l in ['excellent', 'good', 'acceptable']) / total,
            'quality_distribution': level_counts,
            'average_scores': avg_scores,
            'common_issues': sorted(issue_counts.items(), key=lambda x: -x[1])[:5]
        }


# Demo
def demo_quality_assessor():
    """Demonstrate the quality assessment system."""

    assessor = ImageQualityAssessor()

    # Generate test images with different qualities
    np.random.seed(42)

    # Good quality image (good brightness, contrast, sharp)
    good_image = np.random.randint(40, 200, (64, 64, 3)).astype(np.float32)
    # Add some structure for sharpness
    good_image[20:40, 20:40] = 180

    # Bad quality image (dark, low contrast)
    bad_image = np.random.randint(10, 50, (64, 64, 3)).astype(np.float32)

    # Assess images
    print("=" * 60)
    print("IMAGE QUALITY ASSESSMENT DEMO")
    print("=" * 60)

    print("\nGood Image Assessment:")
    result = assessor.assess_single(good_image)
    print(f"  Quality Score: {result['quality_score']:.3f}")
    print(f"  Quality Level: {result['quality_level']}")
    print(f"  Acceptable: {result['is_acceptable']}")

    print("\nBad Image Assessment:")
    result = assessor.assess_single(bad_image)
    print(f"  Quality Score: {result['quality_score']:.3f}")
    print(f"  Quality Level: {result['quality_level']}")
    print(f"  Acceptable: {result['is_acceptable']}")
    print(f"  Recommendations: {result['recommendations']}")

    # Batch processing
    images = [good_image, bad_image] * 5
    accepted, indices, results = assessor.filter_by_quality(images, min_level='acceptable')
    print(f"\nBatch Processing: {len(accepted)}/{len(images)} images accepted")

    # Generate report
    report = assessor.generate_report(results)
    print(f"\nBatch Report:")
    print(f"  Acceptance Rate: {report['acceptance_rate']:.1%}")
    print(f"  Quality Distribution: {report['quality_distribution']}")


if __name__ == '__main__':
    demo_quality_assessor()
```

---

## Summary: Week 3-4 Checklist

### Concepts You Should Understand
- [ ] Why CNNs work better than FC networks for images
- [ ] Convolution operation (filter, stride, padding)
- [ ] Parameter sharing and local connectivity
- [ ] Pooling layers (max pooling, average pooling)
- [ ] CNN architecture patterns (Conv-ReLU-Pool)
- [ ] Famous architectures (LeNet, AlexNet, VGG, ResNet)
- [ ] Data preprocessing (mean subtraction, normalization)
- [ ] Weight initialization (Xavier, He)
- [ ] Batch normalization (training vs. inference)
- [ ] Dropout (inverted dropout)
- [ ] Advanced optimizers (Momentum, RMSprop, Adam)
- [ ] Learning rate schedules (step, exponential, cosine)

### Skills You Should Have
- [ ] Implement convolution forward and backward pass
- [ ] Implement max pooling forward and backward pass
- [ ] Implement batch normalization
- [ ] Implement dropout
- [ ] Build a complete CNN from scratch
- [ ] Choose appropriate hyperparameters
- [ ] Debug training issues (learning curves, gradient flow)

### Key Formulas
```
Output Size: O = (W - F + 2P) / S + 1

BatchNorm: y = γ * (x - μ) / √(σ² + ε) + β

He Init: W ~ N(0, √(2/n_in))

Adam: θ = θ - α * m̂ / (√v̂ + ε)
```

### Next Steps
After completing Week 3-4, you're ready for:
- **Week 5-6**: Self-Supervised Learning and RNNs
- Learning representations without labels
- Processing sequential data
