"""
Week 3-4 Visualizer: CNNs and Training
Interactive ASCII visualizations for convolutional neural networks.
Run: python week_03_04_visualizer.py
"""

import time
import os
import random

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause(msg="Press Enter to continue..."):
    input(f"\n{msg}")

def print_header(title):
    clear_screen()
    width = 60
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
    print()

# ============================================================
# VISUALIZATION 1: Convolution Operation
# ============================================================

def visualize_convolution():
    print_header("VISUALIZATION 1: Convolution Operation")

    print("""
A convolution slides a small filter (kernel) over an image,
computing dot products at each position.

Why convolutions?
  - Weight sharing (same filter everywhere)
  - Translation invariance
  - Local connectivity
  - Far fewer parameters than fully connected

    Image                    Filter (3x3)
    ┌─────────────────┐      ┌─────────┐
    │ 1  2  3  4  5   │      │ 1  0  1 │
    │ 6  7  8  9  10  │  *   │ 0  1  0 │
    │ 11 12 13 14 15  │      │ 1  0  1 │
    │ 16 17 18 19 20  │      └─────────┘
    │ 21 22 23 24 25  │
    └─────────────────┘
    """)
    pause()

    # Step-by-step convolution
    steps = [
        ("""
Step 1: Position filter at top-left
═══════════════════════════════════════════════════════════

    Image                      Filter
    ┌─────────────────┐        ┌─────────┐
    │[1  2  3] 4  5   │        │ 1  0  1 │
    │[6  7  8] 9  10  │   *    │ 0  1  0 │
    │[11 12 13]14 15  │        │ 1  0  1 │
    │ 16 17 18 19 20  │        └─────────┘
    │ 21 22 23 24 25  │
    └─────────────────┘

    Computation:
    (1×1)+(2×0)+(3×1) + (6×0)+(7×1)+(8×0) + (11×1)+(12×0)+(13×1)
    = 1 + 0 + 3 + 0 + 7 + 0 + 11 + 0 + 13
    = 35

    Output[0,0] = 35
""", 35),
        ("""
Step 2: Slide filter right (stride=1)
═══════════════════════════════════════════════════════════

    Image                      Filter
    ┌─────────────────┐        ┌─────────┐
    │ 1 [2  3  4] 5   │        │ 1  0  1 │
    │ 6 [7  8  9] 10  │   *    │ 0  1  0 │
    │ 11[12 13 14]15  │        │ 1  0  1 │
    │ 16 17 18 19 20  │        └─────────┘
    │ 21 22 23 24 25  │
    └─────────────────┘

    = (2×1)+(3×0)+(4×1) + (7×0)+(8×1)+(9×0) + (12×1)+(13×0)+(14×1)
    = 2 + 4 + 8 + 12 + 14
    = 40

    Output[0,1] = 40
""", 40),
        ("""
Step 3: Continue sliding...
═══════════════════════════════════════════════════════════

    Output feature map (3x3):

    ┌──────────────┐
    │ 35  40  45  │  ← First row done
    │ 60  65  70  │
    │ 85  90  95  │
    └──────────────┘

    Note: 5x5 image with 3x3 filter → 3x3 output
          Output size = (Input - Filter + 1) = 5 - 3 + 1 = 3
""", None),
    ]

    for description, result in steps:
        clear_screen()
        print("=" * 60)
        print(f"{'CONVOLUTION STEP-BY-STEP':^60}")
        print("=" * 60)
        print(description)
        time.sleep(2)

    pause()

# ============================================================
# VISUALIZATION 2: Padding and Stride
# ============================================================

def visualize_padding_stride():
    print_header("VISUALIZATION 2: Padding and Stride")

    print("""
PADDING: Adding zeros around the input
═══════════════════════════════════════════════════════════

Why padding?
  - Preserve spatial dimensions
  - Allow filter to see edge pixels properly

No Padding ("valid"):           With Padding ("same"):
┌─────────────┐                ┌─────────────────┐
│ 1  2  3  4  │                │ 0  0  0  0  0  0│
│ 5  6  7  8  │  3x3 filter    │ 0  1  2  3  4  0│
│ 9 10 11 12  │ ─────────────▶ │ 0  5  6  7  8  0│
│13 14 15 16  │                │ 0  9 10 11 12  0│
└─────────────┘                │ 0 13 14 15 16  0│
    4x4 input                  │ 0  0  0  0  0  0│
                               └─────────────────┘
    Output: 2x2                     6x6 padded
    (shrinks!)                      Output: 4x4 (preserved!)
    """)
    pause()

    print("""
STRIDE: How many pixels to skip between positions
═══════════════════════════════════════════════════════════

Stride = 1 (default):           Stride = 2:
┌─────────────┐                ┌─────────────┐
│[█ █]█  █    │                │[█ █]█ [█ █] │
│[█ █]█  █    │                │[█ █]█ [█ █] │
│ █  █  █  █  │                │ █  █  █  █  │
│ █  █  █  █  │                │[█ █]█ [█ █] │
└─────────────┘                │[█ █]█ [█ █] │
                               └─────────────┘
Position 1 → Position 2        Position 1 → Position 2
(move 1 pixel)                 (move 2 pixels)

Stride 1: all positions        Stride 2: skip positions
Output: large                  Output: smaller (downsampled)

Formula:
  Output = (Input - Filter + 2*Padding) / Stride + 1
    """)
    pause()

    print("""
EXAMPLE CALCULATIONS
═══════════════════════════════════════════════════════════

Input: 32×32, Filter: 3×3

Case 1: No padding, Stride 1
  Output = (32 - 3 + 0) / 1 + 1 = 30×30

Case 2: Padding 1, Stride 1 ("same")
  Output = (32 - 3 + 2) / 1 + 1 = 32×32 ✓

Case 3: No padding, Stride 2
  Output = (32 - 3 + 0) / 2 + 1 = 15×15

Case 4: Padding 1, Stride 2
  Output = (32 - 3 + 2) / 2 + 1 = 16×16

┌────────────────────────────────────────────────────────┐
│ Rule of thumb:                                         │
│   - Use padding="same" to preserve dimensions          │
│   - Use stride=2 to downsample (instead of pooling)    │
└────────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 3: Pooling Layers
# ============================================================

def visualize_pooling():
    print_header("VISUALIZATION 3: Pooling Layers")

    print("""
Pooling reduces spatial dimensions and provides:
  - Translation invariance
  - Reduced computation
  - Larger receptive field

MAX POOLING (2×2, stride 2)
═══════════════════════════════════════════════════════════

Input (4×4):              Output (2×2):
┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 4 │         ┌───┬───┐
├───┼───┼───┼───┤         │ 6 │ 8 │  (max of each 2×2 region)
│ 5 │ 6 │ 7 │ 8 │    →    ├───┼───┤
├───┼───┼───┼───┤         │14 │16 │
│ 9 │10 │11 │12 │         └───┴───┘
├───┼───┼───┼───┤
│13 │14 │15 │16 │
└───┴───┴───┴───┘

Top-left region: max(1,3,5,6) = 6
Top-right region: max(2,4,7,8) = 8
Bottom-left region: max(9,10,13,14) = 14
Bottom-right region: max(11,12,15,16) = 16
    """)
    pause()

    print("""
AVERAGE POOLING (2×2, stride 2)
═══════════════════════════════════════════════════════════

Input (4×4):              Output (2×2):
┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 4 │         ┌─────┬─────┐
├───┼───┼───┼───┤         │ 3.75│ 5.25│  (avg of each region)
│ 5 │ 6 │ 7 │ 8 │    →    ├─────┼─────┤
├───┼───┼───┼───┤         │11.5 │13.5 │
│ 9 │10 │11 │12 │         └─────┴─────┘
├───┼───┼───┼───┤
│13 │14 │15 │16 │
└───┴───┴───┴───┘

Top-left: avg(1,3,5,6) = 15/4 = 3.75
    """)
    pause()

    print("""
MAX vs AVERAGE POOLING
═══════════════════════════════════════════════════════════

                    Max Pooling         Average Pooling
    ────────────────────────────────────────────────────
    Keeps:          Strongest signal    Overall signal
    Good for:       Feature detection   Smooth features
    Common use:     Hidden layers       Final layer (GAP)

GLOBAL AVERAGE POOLING (GAP)
═══════════════════════════════════════════════════════════

Takes entire feature map → single value per channel

    Feature Map (7×7)           Output
    ┌───────────────────┐
    │                   │
    │   (49 values)     │   →   [3.14]  (average of all 49)
    │                   │
    └───────────────────┘

Modern networks use GAP instead of flatten + FC layers:
  - Far fewer parameters
  - More robust to input size variations
  - Acts as regularization
    """)
    pause()

# ============================================================
# VISUALIZATION 4: CNN Architecture
# ============================================================

def visualize_cnn_architecture():
    print_header("VISUALIZATION 4: CNN Architecture")

    print("""
A typical CNN for image classification:

INPUT → [CONV → RELU → POOL] × N → FC → OUTPUT

    ┌───────────────────────────────────────────────────────────┐
    │                                                           │
    │   Input     Conv     Pool    Conv     Pool    FC   Output │
    │   Image                                                   │
    │                                                           │
    │   ┌───┐    ┌───┐    ┌───┐   ┌───┐    ┌───┐   ┌─┐   ┌─┐   │
    │   │   │    │   │    │   │   │   │    │   │   │ │   │ │   │
    │   │   │ →  │   │ →  │   │ → │   │ →  │   │ → │ │ → │ │   │
    │   │   │    │   │    │   │   │   │    │   │   │ │   │ │   │
    │   │   │    │   │    │   │   │   │    │   │   │ │   │ │   │
    │   └───┘    └───┘    └───┘   └───┘    └───┘   └─┘   └─┘   │
    │                                                           │
    │   224×224  224×224  112×112  112×112  56×56   4096   1000 │
    │   ×3       ×64      ×64      ×128     ×128                │
    │                                                           │
    └───────────────────────────────────────────────────────────┘

    Spatial:   Large → → → → → → → → → → → → → Small
    Channels:  Few  → → → → → → → → → → → → →  Many
    """)
    pause()

    print("""
WHAT EACH LAYER LEARNS
═══════════════════════════════════════════════════════════

Layer 1 (Early):     Layer 3 (Mid):       Layer 5 (Deep):
Edges, colors        Textures, parts      Objects, faces

    ┌───┐ ─────      ┌─────┐              ┌─────────┐
    │ / │ edge       │░░░░░│ texture      │  ◠ ◠    │
    └───┘            │░░░░░│              │   ○     │
    ┌───┐ ─────      └─────┘              │  ╰─╯    │
    │───│ edge       ┌─────┐              └─────────┘
    └───┘            │  ◠  │ eye           face
    ┌───┐            │ ⬤   │
    │░░░│ color      └─────┘
    └───┘

Hierarchy: Simple → Complex → Semantic
    """)
    pause()

    print("""
RECEPTIVE FIELD
═══════════════════════════════════════════════════════════

The receptive field is the region of input that affects a neuron.

    Layer 1:        Layer 2:           Layer 3:
    3×3 RF          5×5 RF             7×7 RF

    ┌─┬─┬─┐        ┌─┬─┬─┬─┬─┐       ┌─┬─┬─┬─┬─┬─┬─┐
    │█│█│█│        │█│█│█│█│█│       │█│█│█│█│█│█│█│
    ├─┼─┼─┤        ├─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┼─┼─┤
    │█│█│█│        │█│█│█│█│█│       │█│█│█│█│█│█│█│
    ├─┼─┼─┤        ├─┼─┼─┼─┼─┤       │ │...│ │
    │█│█│█│        │█│█│█│█│█│       ├─┼─┼─┼─┼─┼─┼─┤
    └─┴─┴─┘        ├─┼─┼─┼─┼─┤       │█│█│█│█│█│█│█│
                   │█│█│█│█│█│       └─┴─┴─┴─┴─┴─┴─┘
                   ├─┼─┼─┼─┼─┤
                   │█│█│█│█│█│
                   └─┴─┴─┴─┴─┘

    Each layer increases the receptive field!
    Deep layers "see" more of the input image.
    """)
    pause()

# ============================================================
# VISUALIZATION 5: Batch Normalization
# ============================================================

def visualize_batch_norm():
    print_header("VISUALIZATION 5: Batch Normalization")

    print("""
Problem: Internal covariate shift
═══════════════════════════════════════════════════════════

As training progresses, the distribution of layer inputs changes.
This slows training because each layer must adapt to shifting inputs.

Before Batch Norm:               After Batch Norm:
    Epoch 1     Epoch 100           Epoch 1     Epoch 100
    ┌─────┐     ┌─────┐             ┌─────┐     ┌─────┐
    │  ∩  │     │    ∩│             │  ∩  │     │  ∩  │
    │ ╱ ╲ │     │   ╱╲│             │ ╱ ╲ │     │ ╱ ╲ │
    └─────┘     └─────┘             └─────┘     └─────┘
    Distribution shifts!            Distribution stable!
    """)
    pause()

    print("""
Batch Normalization Algorithm
═══════════════════════════════════════════════════════════

For each mini-batch of activations x:

Step 1: Compute batch statistics
        μ = mean(x)           # batch mean
        σ² = variance(x)      # batch variance

Step 2: Normalize
        x̂ = (x - μ) / √(σ² + ε)    # zero mean, unit variance

Step 3: Scale and shift (learnable!)
        y = γ * x̂ + β         # restore representation power

    Before:              After normalize:        After scale/shift:
    ┌────────────┐       ┌────────────┐          ┌────────────┐
    │    ∩       │       │     ∩      │          │      ∩     │
    │   ╱│╲      │  →    │    ╱│╲     │    →     │     ╱│╲    │
    │  ╱ │ ╲     │       │   ╱ │ ╲    │          │    ╱ │ ╲   │
    └────────────┘       └────────────┘          └────────────┘
    Mean=5, Var=4        Mean=0, Var=1           Mean=β, Var=γ²
    """)
    pause()

    print("""
Why γ and β?
═══════════════════════════════════════════════════════════

If we only normalize, we might lose important information.

Example: ReLU after BatchNorm

    Without γ,β:                  With γ,β:
    ┌────────────┐               ┌────────────┐
    │     │ ╱    │               │        ╱   │
    │     │╱     │               │       ╱    │
    │─────0──────│               │──────0─────│
    │    ╱│      │               │     ╱      │
    └────────────┘               └────────────┘
    Half inputs → 0!             Network can learn
    (ReLU kills negatives)       optimal distribution

γ and β let the network LEARN the best distribution:
  - If γ=σ, β=μ → undo normalization
  - If γ=1, β=0 → keep normalized
  - Any value in between!
    """)
    pause()

    print("""
Batch Norm Benefits
═══════════════════════════════════════════════════════════

    ✓ Faster training (higher learning rates)
    ✓ Less sensitive to initialization
    ✓ Acts as regularization (noise from batch stats)
    ✓ Reduces internal covariate shift

    ┌────────────────────────────────────────────────────────┐
    │  Training curves:                                      │
    │                                                        │
    │  Loss │                                               │
    │       │╲                                              │
    │       │ ╲_____ Without BatchNorm                      │
    │       │  ╲                                            │
    │       │   ╲____________________________               │
    │       │    With BatchNorm (faster!)                   │
    │       └──────────────────────────────────── Epochs    │
    └────────────────────────────────────────────────────────┘

Where to put BatchNorm?
  - CONV → BatchNorm → ReLU → POOL  (common)
  - CONV → ReLU → BatchNorm → POOL  (also works)
    """)
    pause()

# ============================================================
# VISUALIZATION 6: Dropout
# ============================================================

def visualize_dropout():
    print_header("VISUALIZATION 6: Dropout Regularization")

    print("""
Dropout: Randomly "drop" neurons during training
═══════════════════════════════════════════════════════════

During each training step, each neuron has probability p
of being set to zero (typically p=0.5 for FC, p=0.1-0.3 for conv).

Training (with dropout):

    Input       Hidden        Hidden        Output
     ○───────────○─────────────○───────────○
     ○───────────✗─────────────○───────────○
     ○───────────○─────────────✗───────────○
     ○───────────✗─────────────○───────────○
     ○───────────○─────────────○───────────○

    ✗ = dropped (set to 0)

Different neurons dropped each batch!
    """)
    pause()

    print("""
Why Dropout Works
═══════════════════════════════════════════════════════════

1. PREVENTS CO-ADAPTATION
   Neurons can't rely on specific other neurons being present.
   Forces each neuron to be independently useful.

   Without dropout:          With dropout:
   ○══════○══════○          ○──────○──────○
   "I only work with Bob"   "I work with anyone"

2. ENSEMBLE EFFECT
   Training with dropout ≈ training many different networks
   Each dropout mask = different sub-network

   Epoch 1:    Epoch 2:    Epoch 3:
   ○──○──○    ○──✗──○    ○──○──✗
   ○──✗──○    ○──○──○    ○──✗──○
   ✗──○──○    ○──○──✗    ○──○──○

   Inference = average of all these networks!
    """)
    pause()

    print("""
Dropout at Test Time
═══════════════════════════════════════════════════════════

At test time, we use ALL neurons but SCALE outputs:

Training:    Expected activation = p × (activation) + (1-p) × 0
                                 = p × activation

Test time:   We use full activation, so multiply by p

Example (p=0.5):
    ┌────────────────────────────────────────────────────────┐
    │  Training: neuron outputs 4.0 (when active)            │
    │            Expected output = 0.5 × 4.0 = 2.0           │
    │                                                        │
    │  Testing:  neuron outputs 4.0 × 0.5 = 2.0              │
    │            Same expected value!                        │
    └────────────────────────────────────────────────────────┘

Or use "inverted dropout" (more common):
    Training: Scale UP active neurons by 1/p
    Testing:  Use unchanged values
    """)
    pause()

# ============================================================
# VISUALIZATION 7: Famous CNN Architectures
# ============================================================

def visualize_architectures():
    print_header("VISUALIZATION 7: Famous CNN Architectures")

    print("""
LENET-5 (1998) - The Pioneer
═══════════════════════════════════════════════════════════
First successful CNN for digit recognition (MNIST)

    Input → C1 → P1 → C2 → P2 → FC → FC → Output
    32×32   28    14   10    5    120  84   10

    ┌────┐   ┌──┐  ┌─┐  ┌──┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐
    │    │→ │  │→│ │→ │  │→│ │→│ │→│ │→│ │
    │    │  │  │ │ │  │  │ │ │ │ │ │ │ │ │
    └────┘   └──┘  └─┘  └──┘  └─┘  └─┘  └─┘  └─┘

    ~60K parameters
    Innovation: Demonstrated CNNs work!
    """)
    pause()

    print("""
ALEXNET (2012) - Deep Learning Revolution
═══════════════════════════════════════════════════════════
Won ImageNet 2012, sparked modern deep learning era

    Input → C1 → P1 → C2 → P2 → C3 → C4 → C5 → P3 → FC → FC → Output
    227×227                                              4096 4096 1000

    Key innovations:
    ┌────────────────────────────────────────────────────────────────┐
    │  • ReLU activation (not sigmoid/tanh)                         │
    │  • Dropout for regularization                                 │
    │  • Data augmentation                                          │
    │  • GPU training (2 GPUs in parallel)                          │
    │  • 60M parameters                                             │
    └────────────────────────────────────────────────────────────────┘

    Error rate: 15.3% (vs 26.2% for runner-up!)
    """)
    pause()

    print("""
VGG (2014) - Depth Matters
═══════════════════════════════════════════════════════════
Very deep networks with small (3×3) filters

    VGG-16: 16 weight layers
    VGG-19: 19 weight layers

    Key insight: Two 3×3 convs = one 5×5 conv receptive field
                 But with fewer parameters and more non-linearity!

    ┌───────────────────────────────────────────────────────────────┐
    │  [Conv3×3]×2 → Pool → [Conv3×3]×2 → Pool → [Conv3×3]×3 → ...  │
    │       64          128              256                        │
    └───────────────────────────────────────────────────────────────┘

    138M parameters (huge!)
    Very regular architecture: easy to understand
    """)
    pause()

    print("""
RESNET (2015) - Skip Connections
═══════════════════════════════════════════════════════════
Residual learning enables VERY deep networks (50, 101, 152 layers!)

    Problem: Very deep networks are hard to train (vanishing gradients)

    Solution: Skip connections!

    Regular block:              Residual block:
    ┌─────────────┐            ┌─────────────┐
    │      x      │            │      x      ├─────────┐
    │      ↓      │            │      ↓      │         │
    │   F(x)      │            │    F(x)     │         │
    │      ↓      │            │      ↓      │         │
    │   output    │            │   F(x)+x ←──┘         │
    └─────────────┘            └─────────────┘

    Now the gradient has a "highway" straight back to early layers!

    H(x) = F(x) + x

    If F(x) should be zero, the network just learns F(x) = 0
    (easier than learning H(x) = x from scratch)
    """)
    pause()

    print("""
ARCHITECTURE COMPARISON
═══════════════════════════════════════════════════════════

    Network    │ Year │ Depth │ Params │ Top-5 Error │ Key Innovation
    ───────────┼──────┼───────┼────────┼─────────────┼───────────────────
    LeNet-5    │ 1998 │   5   │  60K   │   N/A       │ First CNN
    AlexNet    │ 2012 │   8   │  60M   │   15.3%     │ ReLU, Dropout, GPU
    VGG-16     │ 2014 │  16   │ 138M   │   7.3%      │ Small filters
    GoogLeNet  │ 2014 │  22   │   7M   │   6.7%      │ Inception module
    ResNet-50  │ 2015 │  50   │  25M   │   3.6%      │ Skip connections
    ResNet-152 │ 2015 │ 152   │  60M   │   3.0%      │ Very deep!

    ┌─────────────────────────────────────────────────────────────────┐
    │  Trend: Deeper networks + clever architecture = better results  │
    │         But also: Efficiency matters (MobileNet, EfficientNet)  │
    └─────────────────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print_header("WEEK 3-4: CNNs AND TRAINING")
        print("""
Choose a visualization:

    [1] Convolution Operation
        - Step-by-step filter sliding

    [2] Padding and Stride
        - Spatial dimension control

    [3] Pooling Layers
        - Max pooling, average pooling

    [4] CNN Architecture
        - Full network structure, receptive fields

    [5] Batch Normalization
        - Stabilizing training dynamics

    [6] Dropout
        - Regularization through randomness

    [7] Famous Architectures
        - LeNet, AlexNet, VGG, ResNet

    [A] Run ALL visualizations

    [Q] Quit
        """)

        choice = input("Enter choice: ").strip().upper()

        if choice == '1':
            visualize_convolution()
        elif choice == '2':
            visualize_padding_stride()
        elif choice == '3':
            visualize_pooling()
        elif choice == '4':
            visualize_cnn_architecture()
        elif choice == '5':
            visualize_batch_norm()
        elif choice == '6':
            visualize_dropout()
        elif choice == '7':
            visualize_architectures()
        elif choice == 'A':
            visualize_convolution()
            visualize_padding_stride()
            visualize_pooling()
            visualize_cnn_architecture()
            visualize_batch_norm()
            visualize_dropout()
            visualize_architectures()
        elif choice == 'Q':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
