# Week 8-10: Advanced Topics

> Combined guide merging CSE 493G1/CS231n course materials with Stanford CS231n notes

## Overview

This document covers:
- Object detection (R-CNN, YOLO, etc.)
- Semantic and instance segmentation
- Generative models (VAE, GAN, Diffusion)
- Reinforcement learning basics
- 3D vision and video understanding

---

## Part 1: Object Detection

### From Classification to Detection

```
    Task Progression
    ════════════════

    Classification:        Detection:              Instance Segmentation:
    "What is in image?"    "Where are objects?"    "Pixel-level masks"

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │                 │    │  ┌───┐          │    │  ████           │
    │    🐱  🐕       │    │  │🐱 │  ┌───┐   │    │  ████  ▓▓▓▓    │
    │                 │    │  └───┘  │🐕 │   │    │        ▓▓▓▓    │
    │                 │    │         └───┘   │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
    Label: cat, dog        Boxes + labels         Per-pixel labels
```

### Two-Stage Detectors (R-CNN Family)

```
    R-CNN Evolution
    ═══════════════

    R-CNN (2014): Slow but accurate
    ┌──────────┐     ┌───────────┐     ┌─────┐     ┌──────────┐
    │  Image   │ ──▶ │  Region   │ ──▶ │ CNN │ ──▶ │ Classify │
    │          │     │ Proposals │     │(each│     │ + BBox   │
    └──────────┘     │  (~2000)  │     │ one)│     └──────────┘
                     └───────────┘     └─────┘
                     Selective Search  SLOW: CNN runs 2000×!

    Fast R-CNN (2015): Share CNN computation
    ┌──────────┐     ┌────────┐     ┌─────────┐     ┌──────────┐
    │  Image   │ ──▶ │  CNN   │ ──▶ │ RoI     │ ──▶ │ Classify │
    │          │     │ (once) │     │ Pooling │     │ + BBox   │
    └──────────┘     └────────┘     └─────────┘     └──────────┘
                                    Extract features per region

    Faster R-CNN (2016): Learn proposals
    ┌──────────┐     ┌────────┐     ┌─────────┐     ┌──────────┐
    │  Image   │ ──▶ │  CNN   │ ──▶ │ Region  │ ──▶ │ Classify │
    │          │     │        │     │Proposal │     │ + BBox   │
    └──────────┘     └────────┘     │ Network │     └──────────┘
                                    └─────────┘
                                    RPN: CNN predicts proposals!
```

### One-Stage Detectors (YOLO)

```
    YOLO: You Only Look Once
    ════════════════════════

    Key idea: Single forward pass predicts all boxes

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Image → [CNN] → [Grid of predictions]              │
    │                                                     │
    │  Each grid cell predicts:                           │
    │  - B bounding boxes                                 │
    │  - Confidence for each box                          │
    │  - C class probabilities                            │
    │                                                     │
    │  ┌───┬───┬───┬───┐    Output per cell:              │
    │  │   │   │   │   │    [x, y, w, h, conf] × B        │
    │  ├───┼───┼───┼───┤    [p₁, p₂, ..., pC]             │
    │  │   │ 🐱│   │   │                                  │
    │  ├───┼───┼───┼───┤    Total: S×S×(5B + C)          │
    │  │   │   │   │   │                                  │
    │  └───┴───┴───┴───┘                                  │
    │     S×S grid                                        │
    └─────────────────────────────────────────────────────┘

    Advantages:
    - Very fast (real-time)
    - Global reasoning (sees entire image)

    Disadvantages:
    - Lower accuracy for small objects
    - Struggles with objects close together
```

### Detection Metrics

```
    Intersection over Union (IoU)
    ═════════════════════════════

    IoU = Area of Overlap / Area of Union

    ┌─────────────┐
    │  ┌──────┐   │    Ground truth
    │  │ ████ │   │
    │  │ ████─┼───┼────┐  Predicted
    │  └──────┘   │    │
    └─────────────┼────┘
                  └─────
    IoU = ████ / (total area of both boxes)

    IoU > 0.5: typically considered "correct"

    Mean Average Precision (mAP):
    - For each class: compute precision-recall curve
    - Average precision = area under PR curve
    - mAP = mean over all classes
```

---

## Part 2: Semantic Segmentation

### Per-Pixel Classification

```
    Semantic Segmentation
    ═════════════════════

    Input:                    Output:
    ┌─────────────────────┐   ┌─────────────────────┐
    │  ▓▓▓                │   │  sky sky sky        │
    │  ▓▓▓ ┌────┐         │   │  sky pers pers      │
    │      │    │   🚗    │   │      pers pers car  │
    │ ░░░░░└────┘░░░░░░░░ │   │  road road road road│
    └─────────────────────┘   └─────────────────────┘

    Every pixel gets a class label!
```

### Fully Convolutional Networks (FCN)

```
    FCN Architecture
    ════════════════

    Replace FC layers with 1×1 convolutions:

    ┌───────┐    ┌─────────────┐    ┌───────────────┐
    │ Image │ →  │ Contracting │ →  │  Expanding    │ → Segmentation
    │H×W×3  │    │   (CNN)     │    │ (upsample)    │    H×W×C
    └───────┘    └─────────────┘    └───────────────┘

    Contracting path:         Expanding path:
    Conv + Pool               Transposed Conv
    (reduce spatial size)     (increase spatial size)

    Problem: Lose spatial detail during contraction
```

### U-Net Architecture

```
    U-Net: Skip Connections
    ═══════════════════════

    ┌─────┐                               ┌─────┐
    │     │──────────────────────────────▶│     │
    │ 64  │                               │ 64  │
    └──┬──┘                               └──▲──┘
       │ Pool                                │ UpConv
    ┌──▼──┐                               ┌──┴──┐
    │     │──────────────────────────────▶│     │
    │ 128 │                               │ 128 │
    └──┬──┘                               └──▲──┘
       │ Pool                                │ UpConv
    ┌──▼──┐                               ┌──┴──┐
    │     │──────────────────────────────▶│     │
    │ 256 │                               │ 256 │
    └──┬──┘                               └──▲──┘
       │ Pool                                │ UpConv
       └────────▶ ┌─────┐ ◀──────────────────┘
                  │ 512 │
                  │     │  Bottleneck
                  └─────┘

    Skip connections: Preserve spatial details!
```

---

## Part 3: Generative Models

### Generative vs Discriminative

```
    Model Types
    ═══════════

    Discriminative: P(y|x)
    "Given an image, what class is it?"

    Generative: P(x) or P(x|z)
    "Generate new images"

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Training data:  🐱 🐱 🐱 🐱 🐱  (many cat images)   │
    │                                                     │
    │  Generative model learns distribution P(x)          │
    │                                                     │
    │  Then can sample: 🐱 🐱 🐱  (new cat images!)        │
    │                   (never seen before)               │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### Variational Autoencoders (VAE)

```
    VAE Architecture
    ════════════════

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Input x                                            │
    │    │                                                │
    │    ▼                                                │
    │  ┌────────────┐                                     │
    │  │  Encoder   │                                     │
    │  │  q(z|x)    │                                     │
    │  └─────┬──────┘                                     │
    │        │                                            │
    │        ▼                                            │
    │    [μ, σ²]  ──▶  z = μ + σ × ε    (ε ~ N(0,1))     │
    │        │         Reparameterization trick!          │
    │        ▼                                            │
    │  ┌────────────┐                                     │
    │  │  Decoder   │                                     │
    │  │  p(x|z)    │                                     │
    │  └─────┬──────┘                                     │
    │        │                                            │
    │        ▼                                            │
    │     Output x̂                                        │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    Loss = Reconstruction + KL divergence
    L = ||x - x̂||² + KL(q(z|x) || p(z))
```

### Generative Adversarial Networks (GAN)

```
    GAN: Adversarial Training
    ═════════════════════════

    Two networks competing:

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Random noise z ──▶ ┌───────────┐ ──▶ Fake image   │
    │     N(0, I)         │ Generator │                   │
    │                     │     G     │                   │
    │                     └───────────┘                   │
    │                           │                         │
    │                           ▼                         │
    │                     ┌─────────────┐                 │
    │  Real images ─────▶ │Discriminator│ ──▶ Real/Fake  │
    │                     │      D      │                 │
    │                     └─────────────┘                 │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    G tries to fool D:  max_G  log(D(G(z)))
    D tries to catch G: max_D  log(D(x)) + log(1-D(G(z)))

    Min-max game:
    min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

### Diffusion Models

```
    Diffusion: Denoise to Generate
    ══════════════════════════════

    Forward process: Gradually add noise

    x₀ ──▶ x₁ ──▶ x₂ ──▶ ... ──▶ xₜ
    image   │      │              pure noise
           +ε₁   +ε₂     (T steps)

    Reverse process: Learn to denoise

    xₜ ──▶ x̂ₜ₋₁ ──▶ ... ──▶ x̂₁ ──▶ x̂₀
    noise  predict           predict image
           noise             noise

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Training:                                          │
    │  1. Take image x₀                                   │
    │  2. Add noise to get xₜ                             │
    │  3. Train network to predict added noise            │
    │                                                     │
    │  Generation:                                        │
    │  1. Start with pure noise xₜ ~ N(0, I)             │
    │  2. Iteratively denoise using learned model         │
    │  3. Get generated image x̂₀                          │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    Examples: DALL-E 2, Stable Diffusion, Midjourney
```

### Generative Model Comparison

| Model | Pros | Cons |
|-------|------|------|
| **VAE** | Stable training, latent space | Blurry outputs |
| **GAN** | Sharp outputs | Training instability, mode collapse |
| **Diffusion** | Best quality, stable | Slow sampling (many steps) |

---

## Part 4: Reinforcement Learning

### RL Basics

```
    RL Framework
    ════════════

    Agent interacts with environment to maximize reward:

                    ┌─────────────────┐
                    │   Environment   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ State sₜ        │
                    │ Reward rₜ       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Agent       │
                    │   π(a|s)        │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ Action aₜ       │
                    └─────────────────┘

    Goal: Learn policy π(a|s) to maximize expected reward
    E[Σₜ γᵗ rₜ]
```

### Deep RL for Games

```
    DQN: Deep Q-Network
    ════════════════════

    Learn Q-function: Q(s, a) = expected future reward

    State s (pixels) ──▶ [CNN] ──▶ Q(s, a₁), Q(s, a₂), ...

    Choose action with highest Q:  a* = argmax_a Q(s, a)

    Training:
    - Experience replay: store (s, a, r, s') in buffer
    - Target network: stabilize training
```

---

## Part 5: 3D Vision

### Depth Estimation

```
    Monocular Depth Estimation
    ══════════════════════════

    Input: Single RGB image
    Output: Per-pixel depth map

    ┌─────────────────┐    ┌─────────────────┐
    │   RGB Image     │ →  │   Depth Map     │
    │  🏔️ 🏠 🌳      │    │  ░░░ ███ ▓▓▓   │
    │                 │    │  (near to far)  │
    └─────────────────┘    └─────────────────┘

    Methods:
    - Supervised: train on RGB-D datasets
    - Self-supervised: stereo or video supervision
```

### 3D Representations

```
    3D Data Types
    ═════════════

    Point Cloud:           Voxel Grid:           Mesh:
    ● ● ●                  ┌─┬─┬─┐              /\  /\
    ● ● ●                  ├─┼─┼─┤             /  \/  \
    ● ● ●                  └─┴─┴─┘            /________\

    (x,y,z) points         3D pixel grid       Vertices + faces
    Sparse, flexible       Dense, memory       Explicit surface
    PointNet, etc.         3D CNNs             Graph networks
```

### Neural Radiance Fields (NeRF)

```
    NeRF: Novel View Synthesis
    ══════════════════════════

    Input: Multiple views of a scene
    Output: New views from any angle

    Core idea: MLP learns (x, y, z, θ, φ) → (color, density)

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Position (x,y,z) ─┬─▶ ┌─────┐                     │
    │                    │   │     │                     │
    │  Direction (θ,φ) ──┘   │ MLP │──▶ (R,G,B, σ)      │
    │                        │     │    color  density   │
    │                        └─────┘                     │
    │                                                     │
    │  Render: Ray marching through learned volume        │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **Detection** | Two-stage (R-CNN) vs one-stage (YOLO) |
| **Segmentation** | FCN, U-Net skip connections |
| **VAE** | Encode to latent, reconstruct + KL loss |
| **GAN** | Generator vs discriminator game |
| **Diffusion** | Denoise from noise, best quality |
| **RL** | Agent, environment, policy, reward |
| **3D Vision** | Point clouds, voxels, meshes, NeRF |

---

## References

**Detection:**
- Girshick et al., "Rich feature hierarchies for accurate object detection", 2014 (R-CNN)
- Redmon et al., "You Only Look Once", 2016 (YOLO)

**Segmentation:**
- Long et al., "Fully Convolutional Networks", 2015
- Ronneberger et al., "U-Net", 2015

**Generative Models:**
- Kingma & Welling, "Auto-Encoding Variational Bayes", 2014 (VAE)
- Goodfellow et al., "Generative Adversarial Networks", 2014
- Ho et al., "Denoising Diffusion Probabilistic Models", 2020

**3D Vision:**
- Mildenhall et al., "NeRF", 2020
