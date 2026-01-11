# Week 8-10: Advanced Deep Learning Topics
## Detection, Segmentation, Generative Models, and Reinfortic Learning

---

## Table of Contents
1. [Object Detection](#object-detection)
2. [Semantic Segmentation](#semantic-segmentation)
3. [Generative Models Overview](#generative-models-overview)
4. [Variational Autoencoders (VAEs)](#variational-autoencoders)
5. [Generative Adversarial Networks (GANs)](#generative-adversarial-networks)
6. [Diffusion Models](#diffusion-models)
7. [Reinforcement Learning Basics](#reinforcement-learning-basics)
8. [Large Language Models](#large-language-models)
9. [Coding Exercises](#coding-exercises)
10. [Business Applications](#business-applications)

---

## Object Detection

### The Task

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBJECT DETECTION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Classification: ONE label for entire image                               │
│   Detection: MULTIPLE objects with LOCATIONS                               │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Classification:          Detection:                              │  │
│   │   ┌───────────┐            ┌───────────────────────┐              │  │
│   │   │           │            │ ┌─────┐               │              │  │
│   │   │   🐱 🐕   │ → "pets"   │ │ 🐱  │ cat (0.95)   │              │  │
│   │   │           │            │ └─────┘  ┌─────┐      │              │  │
│   │   │           │            │          │ 🐕  │      │ → Multiple   │  │
│   │   │           │            │          └─────┘      │   boxes +    │  │
│   │   │           │            │          dog (0.92)   │   labels +   │  │
│   │   └───────────┘            └───────────────────────┘   confidence │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   OUTPUT FORMAT:                                                           │
│   For each detected object:                                                │
│   • Bounding box: (x, y, width, height)                                   │
│   • Class label: "cat", "dog", "car", etc.                                │
│   • Confidence score: 0.0 - 1.0                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Two-Stage Detectors (R-CNN Family)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE DETECTION (R-CNN)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STAGE 1: Generate region proposals ("where might objects be?")           │
│   STAGE 2: Classify each region ("what is this object?")                  │
│                                                                             │
│   R-CNN (2014):                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input Image                                                       │  │
│   │       │                                                             │  │
│   │       ▼                                                             │  │
│   │   ┌─────────────────┐                                              │  │
│   │   │ Region Proposal │  → ~2000 region proposals                    │  │
│   │   │ (Selective      │                                              │  │
│   │   │  Search)        │                                              │  │
│   │   └────────┬────────┘                                              │  │
│   │            │                                                        │  │
│   │   For EACH region:                                                 │  │
│   │   ┌────────▼────────┐                                              │  │
│   │   │  Crop & Resize  │  → Fixed size (e.g., 224×224)               │  │
│   │   └────────┬────────┘                                              │  │
│   │            │                                                        │  │
│   │   ┌────────▼────────┐                                              │  │
│   │   │      CNN        │  → Extract features (×2000 times!)          │  │
│   │   └────────┬────────┘                                              │  │
│   │            │                                                        │  │
│   │   ┌────────▼────────┐                                              │  │
│   │   │   Classify +    │  → Class + box refinement                   │  │
│   │   │   Box Regress   │                                              │  │
│   │   └─────────────────┘                                              │  │
│   │                                                                     │  │
│   │   PROBLEM: ~47 seconds per image (way too slow!)                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Fast R-CNN (2015): Share CNN computation                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input Image                                                       │  │
│   │       │                                                             │  │
│   │       ▼                                                             │  │
│   │   ┌─────────┐        ┌─────────────┐                              │  │
│   │   │   CNN   │ ─────► │Feature Map  │  Run CNN ONCE on whole image │  │
│   │   └─────────┘        └──────┬──────┘                              │  │
│   │                             │                                      │  │
│   │   Region Proposals ────────►│                                      │  │
│   │                             ▼                                      │  │
│   │                      ┌─────────────┐                              │  │
│   │                      │ ROI Pooling │  Extract features per region │  │
│   │                      └──────┬──────┘                              │  │
│   │                             │                                      │  │
│   │                      ┌──────▼──────┐                              │  │
│   │                      │ FC + Output │                              │  │
│   │                      └─────────────┘                              │  │
│   │                                                                     │  │
│   │   ~0.3 seconds per image (150× faster!)                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Faster R-CNN (2016): Learn region proposals too                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │           Region Proposal Network (RPN)                     │  │  │
│   │   │                                                             │  │  │
│   │   │   Feature Map ──► Slide 3×3 window ──► Predict:            │  │  │
│   │   │                                         • Objectness (2)   │  │  │
│   │   │                                         • Box deltas (4)   │  │  │
│   │   │                                         Per anchor box     │  │  │
│   │   │                                                             │  │  │
│   │   │   Anchor boxes: multiple sizes/ratios at each position     │  │  │
│   │   │   ┌───┐  ┌─────┐  ┌─┐                                      │  │  │
│   │   │   │   │  │     │  │ │  (e.g., 3 scales × 3 ratios = 9)    │  │  │
│   │   │   └───┘  └─────┘  └─┘                                      │  │  │
│   │   │                                                             │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   │   ~0.2 seconds per image (real-time!)                             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### One-Stage Detectors (YOLO, SSD)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   ONE-STAGE DETECTION (YOLO)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IDEA: Skip region proposals, predict directly from feature map           │
│                                                                             │
│   YOLO (You Only Look Once):                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input Image (448×448)                                            │  │
│   │         │                                                           │  │
│   │         ▼                                                           │  │
│   │   ┌─────────────┐                                                  │  │
│   │   │     CNN     │                                                  │  │
│   │   └──────┬──────┘                                                  │  │
│   │          │                                                          │  │
│   │          ▼                                                          │  │
│   │   ┌─────────────┐                                                  │  │
│   │   │ 7×7 Grid    │  Divide image into S×S grid                     │  │
│   │   │             │                                                  │  │
│   │   │  ┌─┬─┬─┬─┬  │  Each cell predicts:                            │  │
│   │   │  ├─┼─┼─┼─┼  │  • B bounding boxes                             │  │
│   │   │  ├─┼─┼─┼─┼  │  • Confidence for each box                      │  │
│   │   │  ├─┼─┼─┼─┼  │  • C class probabilities                        │  │
│   │   │  └─┴─┴─┴─┴  │                                                  │  │
│   │   └─────────────┘                                                  │  │
│   │                                                                     │  │
│   │   Output tensor: S × S × (B×5 + C)                                 │  │
│   │   For S=7, B=2, C=20: 7×7×30 = 1470 predictions                   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Per bounding box prediction:                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   (x, y)     : Center of box relative to grid cell                │  │
│   │   (w, h)     : Width/height relative to image                     │  │
│   │   confidence : P(object) × IOU(pred, truth)                       │  │
│   │                                                                     │  │
│   │   Grid Cell Output:                                                │  │
│   │   [x, y, w, h, conf] × B + [P(class₁), P(class₂), ..., P(class_C)]│  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   SPEED: ~45 FPS (real-time video!)                                       │
│                                                                             │
│   YOLO EVOLUTION:                                                          │
│   • YOLOv1 (2016): Original                                               │
│   • YOLOv2 (2017): Batch norm, anchor boxes, multi-scale                  │
│   • YOLOv3 (2018): Feature pyramid, better backbone                       │
│   • YOLOv4 (2020): CSPDarknet, SPP, PANet                                 │
│   • YOLOv5 (2020): PyTorch, extensive augmentation                        │
│   • YOLOv8 (2023): Anchor-free, decoupled head                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Semantic Segmentation

### The Task

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SEMANTIC SEGMENTATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Classification: One label per IMAGE                                      │
│   Detection: One label per OBJECT (with box)                              │
│   Segmentation: One label per PIXEL                                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input Image              Semantic Segmentation                    │  │
│   │   ┌───────────────┐        ┌───────────────┐                       │  │
│   │   │  🐱  🚗       │        │░░░░░▓▓▓▓▓▓▓▓▓│   ░ = background      │  │
│   │   │               │   →    │░░░░░▓▓▓▓▓▓▓▓▓│   ▓ = car             │  │
│   │   │  ___          │        │██████░░░░░░░░│   █ = cat             │  │
│   │   │ (street)      │        │░░░░░░░░░░░░░░│   ▒ = road            │  │
│   │   │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│        │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│                       │  │
│   │   └───────────────┘        └───────────────┘                       │  │
│   │                                                                     │  │
│   │   Every pixel gets a class label!                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TYPES OF SEGMENTATION:                                                   │
│                                                                             │
│   Semantic:    All cats = same label (no instance distinction)            │
│   Instance:    Each cat = different label (detect + segment)              │
│   Panoptic:    Both! (things = instances, stuff = semantic)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fully Convolutional Networks (FCN)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                FULLY CONVOLUTIONAL NETWORKS (FCN)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IDEA: Replace FC layers with convolutions to get dense predictions       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Classification CNN:                                              │  │
│   │   [Conv-Pool] × N ──► Flatten ──► FC ──► FC ──► 1000 classes      │  │
│   │                              7×7×512 ──► 4096 ──► 4096 ──► 1000    │  │
│   │                                                                     │  │
│   │   Fully Convolutional:                                             │  │
│   │   [Conv-Pool] × N ──► Conv(1×1) ──► Conv(1×1) ──► H×W×C           │  │
│   │                 7×7×512 ──► 7×7×4096 ──► 7×7×C (per-pixel classes)│  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   PROBLEM: Output is small (e.g., 7×7), we need original resolution        │
│                                                                             │
│   SOLUTION: Upsampling (transposed convolution / bilinear)                │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Input: 224×224                                                   │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   ┌─────────┐                                                      │  │
│   │   │ Encoder │  Conv-Pool-Conv-Pool-...  (shrink spatially)        │  │
│   │   └────┬────┘                                                      │  │
│   │        │  7×7×512                                                  │  │
│   │        ▼                                                            │  │
│   │   ┌─────────┐                                                      │  │
│   │   │ Decoder │  Upsample-Conv-Upsample-... (grow back)             │  │
│   │   └────┬────┘                                                      │  │
│   │        │                                                            │  │
│   │        ▼                                                            │  │
│   │   Output: 224×224×C (per-pixel class scores)                       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### U-Net Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          U-NET ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   KEY INNOVATION: Skip connections between encoder and decoder             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │        ENCODER                              DECODER                 │  │
│   │   (Contracting Path)                   (Expanding Path)            │  │
│   │                                                                     │  │
│   │   ┌──────────┐                                   ┌──────────┐      │  │
│   │   │ 572×572  │───────────── SKIP ───────────────►│ 388×388  │      │  │
│   │   │   ×64    │                                   │   ×64    │      │  │
│   │   └────┬─────┘                                   └────▲─────┘      │  │
│   │        │ Pool                                         │ Up         │  │
│   │   ┌────▼─────┐                                   ┌────┴─────┐      │  │
│   │   │ 284×284  │───────────── SKIP ───────────────►│ 196×196  │      │  │
│   │   │  ×128    │                                   │  ×128    │      │  │
│   │   └────┬─────┘                                   └────▲─────┘      │  │
│   │        │ Pool                                         │ Up         │  │
│   │   ┌────▼─────┐                                   ┌────┴─────┐      │  │
│   │   │ 140×140  │───────────── SKIP ───────────────►│  96×96   │      │  │
│   │   │  ×256    │                                   │  ×256    │      │  │
│   │   └────┬─────┘                                   └────▲─────┘      │  │
│   │        │ Pool                                         │ Up         │  │
│   │   ┌────▼─────┐                                   ┌────┴─────┐      │  │
│   │   │  68×68   │───────────── SKIP ───────────────►│  48×48   │      │  │
│   │   │  ×512    │                                   │  ×512    │      │  │
│   │   └────┬─────┘                                   └────▲─────┘      │  │
│   │        │ Pool                                         │ Up         │  │
│   │        │                                              │            │  │
│   │        └──────────► ┌──────────┐ ◄────────────────────┘            │  │
│   │                     │  32×32   │                                   │  │
│   │                     │  ×1024   │  Bottleneck                       │  │
│   │                     └──────────┘                                   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WHY SKIP CONNECTIONS?                                                    │
│   • Encoder loses fine-grained spatial information                        │
│   • Decoder needs this for precise boundaries                             │
│   • Skip connections pass high-resolution features directly               │
│                                                                             │
│   WIDELY USED IN:                                                          │
│   • Medical image segmentation                                            │
│   • Satellite imagery                                                     │
│   • Autonomous driving                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Generative Models Overview

### Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GENERATIVE MODELS OVERVIEW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GOAL: Learn to generate NEW data samples from the same distribution     │
│                                                                             │
│   Training Data: {x₁, x₂, ..., xₙ} ~ P_data(x)                             │
│   Model learns: P_model(x) ≈ P_data(x)                                     │
│   Generate: x_new ~ P_model(x)                                             │
│                                                                             │
│                                                                             │
│   TAXONOMY OF GENERATIVE MODELS:                                           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ┌───────────────────────────────────────────────────────────┐    │  │
│   │   │              GENERATIVE MODELS                            │    │  │
│   │   └───────────────────────────┬───────────────────────────────┘    │  │
│   │                               │                                    │  │
│   │         ┌─────────────────────┼─────────────────────┐             │  │
│   │         │                     │                     │             │  │
│   │         ▼                     ▼                     ▼             │  │
│   │   ┌───────────┐        ┌───────────┐        ┌───────────┐        │  │
│   │   │  Explicit │        │ Implicit  │        │  Hybrid   │        │  │
│   │   │  Density  │        │  Density  │        │           │        │  │
│   │   └─────┬─────┘        └─────┬─────┘        └─────┬─────┘        │  │
│   │         │                    │                    │              │  │
│   │    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐        │  │
│   │    │         │          │         │          │         │        │  │
│   │    ▼         ▼          ▼         ▼          ▼         ▼        │  │
│   │  ┌───┐    ┌───┐      ┌───┐    ┌─────┐    ┌─────┐  ┌─────┐     │  │
│   │  │VAE│    │Flow│     │GAN│    │Energy│   │Diff-│  │Auto-│     │  │
│   │  │   │    │   │      │   │    │Based │   │usion│  │regr.│     │  │
│   │  └───┘    └───┘      └───┘    └─────┘    └─────┘  └─────┘     │  │
│   │                                                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   COMPARISON:                                                              │
│   ┌─────────────┬───────────────┬──────────────┬────────────────────┐     │
│   │   Model     │  Sample Speed │  Sample Qual │  Mode Coverage    │     │
│   ├─────────────┼───────────────┼──────────────┼────────────────────┤     │
│   │ VAE         │ Fast          │ Blurry       │ Good               │     │
│   │ GAN         │ Fast          │ Sharp        │ Mode collapse risk │     │
│   │ Flow        │ Medium        │ Good         │ Good               │     │
│   │ Diffusion   │ Slow          │ Excellent    │ Excellent          │     │
│   │ Autoregr.   │ Very slow     │ Excellent    │ Good               │     │
│   └─────────────┴───────────────┴──────────────┴────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Variational Autoencoders

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VARIATIONAL AUTOENCODER (VAE)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   REGULAR AUTOENCODER:                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   x ──► ENCODER ──► z (latent code) ──► DECODER ──► x̂             │  │
│   │                      (deterministic)                                │  │
│   │                                                                     │  │
│   │   Loss = ||x - x̂||²                                                │  │
│   │                                                                     │  │
│   │   Problem: z is just a compressed code, not a nice distribution    │  │
│   │            Can't sample random z to generate new images            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   VARIATIONAL AUTOENCODER:                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   x ──► ENCODER ──► μ, σ ──► z ~ N(μ, σ²) ──► DECODER ──► x̂       │  │
│   │                      (distribution!)                                │  │
│   │                                                                     │  │
│   │   Encoder outputs PARAMETERS of a distribution, not a point!       │  │
│   │                                                                     │  │
│   │   To generate: Sample z ~ N(0, I), then decode                     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   DETAILED ARCHITECTURE:                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │           ENCODER                              DECODER              │  │
│   │      (Recognition Model)                 (Generative Model)        │  │
│   │                                                                     │  │
│   │   ┌─────────────┐                         ┌─────────────┐          │  │
│   │   │             │                         │             │          │  │
│   │   │      x      │                         │      x̂      │          │  │
│   │   │  (image)    │                         │ (reconstructed)        │  │
│   │   │             │                         │             │          │  │
│   │   └──────┬──────┘                         └──────▲──────┘          │  │
│   │          │                                       │                  │  │
│   │          ▼                                       │                  │  │
│   │   ┌─────────────┐                         ┌──────┴──────┐          │  │
│   │   │   Neural    │                         │   Neural    │          │  │
│   │   │   Network   │                         │   Network   │          │  │
│   │   └──────┬──────┘                         └──────▲──────┘          │  │
│   │          │                                       │                  │  │
│   │     ┌────┴────┐                                  │                  │  │
│   │     │         │                                  │                  │  │
│   │     ▼         ▼                                  │                  │  │
│   │   ┌───┐     ┌───┐                               │                  │  │
│   │   │ μ │     │ σ │                               │                  │  │
│   │   └─┬─┘     └─┬─┘                               │                  │  │
│   │     │         │                                  │                  │  │
│   │     └────┬────┘                                  │                  │  │
│   │          │                                       │                  │  │
│   │          ▼                                       │                  │  │
│   │   ┌─────────────┐                               │                  │  │
│   │   │  SAMPLE z   │   z = μ + σ × ε              │                  │  │
│   │   │  from N(μ,σ)│   (ε ~ N(0,1))  ─────────────┘                  │  │
│   │   └─────────────┘   (reparameterization trick)                     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   VAE LOSS:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   L = Reconstruction Loss + KL Divergence                          │  │
│   │                                                                     │  │
│   │   L = E[log p(x|z)] - KL(q(z|x) || p(z))                          │  │
│   │       └─────┬─────┘   └────────┬─────────┘                        │  │
│   │     Make x̂ close to x  Force q(z|x) close to N(0,I)              │  │
│   │     (reconstruction)    (regularization)                           │  │
│   │                                                                     │  │
│   │   KL term (closed form for Gaussian):                             │  │
│   │   KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Generative Adversarial Networks

### Architecture and Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                GENERATIVE ADVERSARIAL NETWORKS (GANs)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CORE IDEA: Two networks playing a game                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   GENERATOR (G)              vs              DISCRIMINATOR (D)     │  │
│   │   "The Counterfeiter"                        "The Detective"       │  │
│   │                                                                     │  │
│   │   Goal: Create fake                          Goal: Distinguish     │  │
│   │   data that fools D                          real from fake        │  │
│   │                                                                     │  │
│   │   ┌─────────────┐                           ┌─────────────┐        │  │
│   │   │   Noise z   │                           │ Real/Fake?  │        │  │
│   │   │  ~ N(0,I)   │                           │   (0/1)     │        │  │
│   │   └──────┬──────┘                           └──────▲──────┘        │  │
│   │          │                                         │               │  │
│   │          ▼                                         │               │  │
│   │   ┌─────────────┐                           ┌──────┴──────┐        │  │
│   │   │  Generator  │                           │Discriminator│        │  │
│   │   │   Network   │                           │   Network   │        │  │
│   │   └──────┬──────┘                           └──────▲──────┘        │  │
│   │          │                                         │               │  │
│   │          ▼                                    ┌────┴────┐          │  │
│   │   ┌─────────────┐                            │         │          │  │
│   │   │ Fake Image  │ ─────────────────────────► │         │          │  │
│   │   │   G(z)      │                            │         │          │  │
│   │   └─────────────┘                            │         │          │  │
│   │                                              │         │          │  │
│   │   ┌─────────────┐                            │         │          │  │
│   │   │ Real Image  │ ─────────────────────────► │         │          │  │
│   │   │    x        │                            │         │          │  │
│   │   └─────────────┘                                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TRAINING (Minimax Game):                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]                  │  │
│   │           └─────┬─────┘   └─────────┬────────┘                    │  │
│   │         D wants this    D wants this high                         │  │
│   │         high (real=1)   (fake=0, so log(1-0)=0)                   │  │
│   │                                                                     │  │
│   │         G wants this low (fake=1, so log(1-1)=-∞)                 │  │
│   │                                                                     │  │
│   │   ALTERNATING UPDATES:                                             │  │
│   │   1. Train D: maximize log D(x) + log(1 - D(G(z)))                │  │
│   │   2. Train G: minimize log(1 - D(G(z)))                           │  │
│   │              (or equivalently, maximize log D(G(z)))              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TRAINING CHALLENGES:                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   MODE COLLAPSE:           TRAINING INSTABILITY:                   │  │
│   │   G only generates         D becomes too strong,                   │  │
│   │   a few types of images    G gradients vanish                      │  │
│   │                                                                     │  │
│   │   Real data: 🐱 🐕 🚗 🏠    ┌─────────────────────┐                │  │
│   │   Generated:  🐱 🐱 🐱 🐱    │ Loss               │                │  │
│   │                            │  │  ╱╲╱╲╱╲╱╲╱╲╱╲   │ ← Oscillates    │  │
│   │                            │  └─────────────────►│                │  │
│   │                            └─────────────────────┘                │  │
│   │                                                                     │  │
│   │   SOLUTIONS:                                                       │  │
│   │   • Better architectures (DCGAN, StyleGAN)                        │  │
│   │   • Better losses (Wasserstein, hinge)                            │  │
│   │   • Regularization (spectral norm, gradient penalty)              │  │
│   │   • Progressive training                                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Diffusion Models

### The Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIFFUSION MODELS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CORE IDEA: Learn to reverse a gradual noising process                    │
│                                                                             │
│   FORWARD PROCESS (fixed, adds noise):                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   x₀ ──► x₁ ──► x₂ ──► ... ──► x_T                                │  │
│   │   Clean      Progressively more noisy    Pure noise               │  │
│   │                                                                     │  │
│   │   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                     │  │
│   │   │ 🐱  │→ │░🐱░ │→ │░░░░ │→ │░░░░ │→ │noise│                     │  │
│   │   │     │  │     │  │ ░░  │  │░░░░ │  │     │                     │  │
│   │   └─────┘  └─────┘  └─────┘  └─────┘  └─────┘                     │  │
│   │    t=0      t=1      t=2              t=T                         │  │
│   │                                                                     │  │
│   │   q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   REVERSE PROCESS (learned, removes noise):                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   x_T ◄── x_{T-1} ◄── ... ◄── x₁ ◄── x₀                           │  │
│   │   Pure noise      Gradually denoise     Clean!                     │  │
│   │                                                                     │  │
│   │   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                     │  │
│   │   │noise│← │░░░░ │← │░░░  │← │░🐱░ │← │ 🐱  │                     │  │
│   │   │     │  │░░░░ │  │ ░░  │  │     │  │     │                     │  │
│   │   └─────┘  └─────┘  └─────┘  └─────┘  └─────┘                     │  │
│   │    t=T              Neural network predicts                t=0     │  │
│   │                     noise at each step                             │  │
│   │                                                                     │  │
│   │   p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))      │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TRAINING:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. Sample x₀ from training data                                  │  │
│   │   2. Sample random timestep t ~ Uniform(1, T)                      │  │
│   │   3. Sample noise ε ~ N(0, I)                                      │  │
│   │   4. Create noisy image: x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε             │  │
│   │   5. Train network to predict ε from x_t                           │  │
│   │                                                                     │  │
│   │   Loss = ||ε - ε_θ(x_t, t)||²                                      │  │
│   │          └──────────┬─────┘                                        │  │
│   │             Network predicts                                        │  │
│   │             the noise that was added                               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   GENERATION:                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. Sample x_T ~ N(0, I)        (pure noise)                      │  │
│   │   2. For t = T, T-1, ..., 1:                                       │  │
│   │      • Predict noise: ε = ε_θ(x_t, t)                              │  │
│   │      • Denoise: x_{t-1} = (x_t - √(1-α_t) ε) / √(α_t) + σ_t z     │  │
│   │   3. Return x₀                                                     │  │
│   │                                                                     │  │
│   │   Typically T = 1000 steps (slow but high quality!)               │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WHY DIFFUSION WORKS SO WELL:                                             │
│   • Simple training objective (predict noise)                             │
│   • Stable training (no adversarial dynamics)                             │
│   • Excellent mode coverage                                               │
│   • State-of-the-art image quality                                        │
│                                                                             │
│   EXAMPLES: DALL-E 2, Stable Diffusion, Midjourney, Imagen                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Reinforcement Learning Basics

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING BASICS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SETTING: Agent learns by interacting with environment                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │          ┌─────────────────────────────────────────────┐            │  │
│   │          │                                             │            │  │
│   │          │              ENVIRONMENT                    │            │  │
│   │          │                                             │            │  │
│   │          │  ┌───────────────────────────────────────┐ │            │  │
│   │          │  │                                       │ │            │  │
│   │          │  │     State s_t     Reward r_t          │ │            │  │
│   │          │  │        │             │                │ │            │  │
│   │          │  └────────┼─────────────┼────────────────┘ │            │  │
│   │          │           │             │                   │            │  │
│   │          └───────────┼─────────────┼───────────────────┘            │  │
│   │                      │             │                                │  │
│   │                      ▼             ▼                                │  │
│   │                 ┌─────────────────────────┐                         │  │
│   │                 │                         │                         │  │
│   │                 │         AGENT           │                         │  │
│   │                 │                         │                         │  │
│   │                 │    Policy π(a|s)        │                         │  │
│   │                 │                         │                         │  │
│   │                 └───────────┬─────────────┘                         │  │
│   │                             │                                       │  │
│   │                             │ Action a_t                            │  │
│   │                             │                                       │  │
│   │                             ▼                                       │  │
│   │                    Back to environment                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   KEY COMPONENTS:                                                          │
│   • State (s): Observation of environment                                 │
│   • Action (a): What agent can do                                         │
│   • Reward (r): Scalar feedback signal                                    │
│   • Policy (π): Strategy mapping states to actions                        │
│   • Value (V): Expected future reward from state                          │
│                                                                             │
│   GOAL: Find policy π* that maximizes expected cumulative reward          │
│                                                                             │
│   max_π E[Σ_t γᵗ r_t]                                                     │
│         └────┬────┘                                                       │
│         Discounted sum                                                    │
│         (γ < 1 values                                                     │
│          immediate reward)                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Deep Q-Learning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEEP Q-LEARNING (DQN)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Q-VALUE: Expected future reward taking action a in state s              │
│                                                                             │
│   Q(s, a) = E[r + γ max_a' Q(s', a')]                                     │
│                                                                             │
│   Use neural network to approximate Q:                                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   State s         Q-Network                Q-values                │  │
│   │   (e.g., game                                                      │  │
│   │    screen)                                                          │  │
│   │   ┌─────────┐     ┌─────────────┐         ┌─────────────┐          │  │
│   │   │         │     │             │         │ Q(s, left)  │          │  │
│   │   │  🎮     │ ──► │   CNN       │ ──────► │ Q(s, right) │          │  │
│   │   │         │     │    +        │         │ Q(s, up)    │          │  │
│   │   │         │     │   FC        │         │ Q(s, down)  │          │  │
│   │   └─────────┘     └─────────────┘         └─────────────┘          │  │
│   │                                                                     │  │
│   │   Action = argmax Q(s, a)   (pick action with highest Q)           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TRAINING:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. Collect experience: (s, a, r, s')                             │  │
│   │   2. Store in replay buffer                                        │  │
│   │   3. Sample batch from buffer                                      │  │
│   │   4. Compute target: y = r + γ max_a' Q(s', a'; θ⁻)               │  │
│   │                           └───────┬────────┘                       │  │
│   │                           Target network                           │  │
│   │                           (frozen periodically)                    │  │
│   │   5. Update: minimize (Q(s, a; θ) - y)²                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   KEY INNOVATIONS:                                                         │
│   • Experience replay: Break correlations in sequential data              │
│   • Target network: Stabilize training                                    │
│   • ε-greedy: Explore vs exploit                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Large Language Models

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LARGE LANGUAGE MODELS (LLMs)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FOUNDATION: Transformer decoder (autoregressive)                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Predict next token given previous tokens:                        │  │
│   │                                                                     │  │
│   │   P(x_t | x_1, x_2, ..., x_{t-1})                                  │  │
│   │                                                                     │  │
│   │   "The cat sat on the" → "mat" (predict next word)                 │  │
│   │                                                                     │  │
│   │   Training: Maximize likelihood of training text                   │  │
│   │   Loss = -Σ log P(x_t | x_1, ..., x_{t-1})                         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   SCALE:                                                                   │
│   ┌───────────────┬───────────────┬───────────────┬─────────────────────┐  │
│   │    Model      │  Parameters   │ Training Data │      Year           │  │
│   ├───────────────┼───────────────┼───────────────┼─────────────────────┤  │
│   │ GPT-1         │    117M       │    ~5GB       │      2018           │  │
│   │ GPT-2         │    1.5B       │    ~40GB      │      2019           │  │
│   │ GPT-3         │    175B       │   ~570GB      │      2020           │  │
│   │ GPT-4         │   ~1.7T?      │    ???        │      2023           │  │
│   │ Claude 3      │    ???        │    ???        │      2024           │  │
│   │ Llama 3       │    405B       │    15T        │      2024           │  │
│   └───────────────┴───────────────┴───────────────┴─────────────────────┘  │
│                                                                             │
│   EMERGENT ABILITIES (appear at scale):                                    │
│   • In-context learning                                                    │
│   • Chain-of-thought reasoning                                            │
│   • Code generation                                                       │
│   • Multi-step reasoning                                                  │
│   • Following complex instructions                                        │
│                                                                             │
│   PROMPTING TECHNIQUES:                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   ZERO-SHOT:                                                       │  │
│   │   "Translate to French: Hello" → "Bonjour"                        │  │
│   │                                                                     │  │
│   │   FEW-SHOT:                                                        │  │
│   │   "great → positive                                                │  │
│   │    terrible → negative                                             │  │
│   │    amazing →" → "positive"                                         │  │
│   │                                                                     │  │
│   │   CHAIN-OF-THOUGHT:                                                │  │
│   │   "Let's think step by step..."                                   │  │
│   │   Helps with math, reasoning                                       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Coding Exercises

### Exercise 1: Simple Object Detector

```python
#==============================================================================
# EXERCISE 1: SIMPLE OBJECT DETECTOR (Sliding Window)
#==============================================================================

import numpy as np

class SlidingWindowDetector:
    """
    Simple object detector using sliding window approach.
    """

    def __init__(self, classifier, window_sizes=[(64, 64), (128, 128)],
                 stride=16, threshold=0.5):
        """
        Initialize detector.

        Args:
            classifier: Function that takes image patch, returns (class, score)
            window_sizes: List of (height, width) window sizes
            stride: Step size for sliding window
            threshold: Confidence threshold for detection
        """
        self.classifier = classifier
        self.window_sizes = window_sizes
        self.stride = stride
        self.threshold = threshold

    def detect(self, image):
        """
        Detect objects in image using sliding window.

        Args:
            image: Input image of shape (H, W, 3)

        Returns:
            detections: List of (x, y, w, h, class, score)
        """
        H, W = image.shape[:2]
        detections = []

        for win_h, win_w in self.window_sizes:
            for y in range(0, H - win_h + 1, self.stride):
                for x in range(0, W - win_w + 1, self.stride):
                    # Extract window
                    window = image[y:y+win_h, x:x+win_w]

                    # Classify window
                    cls, score = self.classifier(window)

                    if score > self.threshold:
                        detections.append({
                            'x': x,
                            'y': y,
                            'width': win_w,
                            'height': win_h,
                            'class': cls,
                            'score': score
                        })

        # Apply Non-Maximum Suppression
        detections = self.nms(detections)

        return detections

    def nms(self, detections, iou_threshold=0.5):
        """
        Non-Maximum Suppression to remove duplicate detections.

        Args:
            detections: List of detection dictionaries
            iou_threshold: IOU threshold for suppression

        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return []

        # Sort by score
        detections = sorted(detections, key=lambda x: -x['score'])

        keep = []
        while len(detections) > 0:
            # Keep highest scoring detection
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            remaining = []
            for det in detections:
                if self.iou(best, det) < iou_threshold:
                    remaining.append(det)
            detections = remaining

        return keep

    def iou(self, box1, box2):
        """
        Compute Intersection over Union between two boxes.
        """
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection

        return intersection / union


#==============================================================================
# EXERCISE 2: U-NET FOR SEGMENTATION
#==============================================================================

class ConvBlock:
    """Double convolution block for U-Net."""

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weights
        scale = np.sqrt(2.0 / (3 * 3 * in_channels))
        self.W1 = np.random.randn(out_channels, in_channels, 3, 3) * scale
        self.b1 = np.zeros(out_channels)
        self.W2 = np.random.randn(out_channels, out_channels, 3, 3) * scale
        self.b2 = np.zeros(out_channels)

    def forward(self, x):
        """
        Forward pass: Conv -> ReLU -> Conv -> ReLU
        """
        # First conv + ReLU
        h = self.conv2d(x, self.W1, self.b1)
        h = np.maximum(0, h)

        # Second conv + ReLU
        h = self.conv2d(h, self.W2, self.b2)
        h = np.maximum(0, h)

        return h

    def conv2d(self, x, W, b):
        """Simple 2D convolution with same padding."""
        # Simplified implementation
        N, C, H, W_in = x.shape
        F, _, HH, WW = W.shape
        pad = (HH - 1) // 2

        x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

        out = np.zeros((N, F, H, W_in))
        for n in range(N):
            for f in range(F):
                for i in range(H):
                    for j in range(W_in):
                        out[n, f, i, j] = np.sum(
                            x_padded[n, :, i:i+HH, j:j+WW] * W[f]
                        ) + b[f]

        return out


class SimpleUNet:
    """Simplified U-Net for segmentation."""

    def __init__(self, in_channels=3, num_classes=2):
        """Initialize U-Net."""
        self.num_classes = num_classes

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)

        # Output
        self.out_conv = np.random.randn(num_classes, 64, 1, 1) * 0.1

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x: Input of shape (N, C, H, W)

        Returns:
            Segmentation logits of shape (N, num_classes, H, W)
        """
        # Encoder path
        e1 = self.enc1.forward(x)
        e1_pool = self.max_pool(e1)

        e2 = self.enc2.forward(e1_pool)
        e2_pool = self.max_pool(e2)

        e3 = self.enc3.forward(e2_pool)
        e3_pool = self.max_pool(e3)

        # Bottleneck
        b = self.bottleneck.forward(e3_pool)

        # Decoder path with skip connections
        d3 = self.upsample(b)
        d3 = np.concatenate([d3, e3], axis=1)  # Skip connection
        d3 = self.dec3.forward(d3)

        d2 = self.upsample(d3)
        d2 = np.concatenate([d2, e2], axis=1)
        d2 = self.dec2.forward(d2)

        d1 = self.upsample(d2)
        d1 = np.concatenate([d1, e1], axis=1)
        d1 = self.dec1.forward(d1)

        # Output convolution
        out = self.conv1x1(d1, self.out_conv)

        return out

    def max_pool(self, x, size=2):
        """2x2 max pooling."""
        N, C, H, W = x.shape
        out = np.zeros((N, C, H//size, W//size))
        for i in range(H//size):
            for j in range(W//size):
                out[:, :, i, j] = np.max(
                    x[:, :, i*size:(i+1)*size, j*size:(j+1)*size],
                    axis=(2, 3)
                )
        return out

    def upsample(self, x, scale=2):
        """Simple upsampling by repeating."""
        return np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)

    def conv1x1(self, x, W):
        """1x1 convolution for channel mixing."""
        return np.einsum('nchw,fchw->nfhw', x, W)


#==============================================================================
# EXERCISE 3: SIMPLE VAE
#==============================================================================

class SimpleVAE:
    """
    Simple Variational Autoencoder.
    """

    def __init__(self, input_dim, hidden_dim=256, latent_dim=32):
        """
        Initialize VAE.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        scale = np.sqrt(2.0 / input_dim)
        self.W_enc = np.random.randn(input_dim, hidden_dim) * scale
        self.b_enc = np.zeros(hidden_dim)

        # Latent space parameters
        scale = np.sqrt(2.0 / hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * scale
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * scale
        self.b_logvar = np.zeros(latent_dim)

        # Decoder
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * scale
        self.b_dec1 = np.zeros(hidden_dim)
        scale = np.sqrt(2.0 / hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * scale
        self.b_dec2 = np.zeros(input_dim)

    def encode(self, x):
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input of shape (N, input_dim)

        Returns:
            mu: Mean of shape (N, latent_dim)
            logvar: Log variance of shape (N, latent_dim)
        """
        # Hidden layer
        h = np.maximum(0, x @ self.W_enc + self.b_enc)  # ReLU

        # Latent parameters
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean
            logvar: Log variance

        Returns:
            z: Sampled latent vector
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z):
        """
        Decode latent vector to reconstruction.

        Args:
            z: Latent vector of shape (N, latent_dim)

        Returns:
            x_recon: Reconstruction of shape (N, input_dim)
        """
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)  # ReLU
        x_recon = 1 / (1 + np.exp(-(h @ self.W_dec2 + self.b_dec2)))  # Sigmoid
        return x_recon

    def forward(self, x):
        """
        Full forward pass.

        Returns:
            x_recon: Reconstruction
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(self, x, x_recon, mu, logvar):
        """
        VAE loss = Reconstruction + KL divergence

        Args:
            x: Original input
            x_recon: Reconstruction
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
        """
        # Reconstruction loss (binary cross-entropy)
        recon_loss = -np.mean(
            x * np.log(x_recon + 1e-8) + (1 - x) * np.log(1 - x_recon + 1e-8)
        )

        # KL divergence
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))

        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss

    def generate(self, num_samples=10):
        """
        Generate new samples from the prior.

        Args:
            num_samples: Number of samples to generate

        Returns:
            samples: Generated samples of shape (num_samples, input_dim)
        """
        # Sample from prior N(0, I)
        z = np.random.randn(num_samples, self.latent_dim)

        # Decode
        samples = self.decode(z)

        return samples
```

---

## Business Applications

### Comprehensive Computer Vision System

```python
#==============================================================================
# BUSINESS APPLICATION: Automated Visual Inspection System
#==============================================================================

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Detection:
    """Single detection result."""
    x: int
    y: int
    width: int
    height: int
    class_name: str
    confidence: float
    defect_type: Optional[str] = None


class VisualInspectionSystem:
    """
    Automated visual inspection system for manufacturing.

    Use Cases:
    - Quality control on production lines
    - Defect detection in products
    - Package inspection
    - Surface inspection
    """

    DEFECT_TYPES = ['scratch', 'dent', 'discoloration', 'crack', 'missing_part']
    SEVERITY_LEVELS = ['minor', 'moderate', 'critical']

    def __init__(self):
        """Initialize the inspection system."""
        self.detector = SlidingWindowDetector(
            classifier=self._classify_patch,
            window_sizes=[(32, 32), (64, 64), (128, 128)],
            stride=16,
            threshold=0.3
        )

        # Quality thresholds
        self.thresholds = {
            'pass': 0.95,  # No defects above this
            'review': 0.80,  # Needs human review
            'fail': 0.60    # Definite failure
        }

    def _classify_patch(self, patch: np.ndarray) -> Tuple[str, float]:
        """Classify a single image patch."""
        # Simplified defect detection based on variance and edges
        gray = np.mean(patch, axis=2) if len(patch.shape) == 3 else patch

        # Features
        variance = np.var(gray)
        edge_strength = np.mean(np.abs(np.diff(gray, axis=0))) + \
                       np.mean(np.abs(np.diff(gray, axis=1)))

        # Simple heuristic (replace with trained model)
        if variance > 1000 and edge_strength > 20:
            return 'defect', min(0.9, variance / 2000)
        elif variance > 500:
            return 'possible_defect', 0.5
        else:
            return 'ok', 0.1

    def inspect(self, image: np.ndarray) -> Dict:
        """
        Perform complete inspection on an image.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Comprehensive inspection report
        """
        # Run detection
        detections = self.detector.detect(image)

        # Analyze detections
        defects = []
        for det in detections:
            if det['class'] in ['defect', 'possible_defect']:
                defect = Detection(
                    x=det['x'],
                    y=det['y'],
                    width=det['width'],
                    height=det['height'],
                    class_name=det['class'],
                    confidence=det['score'],
                    defect_type=self._classify_defect_type(image, det)
                )
                defects.append(defect)

        # Calculate quality score
        quality_score = self._calculate_quality_score(image, defects)

        # Determine decision
        if quality_score >= self.thresholds['pass']:
            decision = 'PASS'
        elif quality_score >= self.thresholds['review']:
            decision = 'REVIEW'
        else:
            decision = 'FAIL'

        return {
            'quality_score': quality_score,
            'decision': decision,
            'num_defects': len(defects),
            'defects': [self._defect_to_dict(d) for d in defects],
            'summary': self._generate_summary(defects, quality_score, decision)
        }

    def _classify_defect_type(self, image: np.ndarray, detection: Dict) -> str:
        """Classify the type of defect."""
        # Extract patch
        x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
        patch = image[y:y+h, x:x+w]

        if len(patch.shape) == 3:
            gray = np.mean(patch, axis=2)
        else:
            gray = patch

        # Simple heuristic classification
        edges = np.abs(np.diff(gray, axis=0))
        horizontal_edges = np.mean(np.abs(np.diff(gray, axis=0)))
        vertical_edges = np.mean(np.abs(np.diff(gray, axis=1)))

        variance = np.var(gray)
        mean_intensity = np.mean(gray)

        # Classify based on features
        if horizontal_edges > vertical_edges * 1.5 or vertical_edges > horizontal_edges * 1.5:
            return 'scratch'
        elif variance < 100:
            return 'discoloration'
        elif np.max(edges) > 100:
            return 'crack'
        else:
            return 'dent'

    def _calculate_quality_score(self, image: np.ndarray, defects: List[Detection]) -> float:
        """Calculate overall quality score."""
        if len(defects) == 0:
            return 1.0

        # Base score
        score = 1.0

        # Penalty for each defect
        for defect in defects:
            severity = self._get_defect_severity(defect)
            if severity == 'critical':
                score -= 0.3
            elif severity == 'moderate':
                score -= 0.15
            else:
                score -= 0.05

        # Penalty for total defect area
        total_area = sum(d.width * d.height for d in defects)
        image_area = image.shape[0] * image.shape[1]
        area_ratio = total_area / image_area
        score -= area_ratio * 0.5

        return max(0, min(1, score))

    def _get_defect_severity(self, defect: Detection) -> str:
        """Determine severity of a defect."""
        area = defect.width * defect.height

        if defect.defect_type == 'crack':
            return 'critical'
        elif defect.defect_type == 'missing_part':
            return 'critical'
        elif area > 10000:
            return 'critical'
        elif area > 2500 or defect.confidence > 0.8:
            return 'moderate'
        else:
            return 'minor'

    def _defect_to_dict(self, defect: Detection) -> Dict:
        """Convert defect to dictionary."""
        return {
            'location': {'x': defect.x, 'y': defect.y},
            'size': {'width': defect.width, 'height': defect.height},
            'type': defect.defect_type,
            'confidence': defect.confidence,
            'severity': self._get_defect_severity(defect)
        }

    def _generate_summary(self, defects: List[Detection], score: float, decision: str) -> str:
        """Generate human-readable summary."""
        if len(defects) == 0:
            return f"No defects detected. Quality score: {score:.1%}. Decision: {decision}"

        defect_counts = {}
        for d in defects:
            defect_counts[d.defect_type] = defect_counts.get(d.defect_type, 0) + 1

        defect_summary = ", ".join([f"{count} {dtype}" for dtype, count in defect_counts.items()])

        return f"Found {len(defects)} defects ({defect_summary}). Quality score: {score:.1%}. Decision: {decision}"

    def batch_inspect(self, images: List[np.ndarray]) -> Dict:
        """
        Inspect multiple images and generate batch report.

        Args:
            images: List of images to inspect

        Returns:
            Batch inspection report
        """
        results = [self.inspect(img) for img in images]

        # Aggregate statistics
        pass_count = sum(1 for r in results if r['decision'] == 'PASS')
        review_count = sum(1 for r in results if r['decision'] == 'REVIEW')
        fail_count = sum(1 for r in results if r['decision'] == 'FAIL')

        avg_score = np.mean([r['quality_score'] for r in results])
        total_defects = sum(r['num_defects'] for r in results)

        # Defect type distribution
        defect_types = {}
        for r in results:
            for d in r['defects']:
                dtype = d['type']
                defect_types[dtype] = defect_types.get(dtype, 0) + 1

        return {
            'total_inspected': len(images),
            'pass_count': pass_count,
            'review_count': review_count,
            'fail_count': fail_count,
            'pass_rate': pass_count / len(images),
            'average_quality_score': avg_score,
            'total_defects_found': total_defects,
            'defect_distribution': defect_types,
            'individual_results': results
        }


# Demo
def demo_inspection_system():
    """Demonstrate the visual inspection system."""

    inspector = VisualInspectionSystem()

    print("=" * 60)
    print("VISUAL INSPECTION SYSTEM DEMO")
    print("=" * 60)

    # Create test images
    np.random.seed(42)

    # Good product (low variance, uniform)
    good_image = np.ones((256, 256, 3)) * 128 + np.random.randn(256, 256, 3) * 5

    # Defective product (with scratch)
    defective_image = np.ones((256, 256, 3)) * 128 + np.random.randn(256, 256, 3) * 5
    defective_image[100:110, 50:200, :] = 50  # Scratch

    # Inspect good product
    print("\nInspecting Good Product:")
    print("-" * 40)
    result = inspector.inspect(good_image.astype(np.uint8))
    print(f"Quality Score: {result['quality_score']:.1%}")
    print(f"Decision: {result['decision']}")
    print(f"Summary: {result['summary']}")

    # Inspect defective product
    print("\nInspecting Defective Product:")
    print("-" * 40)
    result = inspector.inspect(defective_image.astype(np.uint8))
    print(f"Quality Score: {result['quality_score']:.1%}")
    print(f"Decision: {result['decision']}")
    print(f"Defects Found: {result['num_defects']}")
    print(f"Summary: {result['summary']}")

    # Batch inspection
    print("\n" + "=" * 60)
    print("BATCH INSPECTION")
    print("=" * 60)

    batch = [good_image.astype(np.uint8)] * 8 + [defective_image.astype(np.uint8)] * 2
    batch_result = inspector.batch_inspect(batch)

    print(f"\nTotal Inspected: {batch_result['total_inspected']}")
    print(f"Pass Rate: {batch_result['pass_rate']:.1%}")
    print(f"Pass: {batch_result['pass_count']}, Review: {batch_result['review_count']}, Fail: {batch_result['fail_count']}")
    print(f"Average Quality: {batch_result['average_quality_score']:.1%}")


if __name__ == '__main__':
    demo_inspection_system()
```

---

## Summary: Week 8-10 Checklist

### Concepts You Should Understand
- [ ] Object detection (two-stage vs. one-stage)
- [ ] Region proposals and anchor boxes
- [ ] Non-maximum suppression
- [ ] Semantic vs. instance segmentation
- [ ] U-Net architecture and skip connections
- [ ] VAE encoder-decoder structure
- [ ] Reparameterization trick
- [ ] GAN generator-discriminator training
- [ ] Mode collapse and training instability
- [ ] Diffusion forward and reverse processes
- [ ] Reinforcement learning basics (MDP, policy, value)
- [ ] Deep Q-learning
- [ ] LLM architecture and prompting

### Skills You Should Have
- [ ] Implement sliding window detection
- [ ] Implement NMS
- [ ] Build U-Net for segmentation
- [ ] Implement VAE
- [ ] Train simple GAN
- [ ] Understand diffusion sampling
- [ ] Apply LLMs effectively

### Key Formulas
```
IoU = Intersection / Union

VAE Loss = Reconstruction + KL
KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)

GAN: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]

Diffusion: q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t I)

Q-Learning: Q(s,a) = r + γ max_a' Q(s',a')
```

---

## Course Complete!

Congratulations on completing this comprehensive deep learning course. You now have the knowledge to:

1. **Build and train neural networks** from scratch
2. **Implement CNNs** for image classification and processing
3. **Work with sequences** using RNNs, LSTMs, and Transformers
4. **Detect and segment objects** in images
5. **Generate new content** with VAEs, GANs, and diffusion models
6. **Apply deep learning** to real business problems

### Next Steps
- Implement projects from scratch (no frameworks first)
- Then move to PyTorch/TensorFlow for production
- Read recent papers (arxiv.org)
- Participate in Kaggle competitions
- Build a portfolio of projects
- Stay current with rapid advances in the field

Good luck on your deep learning journey!
