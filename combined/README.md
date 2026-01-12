# Combined Deep Learning Course Materials

This folder contains unified learning materials that merge content from:
- **CSE 493G1** course structure and lesson plans
- **CS231n** Stanford course notes and detailed explanations
- **CS231n 2025 lecture slides** extracted content

## Document Overview

```
combined/
├── README.md                        ← This file
├── Week_01_02_Foundations.md        ← Classification, k-NN, Linear, Neural Networks
├── Week_03_04_CNNs_Training.md      ← CNNs, Pooling, Training, Optimization
├── Week_05_06_SSL_RNNs.md           ← Self-Supervised Learning, RNNs, LSTM
├── Week_07_Attention_Transformers.md ← Attention, Transformers, ViT
├── Week_08_10_Advanced_Topics.md    ← Detection, Segmentation, GANs, Diffusion
└── Week_11_LLMs.md                  ← Tokenization, Pre-training, RLHF, RAG
```

## Learning Path

```
                    START HERE
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Week 1-2: Foundations       │
        │   • Classification basics     │
        │   • Neural network intuition  │
        │   • Backpropagation           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Week 3-4: CNNs & Training   │
        │   • Convolution operations    │
        │   • Modern architectures      │
        │   • Training techniques       │
        └───────────────┬───────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐
│ Week 5-6: SSL/RNNs  │   │ Week 7: Attention   │
│ • Contrastive       │   │ • Self-attention    │
│ • Sequences         │   │ • Transformers      │
│ • LSTM/GRU          │   │ • ViT               │
└─────────┬───────────┘   └───────────┬─────────┘
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Week 8-10: Advanced         │
        │   • Detection/Segmentation    │
        │   • Generative models         │
        │   • 3D vision                 │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Week 11: LLMs               │
        │   • Pre-training & scaling    │
        │   • RLHF & alignment          │
        │   • RAG & agents              │
        └───────────────────────────────┘
```

## Key Features of Each Document

### Week 1-2: Foundations
- Image classification and data-driven approach
- k-NN classifier and distance metrics
- Linear classifiers (SVM, Softmax)
- Loss functions and regularization
- Neural networks and activation functions
- Backpropagation and gradient descent

### Week 3-4: CNNs & Training
- Convolution operation and parameter sharing
- Pooling layers and spatial reduction
- Classic architectures (AlexNet, VGG, ResNet)
- Batch normalization and dropout
- Data augmentation
- Optimizers (SGD, Adam) and learning rate schedules

### Week 5-6: SSL & RNNs
- Self-supervised learning paradigm
- Contrastive learning (SimCLR, MoCo)
- Recurrent neural networks
- Vanishing/exploding gradients
- LSTM and GRU architectures

### Week 7: Attention & Transformers
- Attention mechanism (Q, K, V)
- Self-attention and multi-head attention
- Transformer architecture
- Positional encoding
- Vision Transformers (ViT)
- Encoder-only vs decoder-only

### Week 8-10: Advanced Topics
- Object detection (R-CNN, YOLO)
- Semantic and instance segmentation
- Generative models (VAE, GAN, Diffusion)
- Reinforcement learning basics
- 3D vision and NeRF

### Week 11: LLMs
- Tokenization (BPE)
- Pre-training and scaling laws
- Fine-tuning (LoRA, QLoRA)
- RLHF and DPO
- Prompting techniques (CoT, few-shot)
- RAG and agents

## Related Materials

### In this repository:
- `DEEP_LEARNING_UNIFIED_INTUITION.md` - Conceptual connections
- `DEEP_LEARNING_PITFALLS_GUIDE.md` - Common mistakes
- `DEEP_LEARNING_IN_PRODUCTION.md` - Production systems
- `BUILDING_ML_ORGANIZATIONS.md` - Teams and hiring
- `visualizations/` - Interactive ASCII visualizers

### CS231n source materials:
- `cs231n/` - Stanford course website clone (submodule)
- `cs231n/slides_2025/` - 2025 lecture PDFs and extractions

## How to Use These Materials

**For students:**
1. Read week documents in order
2. Run visualizations for interactive learning
3. Refer to pitfalls guide during assignments
4. Use production doc for capstone projects

**For practitioners:**
1. Use as quick reference for concepts
2. Check architecture diagrams
3. Review training guidelines
4. Reference for interview prep

**For educators:**
1. Adapt materials for your course
2. Use ASCII diagrams in slides
3. Assign readings by week

## Sources and Attribution

Content synthesized from:
- CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)
- CSE 493G1: Deep Learning (University of Washington)
- Original course lecture slides (2025)
- CS231n course notes (https://cs231n.github.io/)

All ASCII diagrams created for educational purposes.
