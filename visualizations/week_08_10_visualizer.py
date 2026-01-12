"""
Week 8-10 Visualizer: Advanced Topics
Interactive ASCII visualizations for advanced deep learning.
Run: python week_08_10_visualizer.py
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
# VISUALIZATION 1: Generative vs Discriminative Models
# ============================================================

def visualize_gen_vs_disc():
    print_header("VISUALIZATION 1: Generative vs Discriminative")

    print("""
Two Approaches to Machine Learning
═══════════════════════════════════════════════════════════

DISCRIMINATIVE: Learn decision boundary
    "Given X, predict Y"

    ┌─────────────────────────────────────┐
    │    ○ ○ ○ ○        × × × ×          │
    │  ○ ○ ○ ○ ○ ╲     × × × × ×        │
    │    ○ ○ ○ ○  ╲    × × × × ×        │
    │  ○ ○ ○ ○ ○   ╲    × × × ×         │
    │    ○ ○ ○      ╲     × × ×          │
    │               ╲                     │
    │          Decision boundary          │
    └─────────────────────────────────────┘
    Models: Logistic Regression, SVM, Neural Nets
    Task: Classification, Regression

GENERATIVE: Learn data distribution
    "Learn P(X) or P(X|Y)"

    ┌─────────────────────────────────────┐
    │                                     │
    │         ∩∩∩∩                        │
    │        ╱    ╲   P(X)                │
    │       ╱      ╲                      │
    │      ╱        ╲                     │
    │    ─╯          ╰─                   │
    │                                     │
    └─────────────────────────────────────┘
    Models: VAE, GAN, Diffusion
    Task: Generate new samples from the distribution
    """)
    pause()

# ============================================================
# VISUALIZATION 2: Variational Autoencoder (VAE)
# ============================================================

def visualize_vae():
    print_header("VISUALIZATION 2: Variational Autoencoder")

    print("""
VAE: Learn a latent representation for generation
═══════════════════════════════════════════════════════════

    Input X                                        Output X'
    ┌─────┐                                        ┌─────┐
    │ 🐱  │                                        │ 🐱  │
    │     │                                        │     │
    └──┬──┘                                        └──▲──┘
       │                                              │
       ▼                                              │
    ┌──────────┐    ┌─────────────────┐    ┌──────────┐
    │ ENCODER  │───▶│  Latent Space   │───▶│ DECODER  │
    │ q(z|x)   │    │       z         │    │ p(x|z)   │
    └──────────┘    └─────────────────┘    └──────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ μ, σ (mean, │
                    │ variance)   │
                    └─────────────┘

    Key insight: Encoder outputs μ and σ, then SAMPLE z ~ N(μ, σ)
    """)
    pause()

    print("""
The Reparameterization Trick
═══════════════════════════════════════════════════════════

Problem: Can't backprop through random sampling!

    z = sample(μ, σ)  ← No gradient through this!

Solution: Reparameterize

    ε ~ N(0, 1)       ← Sample from standard normal
    z = μ + σ × ε     ← Now z is differentiable w.r.t. μ, σ

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Encoder                                           │
    │     │                                               │
    │     ├──▶ μ ──────┐                                 │
    │     │            ├──▶ z = μ + σ × ε ──▶ Decoder    │
    │     └──▶ σ ──────┘            ▲                    │
    │                               │                    │
    │                         ε ~ N(0,1)                 │
    │                         (no gradient)              │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

    print("""
VAE Loss Function
═══════════════════════════════════════════════════════════

    Loss = Reconstruction Loss + KL Divergence

    L = ||x - x'||² + KL(q(z|x) || p(z))

    ┌──────────────────────┐    ┌──────────────────────┐
    │ Reconstruction Loss  │    │ KL Divergence        │
    │                      │    │                      │
    │ Make output look     │    │ Make latent space    │
    │ like input           │    │ look like standard   │
    │                      │    │ normal N(0,1)        │
    │                      │    │                      │
    │ "Be accurate"        │    │ "Be regular"         │
    └──────────────────────┘    └──────────────────────┘

    Without KL: Latent space is messy, can't sample from it
    With KL: Latent space is organized, can sample new z!

    Generation: z ~ N(0,1) → Decoder → New sample!
    """)
    pause()

# ============================================================
# VISUALIZATION 3: Generative Adversarial Network (GAN)
# ============================================================

def visualize_gan():
    print_header("VISUALIZATION 3: Generative Adversarial Network")

    print("""
GAN: Two Networks Playing a Game
═══════════════════════════════════════════════════════════

    Generator (G):  Create fake samples from noise
    Discriminator (D): Distinguish real from fake

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Random        ┌───────────┐   Fake    ┌─────────┐│
    │   Noise z ─────▶│ Generator │──────────▶│         ││
    │                 └───────────┘           │Discrim- ││
    │                                         │inator   ││──▶ Real?
    │   Real          ┌───────────┐   Real    │         ││     Fake?
    │   Data  ───────▶│           │──────────▶│         ││
    │                 └───────────┘           └─────────┘│
    │                                                     │
    └─────────────────────────────────────────────────────┘

    G tries to FOOL D (generate realistic samples)
    D tries to CATCH G (correctly classify real vs fake)
    """)
    pause()

    # Animated GAN training
    print_header("VISUALIZATION 3: GAN Training Animation")

    stages = [
        ("Early Training", """
    Generator output:         Real data:
    ┌──────────────┐         ┌──────────────┐
    │   ???  ☒☒    │         │     🐱       │
    │  ☒☒  ???     │         │              │
    │    ???       │         │              │
    └──────────────┘         └──────────────┘
    D says: "Obviously fake!"  (D accuracy: 99%)
        """),
        ("Mid Training", """
    Generator output:         Real data:
    ┌──────────────┐         ┌──────────────┐
    │     △        │         │     🐱       │
    │    /|\\       │         │              │
    │              │         │              │
    └──────────────┘         └──────────────┘
    D says: "Probably fake"   (D accuracy: 75%)
        """),
        ("Good Training", """
    Generator output:         Real data:
    ┌──────────────┐         ┌──────────────┐
    │     🐱?      │         │     🐱       │
    │              │         │              │
    │              │         │              │
    └──────────────┘         └──────────────┘
    D says: "Hmm... 50/50"    (D accuracy: ~50%)
    G has learned to generate realistic samples!
        """),
    ]

    for title, diagram in stages:
        clear_screen()
        print("=" * 60)
        print(f"{'GAN TRAINING PROGRESS':^60}")
        print("=" * 60)
        print(f"\n{title}")
        print(diagram)
        time.sleep(2)

    pause()

    print("""
GAN Training Dynamics
═══════════════════════════════════════════════════════════

    Minimax Game:

    min_G max_D  V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
                         ↑              ↑
                    D wants high    G wants low
                    (classify real)  (fool D)

    Training loop:
    1. Train D to better distinguish real vs G(z)
    2. Train G to better fool D
    3. Repeat

    Equilibrium: D outputs 0.5 for everything
               (can't tell real from fake!)

    Challenges:
    - Mode collapse (G generates same thing)
    - Training instability
    - Hard to evaluate quality
    """)
    pause()

# ============================================================
# VISUALIZATION 4: Diffusion Models
# ============================================================

def visualize_diffusion():
    print_header("VISUALIZATION 4: Diffusion Models")

    print("""
Diffusion: Gradually Add then Remove Noise
═══════════════════════════════════════════════════════════

Forward process: Add noise step by step until pure noise

    x_0       →  x_1       →  x_2       →  ...  →  x_T
    ┌─────┐     ┌─────┐     ┌─────┐            ┌─────┐
    │ 🐱  │  →  │ 🐱. │  →  │ 🐱.. │  →  ...  →│ ... │
    │     │     │  .  │     │ ..   │            │ ... │
    └─────┘     └─────┘     └─────┘            └─────┘
    Clean       Slightly    More              Pure
    image       noisy       noisy             noise

Reverse process: Learn to DENOISE step by step

    x_T       →  x_{T-1}   →  x_{T-2}   →  ...  →  x_0
    ┌─────┐     ┌─────┐     ┌─────┐            ┌─────┐
    │ ... │  →  │ ... │  →  │ 🐱.. │  →  ...  →│ 🐱  │
    │ ... │     │ ..  │     │  .   │            │     │
    └─────┘     └─────┘     └─────┘            └─────┘
    Pure        Less        Even              Clean
    noise       noisy       less              image!
    """)
    pause()

    # Animated diffusion
    print_header("VISUALIZATION 4: Diffusion Process Animation")

    forward_steps = [
        ("t=0 (Original)", """
    ┌─────────────────┐
    │     ○   ○       │
    │       ▽         │
    │      ───        │
    │     /   \\       │
    │    (face)       │
    └─────────────────┘
        """),
        ("t=250 (Light noise)", """
    ┌─────────────────┐
    │  .  ○ . ○  .    │
    │    .  ▽  .      │
    │   .  ─── .      │
    │  .  /   \\ .     │
    │   . (face) .    │
    └─────────────────┘
        """),
        ("t=500 (Medium noise)", """
    ┌─────────────────┐
    │ . . ○ . ○ . .   │
    │  . . . ▽ . . .  │
    │ . . . ─ . ─ . . │
    │  . . / . \\ . .  │
    │ . . . . . . . . │
    └─────────────────┘
        """),
        ("t=750 (Heavy noise)", """
    ┌─────────────────┐
    │ . . . . . . . . │
    │ . . . . . . . . │
    │ . . . . . . . . │
    │ . . . . . . . . │
    │ . . . . . . . . │
    └─────────────────┘
        """),
        ("t=1000 (Pure noise)", """
    ┌─────────────────┐
    │ ░▒░▓░▒░▓░▒░▓░▒░ │
    │ ▓░▒░▓░▒░▓░▒░▓░▒ │
    │ ░▓░▒░▓░▒░▓░▒░▓░ │
    │ ▒░▓░▒░▓░▒░▓░▒░▓ │
    │ ░▒░▓░▒░▓░▒░▓░▒░ │
    └─────────────────┘
        """),
    ]

    print("FORWARD PROCESS (Adding noise):\n")
    for title, diagram in forward_steps:
        clear_screen()
        print("=" * 60)
        print(f"{'FORWARD DIFFUSION':^60}")
        print("=" * 60)
        print(f"\n{title}")
        print(diagram)
        time.sleep(1)

    print("\n\nREVERSE PROCESS (Denoising - what the model learns):\n")

    for title, diagram in reversed(forward_steps):
        clear_screen()
        print("=" * 60)
        print(f"{'REVERSE DIFFUSION (GENERATION)':^60}")
        print("=" * 60)
        print(f"\n{title}")
        print(diagram)
        time.sleep(1)

    pause()

    print("""
Diffusion Training Objective
═══════════════════════════════════════════════════════════

Train a neural network ε_θ to predict the noise added:

    L = ||ε - ε_θ(x_t, t)||²

    Where:
    - ε is the actual noise added at step t
    - ε_θ predicts what noise was added
    - x_t is the noisy image at step t
    - t is the timestep

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Input: (noisy image, timestep) ─────▶ ε_θ         │
    │                                          │          │
    │                                          ▼          │
    │  Output: Predicted noise ε̂                         │
    │                                                     │
    │  Loss: Mean squared error between ε and ε̂          │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    Generation: Start from pure noise, iteratively denoise
    """)
    pause()

# ============================================================
# VISUALIZATION 5: Model Interpretability
# ============================================================

def visualize_interpretability():
    print_header("VISUALIZATION 5: Model Interpretability")

    print("""
Why Interpretability Matters
═══════════════════════════════════════════════════════════

    "The model says you have cancer."
    "Why?"
    "I don't know, it's a black box."

    Not acceptable for high-stakes decisions!

Methods for Understanding Models:

    1. FEATURE IMPORTANCE
       Which inputs matter most?

    2. SALIENCY MAPS
       Which pixels influence the prediction?

    3. ATTENTION VISUALIZATION
       What is the model "looking at"?

    4. PROBING
       What information is encoded in representations?
    """)
    pause()

    print("""
Saliency Maps
═══════════════════════════════════════════════════════════

Compute gradient of output w.r.t. input pixels:

    saliency = |∂output / ∂input|

    Original Image:        Saliency Map:
    ┌──────────────┐      ┌──────────────┐
    │              │      │              │
    │     🐱       │      │     ████     │
    │    /||\\      │  →   │    ████     │
    │   / || \\     │      │   ░████░    │
    │              │      │              │
    └──────────────┘      └──────────────┘

    Bright areas = high gradient = important for prediction

    Reveals: Model focuses on cat's face for "cat" prediction
             Not on background (good!)
    """)
    pause()

    print("""
Attention Visualization
═══════════════════════════════════════════════════════════

For Transformers, visualize attention weights:

    Input: "The cat sat on the mat"

    Attention pattern for word "sat":
    ┌──────────────────────────────────────┐
    │ The   cat   sat   on   the   mat    │
    │  ░     ███   ██    ░    ░    ░      │
    │ 0.05  0.45  0.40  0.05 0.03 0.02    │
    └──────────────────────────────────────┘

    "sat" attends strongly to "cat" (subject-verb relation!)

Different heads learn different patterns:
    - Head 1: Subject-verb
    - Head 2: Noun-noun
    - Head 3: Adjacent words
    - Head 4: Beginning of sentence
    """)
    pause()

# ============================================================
# VISUALIZATION 6: Ethics and Bias
# ============================================================

def visualize_ethics():
    print_header("VISUALIZATION 6: Ethics and Bias in AI")

    print("""
Sources of Bias in ML
═══════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────┐
    │                    DATA BIAS                        │
    │                                                     │
    │  Training data reflects historical biases           │
    │                                                     │
    │  Example: Resume screening                          │
    │    Past hires: 90% male                            │
    │    Model learns: "male names → higher score"        │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │                 REPRESENTATION BIAS                 │
    │                                                     │
    │  Some groups underrepresented in data               │
    │                                                     │
    │  Example: Face recognition                          │
    │    Training data: 80% light skin                   │
    │    Model accuracy: High for light, low for dark    │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │                  MEASUREMENT BIAS                   │
    │                                                     │
    │  What we measure may be biased proxy                │
    │                                                     │
    │  Example: "Predict employee success"                │
    │    Measured by: manager ratings                     │
    │    But managers may rate biased                     │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

    print("""
Fairness Metrics
═══════════════════════════════════════════════════════════

Multiple (sometimes conflicting!) definitions of fair:

    1. DEMOGRAPHIC PARITY
       P(positive | group A) = P(positive | group B)
       Same rate of positive outcomes

    2. EQUALIZED ODDS
       Same true positive and false positive rates
       P(ŷ=1 | y=1, A) = P(ŷ=1 | y=1, B)  (TPR)
       P(ŷ=1 | y=0, A) = P(ŷ=1 | y=0, B)  (FPR)

    3. INDIVIDUAL FAIRNESS
       Similar individuals get similar predictions
       d(x₁, x₂) small → d(ŷ₁, ŷ₂) small

    ┌─────────────────────────────────────────────────────┐
    │  IMPOSSIBILITY THEOREM:                             │
    │  You cannot satisfy all fairness metrics            │
    │  simultaneously (except in trivial cases).          │
    │                                                     │
    │  Must choose which type of fairness matters         │
    │  for your specific application.                     │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

    print("""
Mitigation Strategies
═══════════════════════════════════════════════════════════

    PRE-PROCESSING:
    ├── Collect more diverse data
    ├── Reweight samples to balance groups
    └── Remove or blind protected attributes

    IN-PROCESSING:
    ├── Add fairness constraints to loss function
    ├── Adversarial debiasing
    └── Fair representation learning

    POST-PROCESSING:
    ├── Adjust thresholds per group
    ├── Reject option classification
    └── Calibrate probabilities

    ┌─────────────────────────────────────────────────────┐
    │  BEST PRACTICE:                                     │
    │  1. Define fairness criteria BEFORE building        │
    │  2. Audit model on protected groups                 │
    │  3. Document limitations and biases                 │
    │  4. Monitor deployed model for drift                │
    │  5. Create feedback mechanisms                      │
    └─────────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# VISUALIZATION 7: Model Deployment
# ============================================================

def visualize_deployment():
    print_header("VISUALIZATION 7: Model Deployment")

    print("""
From Notebook to Production
═══════════════════════════════════════════════════════════

    Research                        Production
    ┌──────────────┐               ┌──────────────┐
    │ Jupyter      │               │ API Server   │
    │ Notebook     │  ─────────▶   │ + Monitoring │
    │              │               │ + Scaling    │
    │ model.train()│               │ + Versioning │
    └──────────────┘               └──────────────┘
        Works on                      Works for
        my laptop                     millions!

Production Requirements:
    □ Low latency (< 100ms response)
    □ High throughput (1000s req/sec)
    □ High availability (99.9% uptime)
    □ Monitoring and alerting
    □ A/B testing capability
    □ Rollback mechanism
    """)
    pause()

    print("""
Deployment Architecture
═══════════════════════════════════════════════════════════

                    ┌─────────────────────────────────────┐
                    │           Load Balancer             │
                    └───────────────┬─────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   Model      │     │   Model      │     │   Model      │
    │   Server 1   │     │   Server 2   │     │   Server 3   │
    │              │     │              │     │              │
    │   GPU: A100  │     │   GPU: A100  │     │   GPU: A100  │
    └──────────────┘     └──────────────┘     └──────────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                    ┌───────────────┴─────────────────────┐
                    │         Model Registry              │
                    │    (versioned model storage)        │
                    └─────────────────────────────────────┘

Horizontal scaling: Add more model servers as traffic grows
    """)
    pause()

    print("""
Optimization for Production
═══════════════════════════════════════════════════════════

    1. QUANTIZATION
       FP32 → INT8: 4x smaller, 2-4x faster
       ┌────────────────────────────────┐
       │ 3.14159... → 3 (rounded)       │
       │ Tiny accuracy loss, big speedup│
       └────────────────────────────────┘

    2. PRUNING
       Remove unimportant weights
       ┌────────────────────────────────┐
       │ Before: ○──○──○──○            │
       │ After:  ○──○  ○──○  (sparse)  │
       └────────────────────────────────┘

    3. KNOWLEDGE DISTILLATION
       Train small model to mimic large model
       ┌────────────────────────────────┐
       │ Teacher (BERT-large) → Student │
       │         (BERT-tiny)            │
       └────────────────────────────────┘

    4. BATCHING
       Process multiple requests together
       ┌────────────────────────────────┐
       │ Single: [1] [1] [1] [1] = 4 calls │
       │ Batch:  [1,1,1,1] = 1 call     │
       └────────────────────────────────┘
    """)
    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print_header("WEEK 8-10: ADVANCED TOPICS")
        print("""
Choose a visualization:

    [1] Generative vs Discriminative
        - Two paradigms of ML

    [2] Variational Autoencoder (VAE)
        - Latent space, reparameterization trick

    [3] Generative Adversarial Network (GAN)
        - Generator vs discriminator game

    [4] Diffusion Models
        - Forward and reverse process, denoising

    [5] Model Interpretability
        - Saliency maps, attention visualization

    [6] Ethics and Bias
        - Sources of bias, fairness metrics

    [7] Model Deployment
        - Production architecture, optimization

    [A] Run ALL visualizations

    [Q] Quit
        """)

        choice = input("Enter choice: ").strip().upper()

        if choice == '1':
            visualize_gen_vs_disc()
        elif choice == '2':
            visualize_vae()
        elif choice == '3':
            visualize_gan()
        elif choice == '4':
            visualize_diffusion()
        elif choice == '5':
            visualize_interpretability()
        elif choice == '6':
            visualize_ethics()
        elif choice == '7':
            visualize_deployment()
        elif choice == 'A':
            visualize_gen_vs_disc()
            visualize_vae()
            visualize_gan()
            visualize_diffusion()
            visualize_interpretability()
            visualize_ethics()
            visualize_deployment()
        elif choice == 'Q':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
