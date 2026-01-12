# Module 7: Generative Models

## Learning Objectives

By the end of this module, you will understand:
- The fundamental goal of generative modeling and different approaches
- Autoencoders and their limitations for generation
- Variational Autoencoders (VAE): architecture, ELBO, and reparameterization trick
- Generative Adversarial Networks (GAN): adversarial training and challenges
- Diffusion models: forward process, reverse process, and denoising
- Latent diffusion and text-to-image generation (Stable Diffusion)
- When to use each type of generative model

---

## 7.1 What is Generative Modeling?

### The Fundamental Goal

Generative modeling aims to learn the underlying distribution of data so we can generate
new samples that look like they came from the same distribution:

```
GENERATIVE MODELING GOAL
════════════════════════

Given: Training data {x₁, x₂, ..., xₙ} sampled from unknown p_data(x)

Goal:  Learn p_model(x) such that p_model(x) ≈ p_data(x)

Then:  Sample new x ~ p_model(x) to generate new data!


VISUALIZATION:
──────────────

Training Data (faces):                Generated Samples:
┌─────────────────────┐               ┌─────────────────────┐
│  😀 😃 😄 😁 😆     │               │  🤩 🥳 😎 🤔 😊     │
│  🙂 🙃 😉 😊 😇     │    Learn      │  🤗 😋 🤭 🥸 😏     │
│  🥰 😍 🤩 😘 😗     │ ──────────►   │  🤠 🤑 😜 🤪 😝     │
│  ☺️ 😚 😙 🥲 😋     │   p_model     │  NEW FACES!         │
│  😛 😜 🤪 😝 🤑     │               │  Never seen before  │
└─────────────────────┘               └─────────────────────┘

Key insight: We don't just memorize training data.
             We learn the "rules" for what makes a valid sample.
```

### Why is Generation Hard?

High-dimensional data distributions are incredibly complex:

```
THE CURSE OF DIMENSIONALITY
═══════════════════════════

A 256×256 RGB image:
  • 256 × 256 × 3 = 196,608 dimensions
  • Each pixel: 0-255 (256 values)
  • Total possible images: 256^196,608 ≈ 10^473,000

That's more than atoms in the observable universe!

Yet only a TINY fraction of these are "valid" images:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   All possible 256×256 images                                       │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                                                             │  │
│   │                                                             │  │
│   │                        ● ← Valid faces                      │  │
│   │                                                             │  │
│   │           (Most of this space is random noise)              │  │
│   │                                                             │  │
│   │                                                             │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Generative modeling = Finding and sampling from that tiny valid region
```

### Two Approaches to Generative Modeling

```
EXPLICIT vs IMPLICIT DENSITY MODELING
═════════════════════════════════════

EXPLICIT DENSITY:
─────────────────
Define p(x) directly, optimize it to match data.

Examples:
  • Autoregressive: p(x) = ∏ p(xᵢ | x₁...xᵢ₋₁)
  • Normalizing Flows: p(x) = p(z) |det(∂f/∂z)|⁻¹
  • VAE: p(x) = ∫ p(x|z) p(z) dz (approximate)

Pros: ✓ Can compute likelihood p(x)
      ✓ Principled training objective
Cons: ✗ Constraints on model architecture
      ✗ Often lower sample quality


IMPLICIT DENSITY:
─────────────────
Learn to sample without explicitly modeling p(x).

Examples:
  • GAN: Learn generator G such that G(z) ~ p_data
  • Diffusion: Learn to reverse a noise process

Pros: ✓ Flexible architectures
      ✓ Often higher sample quality
Cons: ✗ Can't compute p(x)
      ✗ Harder to train (GAN instability)


COMPARISON:
───────────

                    Explicit           Implicit
                    ─────────          ────────
Examples            VAE, Flows         GAN, Diffusion
Likelihood p(x)     Yes                No
Sample quality      Good               Very Good
Training            Stable             Can be tricky
Architecture        Constrained        Flexible
```

### Applications of Generative Models

```
GENERATIVE MODEL APPLICATIONS
═════════════════════════════

1. IMAGE SYNTHESIS
   ───────────────
   • Art generation (DALL-E, Midjourney)
   • Face generation (StyleGAN)
   • Image editing (inpainting, super-resolution)

2. TEXT GENERATION
   ───────────────
   • Language models (GPT, LLaMA)
   • Story writing, dialogue
   • Code generation

3. AUDIO/MUSIC
   ────────────
   • Speech synthesis (WaveNet)
   • Music generation (Jukebox)
   • Voice cloning

4. VIDEO
   ─────
   • Video prediction
   • Video generation (Sora)
   • Animation

5. SCIENTIFIC
   ──────────
   • Drug discovery (molecule generation)
   • Protein structure (AlphaFold)
   • Materials design

6. DATA AUGMENTATION
   ─────────────────
   • Training data synthesis
   • Balancing datasets
   • Privacy-preserving data sharing
```

---

## 7.2 Autoencoders

### The Basic Architecture

An autoencoder learns to compress data to a low-dimensional representation and reconstruct it:

```
AUTOENCODER ARCHITECTURE
════════════════════════

    Input x (e.g., 784 pixels for MNIST)
           │
           ▼
    ┌─────────────────────────┐
    │       ENCODER           │
    │                         │
    │   784 → 256 → 64 → 32   │
    │      ↓     ↓     ↓      │
    │     ReLU  ReLU  ReLU    │
    │                         │
    └───────────┬─────────────┘
                │
                ▼
         z (latent code)
         32 dimensions
         "Compressed representation"
                │
                ▼
    ┌─────────────────────────┐
    │       DECODER           │
    │                         │
    │   32 → 64 → 256 → 784   │
    │      ↓     ↓     ↓      │
    │     ReLU  ReLU Sigmoid  │
    │                         │
    └───────────┬─────────────┘
                │
                ▼
    Output x̂ (reconstruction)


TRAINING OBJECTIVE:
───────────────────

Loss = ||x - x̂||²   (Mean Squared Error)

Minimize reconstruction error!

The bottleneck forces the network to learn
a compressed, meaningful representation.
```

### Why the Bottleneck Matters

```
THE BOTTLENECK FORCES COMPRESSION
═════════════════════════════════

Without bottleneck (identity):
  784 → 784 → 784 → 784
  Network just learns identity function
  No useful representation!

With bottleneck (compression):
  784 → 256 → 32 → 256 → 784
             ↑
        Bottleneck
        32 dimensions

The network MUST learn what features matter!

MNIST example:
  • Input: 784 pixels (28×28)
  • Latent: 32 values
  • Compression: 24× reduction

What does the latent space capture?
  • Digit identity (0-9)
  • Writing style
  • Slant, thickness
  • Position

These are the "essential" features for reconstruction.
```

### Autoencoder Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()

        # Encoder: compress input to latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # No activation - latent can be any value
        )

        # Decoder: reconstruct from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output in [0, 1] for normalized images
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


# Training
def train_autoencoder(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].view(batch[0].size(0), -1)  # Flatten

            # Forward
            x_recon = model(x)

            # Reconstruction loss
            loss = F.mse_loss(x_recon, x)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")
```

### Convolutional Autoencoder

For images, use convolutions instead of fully-connected layers:

```
CONVOLUTIONAL AUTOENCODER
═════════════════════════

    Input: 64×64×3 RGB image
           │
           ▼
    ┌─────────────────────────────────┐
    │          ENCODER                │
    │                                 │
    │   Conv 32 filters, stride 2     │  64×64×3 → 32×32×32
    │          ↓                      │
    │   Conv 64 filters, stride 2     │  32×32×32 → 16×16×64
    │          ↓                      │
    │   Conv 128 filters, stride 2    │  16×16×64 → 8×8×128
    │          ↓                      │
    │   Flatten + Dense               │  8×8×128 → 256
    │                                 │
    └────────────────┬────────────────┘
                     │
                     ▼
              z (256-dim latent)
                     │
                     ▼
    ┌─────────────────────────────────┐
    │          DECODER                │
    │                                 │
    │   Dense + Reshape               │  256 → 8×8×128
    │          ↓                      │
    │   ConvTranspose 64, stride 2    │  8×8×128 → 16×16×64
    │          ↓                      │
    │   ConvTranspose 32, stride 2    │  16×16×64 → 32×32×32
    │          ↓                      │
    │   ConvTranspose 3, stride 2     │  32×32×32 → 64×64×3
    │                                 │
    └────────────────┬────────────────┘
                     │
                     ▼
    Output: 64×64×3 reconstruction
```

### Limitations for Generation

Why can't we just sample from the autoencoder's latent space?

```
AUTOENCODER GENERATION PROBLEM
══════════════════════════════

Training: Encoder maps inputs to latent codes
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Input images          Latent space                              │
│   ┌───────────┐         ┌───────────────────────┐                 │
│   │ 🔴 🔵 🟢  │  Encoder │                       │                 │
│   │ 🟡 🟣 🟤  │ ───────► │  • • •   • •         │                 │
│   │ ⚫ ⬜ 🟥  │         │      • • •   • •      │                 │
│   └───────────┘         │                       │                 │
│                         └───────────────────────┘                 │
│                                                                   │
│   The latent space is unstructured!                               │
│   Points cluster wherever the encoder puts them.                  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘


Problem: Sampling random z doesn't work!
──────────────────────────────────────────

┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Latent space with random samples:                               │
│   ┌───────────────────────────────────────┐                       │
│   │         ★                             │                       │
│   │    • •      ★                         │                       │
│   │       • • •     ★ Random samples      │                       │
│   │           • •                         │                       │
│   │              • •    ★                 │                       │
│   │                 ★                     │                       │
│   └───────────────────────────────────────┘                       │
│                                                                   │
│   ★ points are NOT near any training data!                        │
│   Decoder produces garbage for these inputs.                      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘


The Solution: VAE
─────────────────
Force the latent space to have a known structure (Gaussian)!
Then we CAN sample from it meaningfully.
```

---

## 7.3 Variational Autoencoders (VAE)

### The Key Insight

Make the latent space **structured** by enforcing a prior distribution:

```
VAE KEY IDEA
════════════

Autoencoder: Encode to point z
VAE:         Encode to distribution q(z|x), sample z ~ q(z|x)

And enforce: q(z|x) should be close to prior p(z) = N(0, I)


VISUALIZATION:
──────────────

Autoencoder latent space:            VAE latent space:
(unstructured)                       (structured as Gaussian)

    ┌─────────────────────┐          ┌─────────────────────┐
    │                     │          │                     │
    │  •    •             │          │      ░░░░░░         │
    │     •  • •          │          │    ░░░░░░░░░        │
    │        •   •        │          │   ░░░██████░░░      │
    │           •  •      │          │   ░░████████░░      │
    │              •      │          │    ░░██████░░       │
    │                     │          │      ░░░░░░         │
    └─────────────────────┘          └─────────────────────┘

• = encoded points                   █ = high probability region
                                     ░ = lower probability

In VAE, ANY point in the high-probability region
can be decoded to a meaningful output!
```

### VAE Architecture

```
VAE ARCHITECTURE
════════════════

    Input x
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                 │
│                                                                 │
│   x → [Neural Network] → μ (mean vector)                        │
│                       → log σ² (log variance vector)            │
│                                                                 │
│   Output: Parameters of q(z|x) = N(μ, σ²I)                      │
│                                                                 │
└────────────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │     REPARAMETERIZATION         │
                    │                                │
                    │   ε ~ N(0, I)  (random noise)  │
                    │   z = μ + σ × ε                │
                    │                                │
                    │   This makes sampling          │
                    │   differentiable!              │
                    │                                │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER                                 │
│                                                                 │
│   z → [Neural Network] → x̂ (reconstruction)                     │
│                                                                 │
│   Models p(x|z)                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                                Output x̂


During generation:
──────────────────
1. Sample z ~ N(0, I)
2. Decode: x = Decoder(z)
3. Output x is a new sample!
```

### The ELBO Loss Function

VAE optimizes the Evidence Lower BOund (ELBO):

```
VAE LOSS: THE ELBO
══════════════════

We want to maximize: log p(x)  (likelihood of data)

But this is intractable! Instead, maximize a lower bound:

log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
           ─────────────    ─────────────────────
           Reconstruction   KL Divergence
           Term             (Regularization)


THE TWO TERMS:
──────────────

1. RECONSTRUCTION TERM: E_q[log p(x|z)]
   ─────────────────────────────────────
   "How well can we reconstruct x from z?"

   Approximated as: -||x - x̂||²  (MSE)
   or: -BCE(x, x̂)  (Binary Cross Entropy)

   Encourages: Accurate reconstruction


2. KL DIVERGENCE TERM: KL(q(z|x) || p(z))
   ────────────────────────────────────────
   "How close is the encoder distribution to the prior?"

   For Gaussian q and p:
   KL = 0.5 × Σ(μ² + σ² - log(σ²) - 1)

   Encourages: Latent codes close to N(0, I)


TOTAL LOSS:
───────────

L = -E_q[log p(x|z)] + KL(q(z|x) || p(z))
  = Reconstruction Loss + KL Loss

In code:
  loss = recon_loss + beta * kl_loss

(beta can be tuned; beta=1 is standard VAE)


INTUITION:
──────────

Reconstruction pulls latent codes to be unique (spread out)
KL pulls latent codes toward origin (compress together)

The balance creates a structured, continuous latent space!

Without KL:   ┌─────────┐        With KL:   ┌─────────┐
              │ •     • │                    │  ••••   │
              │    •    │                    │ •••••   │
              │  •   •  │                    │  ••••   │
              └─────────┘                    └─────────┘
              Spread out                     Compact, near origin
```

### The Reparameterization Trick

The key innovation that makes VAE training possible:

```
THE REPARAMETERIZATION TRICK
════════════════════════════

Problem: We need gradients through sampling!

    μ, σ² = Encoder(x)
    z ~ N(μ, σ²)         ← Random sampling! No gradient!
    x̂ = Decoder(z)
    loss = ||x - x̂||²

Can't backprop through "sample from distribution"!


Solution: Reparameterize the sampling
────────────────────────────────────

Instead of:  z ~ N(μ, σ²)

Write as:    ε ~ N(0, 1)        ← Sample standard normal
             z = μ + σ × ε      ← Deterministic transformation!

Now z is a deterministic function of μ and σ!


GRADIENT FLOW:
──────────────

Before (no gradient):

    μ ─────────┐
               │─── Sample ──→ z ──→ Decoder ──→ loss
    σ ─────────┘     ✗ BLOCKED


After (gradient flows):

                ε ~ N(0, 1) (no gradient needed)
                     │
    μ ────────────►[+]
                     │
    σ ──────[×]─────►│──────────→ z ──→ Decoder ──→ loss
               ▲     │
               │     │
        ε ─────┘     ▼
                  z = μ + σ × ε

    ✓ Gradients flow through μ and σ!


IMPLEMENTATION:
───────────────

def reparameterize(mu, log_var):
    # log_var instead of σ² for numerical stability
    std = torch.exp(0.5 * log_var)  # σ = exp(0.5 * log(σ²))
    eps = torch.randn_like(std)      # ε ~ N(0, 1)
    return mu + std * eps            # z = μ + σ × ε
```

### VAE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Encoder outputs μ and log(σ²)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction + beta * KL divergence
    """
    # Reconstruction loss (binary cross entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence: KL(N(μ, σ²) || N(0, 1))
    # = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss


# Training
def train_vae(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].view(batch[0].size(0), -1)

            # Forward
            x_recon, mu, logvar = model(x)

            # Loss
            loss = vae_loss(x, x_recon, mu, logvar)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader.dataset):.4f}")


# Generation
def generate_samples(model, num_samples=10):
    with torch.no_grad():
        # Sample from prior
        z = torch.randn(num_samples, model.fc_mu.out_features)
        # Decode
        samples = model.decode(z)
        return samples
```

### Latent Space Interpolation

One beautiful property of VAEs is smooth interpolation in latent space:

```
VAE LATENT INTERPOLATION
════════════════════════

Encode two images to latent space, interpolate, decode:

Image A                             Image B
  "3"                                 "7"
   │                                   │
   ▼                                   ▼
Encode                             Encode
   │                                   │
   ▼                                   ▼
  z_A ─────────────────────────────► z_B

Interpolate: z_t = (1-t) × z_A + t × z_B

  t=0.0   t=0.2   t=0.4   t=0.6   t=0.8   t=1.0
  ─────   ─────   ─────   ─────   ─────   ─────
   "3"    "3?"    "37"    "7?"    "7"     "7"

The transition is SMOOTH because:
  • Latent space is continuous (no "holes")
  • Nearby points in latent space → similar images
  • KL regularization ensures good coverage
```

### VAE Variations

```
VAE VARIANTS
════════════

1. β-VAE (Higgins et al., 2017)
   ────────────────────────────
   Increase β in: Loss = Recon + β × KL

   Higher β → More disentangled latent factors
   Trade-off: Reconstruction quality decreases

   Example: Latent dims might capture:
     z₁: Face orientation (left/right)
     z₂: Hair color
     z₃: Age
     Each dimension = one meaningful factor!


2. CVAE (Conditional VAE)
   ──────────────────────
   Condition on class label y:

   Encoder: q(z|x, y)
   Decoder: p(x|z, y)

   Can generate: "Create a '7' that looks like this style"


3. VQ-VAE (Vector Quantized VAE)
   ─────────────────────────────
   Discrete latent codes instead of continuous.

   z_continuous → [Quantize to codebook] → z_discrete

   Used in: DALL-E 1, audio generation


4. Hierarchical VAE
   ─────────────────
   Multiple levels of latent variables:

   z₁ → z₂ → z₃ → x

   Better modeling of complex data.
```

---

## 7.4 Generative Adversarial Networks (GAN)

### The Adversarial Idea

Two networks compete in a game:

```
GAN: THE ADVERSARIAL GAME
═════════════════════════

GENERATOR (G): Creates fake samples from noise
  "I'll make fakes so good you can't tell!"

DISCRIMINATOR (D): Distinguishes real from fake
  "I'll catch every fake you make!"

They compete, and both get better!


THE ARCHITECTURE:
─────────────────

Random noise z ~ N(0, I)
      │
      ▼
┌─────────────────────────┐
│      GENERATOR G        │
│                         │
│  z → Neural Net → x_fake│
│                         │
└───────────┬─────────────┘
            │
            ▼
        x_fake (fake sample)
            │
            │         Real data x_real
            │              │
            ▼              ▼
    ┌───────────────────────────────────┐
    │        DISCRIMINATOR D            │
    │                                   │
    │   x → Neural Net → P(x is real)   │
    │                                   │
    └───────────────────────┬───────────┘
                            │
                            ▼
                   Score in [0, 1]
                   1 = "definitely real"
                   0 = "definitely fake"


THE GAME:
─────────

D wants: D(x_real) → 1, D(x_fake) → 0
         (Correctly classify real and fake)

G wants: D(G(z)) → 1
         (Fool D into thinking fakes are real)

At equilibrium: D can't tell real from fake
                G creates perfect samples!
```

### The GAN Objective

```
GAN MINIMAX OBJECTIVE
═════════════════════

min_G max_D  V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
──────────            ────────────   ──────────────────
G minimizes           D on real      D on fake
D maximizes


INTUITION FOR D:
────────────────
D wants to maximize:
  • log D(x_real)     → Want D(x_real) = 1 → log(1) = 0 (max)
  • log(1 - D(x_fake))→ Want D(x_fake) = 0 → log(1-0) = 0 (max)

So D is trained like a classifier!


INTUITION FOR G:
────────────────
G wants to minimize log(1 - D(G(z)))
  → Want D(G(z)) = 1 (fool D)
  → log(1 - 1) = -∞ (minimize)

In practice, use: maximize log D(G(z))
(Same optimum, better gradients when G is bad)


TRAINING ALTERNATION:
─────────────────────

1. Train D:
   └── See real images → predict 1
   └── See G(z) fakes  → predict 0

2. Train G:
   └── Generate fakes
   └── Update G to make D(G(z)) higher

3. Repeat
```

### GAN Training Algorithm

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Output probability
        )

    def forward(self, x):
        return self.net(x)


def train_gan(generator, discriminator, dataloader, epochs=100, latent_dim=100):
    criterion = nn.BCELoss()

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)

            # Labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # ================== Train Discriminator ==================
            d_optimizer.zero_grad()

            # Loss on real images
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            # Loss on fake images
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())  # Don't backprop to G
            d_loss_fake = criterion(outputs, fake_labels)

            # Total D loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ================== Train Generator ==================
            g_optimizer.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            outputs = discriminator(fake_images)

            # G wants D to think fakes are real
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch {epoch+1}: D Loss = {d_loss.item():.4f}, G Loss = {g_loss.item():.4f}")
```

### GAN Challenges

```
GAN TRAINING CHALLENGES
═══════════════════════

1. MODE COLLAPSE
   ─────────────
   G only learns to produce a few types of outputs.

   Training data:                    G outputs:
   ┌─────────────────────┐          ┌─────────────────────┐
   │ 0 1 2 3 4 5 6 7 8 9 │          │ 7 7 7 7 7 7 7 7 7 7 │
   └─────────────────────┘          └─────────────────────┘

   G found ONE mode that fools D, and sticks with it!


2. TRAINING INSTABILITY
   ────────────────────
   D and G can oscillate instead of converging.

   Loss over time:
   │     /\    /\    /\
   │    /  \  /  \  /  \
   │   /    \/    \/    \
   │  /
   └──────────────────────→
           Time

   Solution: Careful learning rates, architectural tricks


3. VANISHING GRADIENTS
   ───────────────────
   If D is too good, G gets no learning signal.

   D(G(z)) ≈ 0 for all G outputs
   → log(1 - 0) = 0
   → No gradient for G!

   Solution: Use -log D(G(z)) instead of log(1 - D(G(z)))


4. EVALUATION DIFFICULTY
   ─────────────────────
   No likelihood! How do we know if G is good?

   Solutions:
   • Inception Score (IS)
   • Fréchet Inception Distance (FID)
   • Human evaluation
```

### DCGAN (Deep Convolutional GAN)

Architecture guidelines that made GANs work well for images:

```
DCGAN ARCHITECTURE GUIDELINES
═════════════════════════════

1. Replace pooling with strided convolutions
   • D: Conv with stride 2 (downsample)
   • G: ConvTranspose with stride 2 (upsample)

2. Use BatchNorm in both G and D
   • Except: D's input layer, G's output layer

3. Use ReLU in G (except output: Tanh)
   Use LeakyReLU in D (slope 0.2)

4. Remove fully connected layers
   • Global average pooling for D


GENERATOR ARCHITECTURE:
───────────────────────

z (100 dim)
    │
    ▼
Project & Reshape to 4×4×1024
    │
    ▼
ConvTranspose 512, 4×4, stride 2  →  8×8×512
    │ BatchNorm + ReLU
    ▼
ConvTranspose 256, 4×4, stride 2  →  16×16×256
    │ BatchNorm + ReLU
    ▼
ConvTranspose 128, 4×4, stride 2  →  32×32×128
    │ BatchNorm + ReLU
    ▼
ConvTranspose 3, 4×4, stride 2    →  64×64×3
    │ Tanh
    ▼
Output image
```

### GAN Variants

```
NOTABLE GAN VARIANTS
════════════════════

1. WGAN (Wasserstein GAN)
   ──────────────────────
   Use Wasserstein distance instead of JS divergence.
   More stable training, meaningful loss curves.

   Key change: D outputs score (not probability)
              Clip D weights or use gradient penalty


2. StyleGAN (Karras et al.)
   ────────────────────────
   State-of-the-art face generation.

   Innovations:
   • Mapping network: z → w (intermediate latent)
   • Adaptive Instance Norm (AdaIN)
   • Progressive growing (low → high resolution)
   • Style mixing for diversity


3. CycleGAN
   ─────────
   Unpaired image-to-image translation.

   Example: Horse ↔ Zebra without paired training data!

   Uses cycle consistency:
   A → G_AB → B → G_BA → A' ≈ A


4. Conditional GAN (cGAN)
   ──────────────────────
   Condition on class label or other info.

   G(z, y) → x   (y = class label)
   D(x, y) → real/fake

   "Generate a cat" vs "generate something"


5. Pix2Pix
   ────────
   Paired image-to-image translation.

   Example: Sketch → Photo (with training pairs)
```

---

## 7.5 Diffusion Models

### The Core Idea

Gradually add noise to data, then learn to reverse the process:

```
DIFFUSION MODEL INTUITION
═════════════════════════

FORWARD PROCESS (fixed, no learning):
─────────────────────────────────────

Start with real image x₀, gradually add noise:

x₀ ──► x₁ ──► x₂ ──► ... ──► x_T
clean   │      │             pure
image   │      │             noise
        │      │
       +ε₁    +ε₂           +ε_T
      (noise) (noise)       (noise)

After T steps: x_T ≈ N(0, I)  (just random noise!)


REVERSE PROCESS (learned):
──────────────────────────

Start with noise x_T, gradually denoise:

x_T ──► x_{T-1} ──► ... ──► x₁ ──► x₀
pure    │                         clean
noise   │                         image!
        │
    Neural network predicts how to denoise


KEY INSIGHT:
────────────
Forward: Easy (just add noise!)
Reverse: Hard (need to learn)

But! Each reverse step is "small" - we only remove
a little noise at a time. This is much easier than
generating from scratch!
```

### The Forward Process (Adding Noise)

```
FORWARD PROCESS MATHEMATICS
═══════════════════════════

At each step t, add Gaussian noise:

q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

Where:
  • β_t is the noise schedule (typically 0.0001 to 0.02)
  • √(1-β_t) scales down the signal
  • β_t determines how much noise to add


NOISE SCHEDULE:
───────────────

β_t over time:
│
│                           ╱────
│                      ____╱
│                 ____╱
│            ____╱
│       ____╱
│  ────╱
└───────────────────────────────→
  0                           T
       More noise added over time


CLOSED-FORM SAMPLING:
─────────────────────

We can sample x_t directly from x₀ (skip intermediate steps!):

Define: α_t = 1 - β_t
        ᾱ_t = α₁ × α₂ × ... × α_t (cumulative product)

Then: q(x_t | x₀) = N(x_t; √ᾱ_t x₀, (1-ᾱ_t) I)

Or: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε,  where ε ~ N(0, I)

This is efficient! We can jump to any timestep directly.


VISUALIZATION:
──────────────

t=0        t=250      t=500      t=750      t=1000
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│     │    │░░░░░│    │▒▒▒▒▒│    │▓▓▓▓▓│    │█████│
│ 🐱  │ →  │░🐱░░│ →  │▒▒▒▒▒│ →  │▓▓▓▓▓│ →  │█████│
│     │    │░░░░░│    │▒▒▒▒▒│    │▓▓▓▓▓│    │█████│
└─────┘    └─────┘    └─────┘    └─────┘    └─────┘
Clean      Light      Medium     Heavy      Pure
           noise      noise      noise      noise
```

### The Reverse Process (Denoising)

```
REVERSE PROCESS (LEARNED)
═════════════════════════

Learn to reverse each noising step:

p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

A neural network predicts:
  • μ_θ: The mean of the denoised distribution
  • Σ_θ: The variance (often fixed or learned)


THE NEURAL NETWORK:
───────────────────

Input:  x_t (noisy image at timestep t)
        t (timestep, as embedding)

Output: Prediction for denoising

                x_t
                 │
                 ▼
         ┌─────────────┐
         │    U-Net    │ ← t (timestep embedding)
         │   or other  │
         │   network   │
         └──────┬──────┘
                │
                ▼
           ε_θ(x_t, t)
        (predicted noise)


KEY INSIGHT:
────────────
Instead of predicting μ directly, we predict the NOISE!

From: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε

If we can predict ε, we can recover x₀:
  x̂₀ = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t

Then compute μ from x̂₀.
```

### Training Objective

The training is surprisingly simple: predict the noise!

```
DIFFUSION TRAINING OBJECTIVE
════════════════════════════

Loss = E_{t, x₀, ε} [ ||ε - ε_θ(x_t, t)||² ]

In words:
  1. Sample a random training image x₀
  2. Sample random timestep t ~ Uniform(1, T)
  3. Sample random noise ε ~ N(0, I)
  4. Create noisy image: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
  5. Predict noise: ε̂ = ε_θ(x_t, t)
  6. Loss = ||ε - ε̂||² (MSE between true and predicted noise)


TRAINING ALGORITHM:
───────────────────

repeat:
    x₀ = sample from training data
    t = random integer from [1, T]
    ε = sample from N(0, I)

    x_t = √ᾱ_t × x₀ + √(1-ᾱ_t) × ε

    loss = ||ε - ε_θ(x_t, t)||²

    update θ via gradient descent


WHY IS THIS SIMPLE?
───────────────────
  • Just MSE loss! No adversarial training.
  • Each timestep is independent (can sample any t)
  • Stable training (no mode collapse, no D/G balance)
```

### Sampling (Generation)

```
DIFFUSION SAMPLING ALGORITHM
════════════════════════════

1. Start with pure noise: x_T ~ N(0, I)

2. For t = T, T-1, ..., 1:
   a. Predict noise: ε̂ = ε_θ(x_t, t)
   b. Compute x_{t-1} using:

      x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε̂) + σ_t z

   Where z ~ N(0, I) and σ_t is the noise scale.

3. Return x₀ (final denoised image)


VISUALIZATION:
──────────────

t=1000     t=750      t=500      t=250      t=0
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│█████│    │▓▓▓▓▓│    │▒▒▒▒▒│    │░░░░░│    │     │
│█████│ →  │▓▓▓▓▓│ →  │▒?▒▒▒│ →  │░🐱░░│ →  │ 🐱  │
│█████│    │▓▓▓▓▓│    │▒▒▒▒▒│    │░░░░░│    │     │
└─────┘    └─────┘    └─────┘    └─────┘    └─────┘
Pure       Starting   Structure  Details   Clean
noise      to form    emerges    appear    image!


INTUITION:
──────────
Early steps (high t): Determine global structure
  "Is this a cat or a dog? Where is the head?"

Late steps (low t): Refine fine details
  "What color are the eyes? Add fur texture"
```

### Diffusion Model Implementation

```python
import torch
import torch.nn as nn
import math

class DiffusionModel:
    def __init__(self, model, T=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model  # The noise prediction network
        self.T = T

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: add noise to x_0 to get x_t
        x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        return (
            torch.sqrt(alpha_bar_t) * x_0 +
            torch.sqrt(1 - alpha_bar_t) * noise
        )

    def training_step(self, x_0):
        """One training step"""
        batch_size = x_0.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.T, (batch_size,))

        # Sample noise
        noise = torch.randn_like(x_0)

        # Create noisy images
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # MSE loss
        loss = nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse process: denoise x_t to get x_{t-1}
        """
        # Get schedule values
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Compute mean
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            return mean + sigma_t * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, shape):
        """Generate samples from noise"""
        # Start with pure noise
        x = torch.randn(shape)

        # Iteratively denoise
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t)

        return x
```

### Why Diffusion Models Work So Well

```
DIFFUSION ADVANTAGES
════════════════════

1. TRAINING STABILITY
   ──────────────────
   • Simple MSE loss (no adversarial dynamics)
   • No mode collapse
   • Loss correlates with sample quality

   GAN Loss:          Diffusion Loss:
   │  /\  /\  /\     │ ╲
   │ /  \/  \/  \    │  ╲
   │/              \ │   ╲___________
   └────────────────  └────────────────
        Unstable           Stable


2. MODE COVERAGE
   ─────────────
   GANs might miss some modes (mode collapse).
   Diffusion covers all modes (trained on all data).


3. LIKELIHOOD BOUND
   ────────────────
   Can compute a lower bound on log p(x).
   (Though typically not used for evaluation)


4. FLEXIBLE ARCHITECTURE
   ────────────────────
   Can use any network (U-Net is common).
   Easy to condition on text, class, etc.


5. CONTROLLABILITY
   ───────────────
   • Classifier guidance
   • Classifier-free guidance
   • Inpainting, super-resolution, etc.
```

---

## 7.6 Latent Diffusion (Stable Diffusion)

### The Efficiency Problem

Running diffusion in pixel space is expensive:

```
PIXEL-SPACE DIFFUSION COST
══════════════════════════

512×512×3 image = 786,432 dimensions

Each denoising step:
  • U-Net forward pass on 786K-dim input
  • ~1000 steps to generate one image
  • Very slow and memory-intensive!

Example timing:
  • Pixel-space diffusion: ~60 seconds/image
  • Latent diffusion: ~5 seconds/image


SOLUTION: LATENT DIFFUSION
══════════════════════════

Compress image to latent space FIRST, then do diffusion there!

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  512×512×3 image                                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                    │
│  │ VAE Encoder │  (pre-trained, frozen)                             │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│    64×64×4 latent  ← 8× smaller in each spatial dimension!          │
│         │            64×64×4 = 16,384 dims (48× reduction!)         │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────┐                                        │
│  │  DIFFUSION IN LATENT    │  (this is what we train)               │
│  │  SPACE (U-Net)          │                                        │
│  └───────────┬─────────────┘                                        │
│              │                                                      │
│              ▼                                                      │
│         Denoised latent                                             │
│              │                                                      │
│              ▼                                                      │
│  ┌─────────────┐                                                    │
│  │ VAE Decoder │  (pre-trained, frozen)                             │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│    512×512×3 image                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Text-to-Image with Cross-Attention

Adding text conditioning to latent diffusion:

```
TEXT-TO-IMAGE ARCHITECTURE
══════════════════════════

         Text prompt: "A cat wearing a hat"
                           │
                           ▼
               ┌───────────────────────┐
               │   TEXT ENCODER        │
               │   (CLIP or T5)        │
               └──────────┬────────────┘
                          │
                          ▼
               Text embeddings [77 × 768]
                          │
                          │ Cross-attention
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       │
    ┌─────────────────┐              │
    │                 │              │
    │     U-NET       │◄─────────────┘
    │                 │
    │  • Self-attention (within image)
    │  • Cross-attention (text → image)
    │  • Convolutions
    │                 │
    └────────┬────────┘
             │
             ▼
      Denoised latent


CROSS-ATTENTION MECHANISM:
──────────────────────────

Q = Linear(image_features)   # Query from image
K = Linear(text_embeddings)  # Key from text
V = Linear(text_embeddings)  # Value from text

Attention = softmax(Q × K^T / √d) × V

This lets the U-Net "look at" the text
while denoising!
```

### Classifier-Free Guidance

The key to high-quality text-to-image generation:

```
CLASSIFIER-FREE GUIDANCE
════════════════════════

Problem: Conditioning on text gives okay results,
         but we want STRONG adherence to the prompt.

Solution: Amplify the effect of the condition!


THE IDEA:
─────────

Train model to work both with and without text:
  • With text:    ε_θ(x_t, t, text)
  • Without text: ε_θ(x_t, t, ∅)    (∅ = empty text)

At inference, extrapolate AWAY from unconditional:

  ε_guided = ε_unconditional + w × (ε_conditional - ε_unconditional)

Where w is the guidance scale (typically 7-15).


VISUALIZATION:
──────────────

                    Noise predictions in latent space

                         ε_conditional
                              ●
                             /
                            /
              ε_guided     /
                 ★        /
                         /  ← Guidance amplifies
                        /     the effect of text
                       /
               ●──────●
   ε_unconditional    Direction: ε_cond - ε_uncond

  w = 1.0: ε_guided = ε_conditional (no amplification)
  w = 7.5: ε_guided goes 7.5× in the text direction
  w > 1:   Stronger adherence to text, but may oversaturate


EFFECT OF GUIDANCE SCALE:
─────────────────────────

w = 1.0     w = 5.0     w = 10.0    w = 20.0
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│     │     │     │     │     │     │     │
│ 🐱? │     │ 🐱  │     │ 🐱! │     │ 🐱!! │
│     │     │     │     │     │     │     │
└─────┘     └─────┘     └─────┘     └─────┘
Vague       Good        Strong      Over-
                                    saturated
```

### Stable Diffusion Architecture

```
STABLE DIFFUSION COMPONENTS
═══════════════════════════

1. VAE (Variational Autoencoder)
   ─────────────────────────────
   • Encoder: 512×512×3 → 64×64×4
   • Decoder: 64×64×4 → 512×512×3
   • Pre-trained, frozen during diffusion training


2. U-Net (Noise Predictor)
   ───────────────────────
   • Input: 64×64×4 noisy latent + timestep + text
   • Output: 64×64×4 predicted noise

   Architecture:
   ┌────────────────────────────────────────────────────┐
   │                                                    │
   │    Encoder         Middle          Decoder        │
   │   (downsample)    (bottleneck)    (upsample)      │
   │                                                    │
   │   64×64×320   →   8×8×1280   →   64×64×320        │
   │     │  ↓            │              ↑  │           │
   │   32×32×640        ...         32×32×640          │
   │     │  ↓                           ↑  │           │
   │   16×16×1280                   16×16×1280         │
   │     │  ↓                           ↑  │           │
   │   8×8×1280                     8×8×1280           │
   │                                                    │
   │   Skip connections: ───────────────────►          │
   │                                                    │
   │   Each block has:                                 │
   │   • ResNet blocks                                 │
   │   • Self-attention                                │
   │   • Cross-attention (to text)                     │
   │                                                    │
   └────────────────────────────────────────────────────┘


3. Text Encoder (CLIP or OpenCLIP)
   ────────────────────────────────
   • Input: Text prompt (max 77 tokens)
   • Output: 77 × 768 text embeddings
   • Pre-trained, frozen


4. Scheduler
   ─────────
   • Controls noise schedule
   • Options: DDPM, DDIM, Euler, DPM++, etc.
   • DDIM allows fewer steps (50 instead of 1000)
```

---

## 7.7 Comparison of Generative Models

```
GENERATIVE MODEL COMPARISON
═══════════════════════════

                VAE         GAN         Diffusion
                ───         ───         ─────────
Sample Quality  Medium      High        Very High
Training        Stable      Unstable    Stable
Mode Coverage   Full        May miss    Full
Training Speed  Fast        Medium      Slow
Sampling Speed  Fast        Fast        Slow
Likelihood      Yes (bound) No          Yes (bound)
Controllability Medium      Medium      High


WHEN TO USE WHICH:
──────────────────

VAE:
  ✓ Fast inference needed
  ✓ Latent space manipulation important
  ✓ Lower quality acceptable
  Example: Compression, representation learning

GAN:
  ✓ Real-time generation needed
  ✓ Specific domain (faces, etc.)
  ✓ Can handle training instability
  Example: Face generation (StyleGAN)

Diffusion:
  ✓ Highest quality required
  ✓ Text/condition control important
  ✓ Can afford slower generation
  Example: Text-to-image (DALL-E, Stable Diffusion)


SAMPLE QUALITY RANKING:
───────────────────────

Low Quality                              High Quality
    │                                          │
    ▼                                          ▼
   VAE ──────────── GAN ──────────── Diffusion
         ↑                    ↑           ↑
     Simple AE            DCGAN      State-of-art
                          StyleGAN   Stable Diffusion
```

---

## 7.8 Summary

### Key Concepts

```
GENERATIVE MODELS SUMMARY
═════════════════════════

1. AUTOENCODERS
   ─────────────
   • Encoder-bottleneck-decoder architecture
   • Learn compressed representations
   • Can't sample (unstructured latent space)

2. VARIATIONAL AUTOENCODERS (VAE)
   ─────────────────────────────
   • Structured latent space (Gaussian)
   • ELBO loss = Reconstruction + KL divergence
   • Reparameterization trick for differentiability
   • Can sample: z ~ N(0,I), decode to image

3. GENERATIVE ADVERSARIAL NETWORKS (GAN)
   ──────────────────────────────────────
   • Generator vs Discriminator game
   • High quality samples
   • Training challenges: mode collapse, instability
   • Variants: DCGAN, StyleGAN, CycleGAN

4. DIFFUSION MODELS
   ─────────────────
   • Forward: gradually add noise
   • Reverse: learn to denoise
   • Simple training (MSE loss on noise)
   • State-of-the-art quality

5. LATENT DIFFUSION
   ─────────────────
   • Diffusion in compressed latent space
   • Much faster than pixel-space
   • Text conditioning via cross-attention
   • Classifier-free guidance for quality
```

### Glossary Terms Covered

```
GLOSSARY
════════

Architecture:
─────────────
• Autoencoder - Encoder-decoder for reconstruction
• VAE - Variational autoencoder with probabilistic latent space
• GAN - Generator-discriminator adversarial setup
• Diffusion Model - Iterative denoising generative model
• U-Net - Encoder-decoder with skip connections

Training:
─────────
• ELBO - Evidence Lower Bound (VAE objective)
• KL Divergence - Distance between distributions
• Reparameterization Trick - Makes sampling differentiable
• Adversarial Training - Two networks competing
• Mode Collapse - GAN generates limited variety

Concepts:
─────────
• Latent Space - Compressed representation space
• Latent Code - Compressed representation (z)
• Prior - Assumed distribution p(z)
• Posterior - Learned distribution q(z|x)
• Likelihood - p(x|z)

Diffusion:
──────────
• Forward Process - Adding noise (fixed)
• Reverse Process - Removing noise (learned)
• Noise Schedule - How noise is added over time
• Denoising - Removing noise from data
• Classifier-Free Guidance - Amplifying conditions
• Latent Diffusion - Diffusion in latent space
```

### What's Next

Module 8 covers **Advanced Topics**: Retrieval-Augmented Generation (RAG), AI agents
and tool use, model deployment and serving, evaluation and benchmarking, and ethical
considerations.

---

## References

### Key Papers

**Autoencoders:**
- Hinton & Salakhutdinov, ["Reducing the Dimensionality of Data with Neural Networks"](https://www.science.org/doi/10.1126/science.1127647) (2006)

**VAE:**
- Kingma & Welling, ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) (2013)
- Higgins et al., ["β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](https://openreview.net/forum?id=Sy2fzU9gl) (2017)
- van den Oord et al., ["Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) (VQ-VAE, 2017)

**GAN:**
- Goodfellow et al., ["Generative Adversarial Networks"](https://arxiv.org/abs/1406.2661) (2014)
- Radford et al., ["Unsupervised Representation Learning with Deep Convolutional GANs"](https://arxiv.org/abs/1511.06434) (DCGAN, 2015)
- Karras et al., ["A Style-Based Generator Architecture for GANs"](https://arxiv.org/abs/1812.04948) (StyleGAN, 2018)
- Arjovsky et al., ["Wasserstein GAN"](https://arxiv.org/abs/1701.07875) (WGAN, 2017)

**Diffusion:**
- Ho et al., ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) (DDPM, 2020)
- Song et al., ["Score-Based Generative Modeling through Stochastic Differential Equations"](https://arxiv.org/abs/2011.13456) (2020)
- Nichol & Dhariwal, ["Improved Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2102.09672) (2021)
- Song et al., ["Denoising Diffusion Implicit Models"](https://arxiv.org/abs/2010.02502) (DDIM, 2020)

**Text-to-Image:**
- Rombach et al., ["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752) (Stable Diffusion, 2022)
- Ramesh et al., ["Hierarchical Text-Conditional Image Generation with CLIP Latents"](https://arxiv.org/abs/2204.06125) (DALL-E 2, 2022)
- Saharia et al., ["Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding"](https://arxiv.org/abs/2205.11487) (Imagen, 2022)
- Ho & Salimans, ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2207.12598) (2022)

### Courses and Tutorials
- [Stanford CS231n: Generative Models Lecture](https://cs231n.stanford.edu/)
- [Lilian Weng's Blog: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/)

### Tools and Libraries
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Diffusion model library
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Popular interface
- [PyTorch Lightning](https://lightning.ai/) - Training framework
- [CompVis](https://github.com/CompVis/stable-diffusion) - Original Stable Diffusion

