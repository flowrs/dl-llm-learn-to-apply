# Module 7: Generative Models

## Learning Objectives

By the end of this module, you will understand:
- The goal of generative modeling
- Autoencoders and Variational Autoencoders (VAE)
- Generative Adversarial Networks (GAN)
- Diffusion models
- Text-to-image generation

---

## 7.1 What is Generative Modeling?

### The Goal

Learn to generate new samples from the same distribution as training data:

```
Training data: {x₁, x₂, ..., xₙ} ~ p_data(x)
Goal: Learn p_model(x) ≈ p_data(x)
Then: Sample new x ~ p_model(x)
```

### Two Approaches

**Explicit density**: Define and optimize p(x) directly
- Autoregressive models (PixelCNN, GPT)
- Variational Autoencoders

**Implicit density**: Learn to sample without explicit p(x)
- GANs
- Diffusion models

### Applications

- Image synthesis
- Text generation
- Drug discovery
- Data augmentation
- Art and creativity

---

## 7.2 Autoencoders

### Architecture

```
Input x → [Encoder] → z (latent) → [Decoder] → x̂ (reconstruction)

Goal: x̂ ≈ x (minimize reconstruction error)
```

**Encoder**: Compresses input to lower-dimensional representation
**Latent code z**: Compressed representation (bottleneck)
**Decoder**: Reconstructs input from latent code

### Code

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Training
loss = F.mse_loss(model(x), x)  # Reconstruction loss
```

### Limitations for Generation

- Latent space is unstructured
- Random z doesn't produce meaningful outputs
- No way to sample new data

---

## 7.3 Variational Autoencoders (VAE)

### Key Idea

Make latent space **structured** by enforcing a prior distribution:

```
z ~ N(0, I)  (Standard Gaussian prior)
```

### Architecture

```
x → [Encoder] → μ, σ → [Sample z] → [Decoder] → x̂
                         ↑
                z = μ + σ × ε, where ε ~ N(0, I)
```

Encoder outputs **distribution parameters** (μ, σ), not a single z.

### The ELBO (Loss Function)

```
L = Reconstruction Loss + KL Divergence
  = E[log p(x|z)] - KL(q(z|x) || p(z))
  = -||x - x̂||² - KL(N(μ, σ²) || N(0, 1))
```

**Reconstruction**: Make output match input
**KL term**: Keep latent distribution close to N(0, I)

### Reparameterization Trick

Can't backprop through sampling! Solution:

```python
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)  # Random noise
    return mu + eps * std        # Deterministic function of inputs
```

### Generation

Sample z ~ N(0, I), decode to get new image.

### VAE Code

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return self.decode(z), mu, log_var

def vae_loss(x, x_recon, mu, log_var):
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl
```

---

## 7.4 Generative Adversarial Networks (GAN)

### The Idea

Two networks playing a game:

```
Generator G: Creates fake samples from noise
Discriminator D: Distinguishes real from fake

         z (noise)
            ↓
        [Generator G]
            ↓
       Fake sample
            ↓
      ┌─────────────┐
      │Discriminator│ ← Also receives real samples
      │      D      │
      └─────────────┘
            ↓
      Real or Fake?
```

### Training Objective

Minimax game:
```
min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
```

**D wants**: D(real) → 1, D(fake) → 0
**G wants**: D(fake) → 1 (fool D)

### Training Algorithm

```python
for epoch in range(epochs):
    # Train Discriminator
    real_labels = torch.ones(batch_size)
    fake_labels = torch.zeros(batch_size)

    # Real samples
    d_real = discriminator(real_images)
    loss_real = F.binary_cross_entropy(d_real, real_labels)

    # Fake samples
    z = torch.randn(batch_size, latent_dim)
    fake_images = generator(z)
    d_fake = discriminator(fake_images.detach())
    loss_fake = F.binary_cross_entropy(d_fake, fake_labels)

    d_loss = loss_real + loss_fake
    d_loss.backward()
    d_optimizer.step()

    # Train Generator
    z = torch.randn(batch_size, latent_dim)
    fake_images = generator(z)
    d_fake = discriminator(fake_images)
    g_loss = F.binary_cross_entropy(d_fake, real_labels)  # Fool D

    g_loss.backward()
    g_optimizer.step()
```

### GAN Challenges

1. **Mode collapse**: G generates limited variety
2. **Training instability**: D/G imbalance
3. **No explicit likelihood**: Hard to evaluate quality

### GAN Variants

- **DCGAN**: Convolutional architecture
- **StyleGAN**: State-of-the-art face generation
- **CycleGAN**: Unpaired image translation
- **Conditional GAN**: Control what to generate

---

## 7.5 Diffusion Models

### The Idea

1. **Forward process**: Gradually add noise until data becomes pure noise
2. **Reverse process**: Learn to gradually denoise

```
x₀ (data) → x₁ → x₂ → ... → x_T (noise)
     ↑_______________________________|
        Learn to reverse this process
```

### Forward Process (Fixed)

Add Gaussian noise at each step:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

After T steps, x_T ≈ N(0, I).

### Reverse Process (Learned)

Neural network predicts noise to remove:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Training

Simple objective: Predict the noise!
```
L = E[||ε - ε_θ(x_t, t)||²]
```

Where:
- ε is the actual noise added
- ε_θ is the network's noise prediction
- x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε

### Sampling

Start from noise, iteratively denoise:
```python
x = torch.randn(batch_size, channels, height, width)  # Pure noise

for t in reversed(range(T)):
    predicted_noise = model(x, t)
    x = denoise_step(x, predicted_noise, t)

# x is now a generated image
```

### Why Diffusion Works Well

- Stable training (simple MSE loss)
- High-quality samples
- Good diversity (no mode collapse)
- Controllable generation

---

## 7.6 Latent Diffusion (Stable Diffusion)

### Problem with Pixel-Space Diffusion

Processing full-resolution images is expensive:
- 512×512×3 = 786K dimensions per image
- Many denoising steps required

### Solution: Diffusion in Latent Space

```
Image → [VAE Encoder] → Latent → [Diffusion] → Latent → [VAE Decoder] → Image
         (compress)      (64×64)   (denoise)              (decompress)
```

Benefits:
- 8× smaller latent space
- Faster training and inference
- Equally good quality

### Text-to-Image

Add text conditioning via cross-attention:
```
Text → [CLIP/T5] → Text embeddings
                        ↓
Noise → [U-Net with cross-attention to text] → Denoised latent
```

The U-Net learns to denoise **conditioned on text**.

---

## 7.7 Comparison of Generative Models

| Model | Quality | Diversity | Training | Controllability |
|-------|---------|-----------|----------|-----------------|
| VAE | Medium | High | Stable | Medium |
| GAN | High | Medium | Unstable | Medium |
| Diffusion | Very High | High | Stable | High |
| Autoregressive | High | High | Stable | High |

### When to Use What

- **VAE**: Fast inference needed, lower quality OK
- **GAN**: Real-time generation, specific domains
- **Diffusion**: Highest quality, time/compute available
- **Autoregressive**: Text, audio, when order matters

---

## 7.8 Summary

### Key Concepts

1. **Autoencoders** learn compressed representations
2. **VAEs** add probabilistic structure for generation
3. **GANs** use adversarial training (generator vs discriminator)
4. **Diffusion models** learn to reverse a noising process
5. **Latent diffusion** combines VAE compression with diffusion

### Glossary Terms Covered

- Generative Model
- Autoencoder
- Variational Autoencoder (VAE)
- Latent Space
- ELBO
- Generative Adversarial Network (GAN)
- Discriminator, Generator
- Mode Collapse
- Diffusion Model
- Denoising
- Latent Diffusion

### What's Next

Module 8 covers **Advanced Topics**: RAG, agents, deployment, evaluation, and ethics.

---

## References

- Kingma & Welling, "Auto-Encoding Variational Bayes" (VAE)
- Goodfellow et al., "Generative Adversarial Networks"
- Ho et al., "Denoising Diffusion Probabilistic Models"
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion"
- CS231n: Generative Models
