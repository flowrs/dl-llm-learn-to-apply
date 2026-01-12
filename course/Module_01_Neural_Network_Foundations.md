# Module 1: Neural Network Foundations

## Learning Objectives

By the end of this module, you will understand:
- What neural networks are and how they compute
- The biological inspiration and mathematical model of neurons
- The role of weights, biases, and activation functions
- How to construct and reason about multi-layer networks
- The forward pass and score function computation
- Why depth and non-linearity matter

---

## 1.1 From Linear Classifiers to Neural Networks

### The Limitations of Linear Models

In linear classification, we compute scores using a simple formula:

```
s = Wx + b
```

Where:
- `W` is a weight matrix [K × D] (K classes, D features)
- `x` is the input vector [D × 1]
- `b` is a bias vector [K × 1]
- `s` is the score vector [K × 1]

**Visual Example with CIFAR-10**:

```
Input Image (32×32×3 = 3072 pixels):
┌────────────────────┐
│  ████████████████  │
│  ██  Airplane  ██  │
│  ████████████████  │
│  ████████████████  │
└────────────────────┘
         │
         │ Flatten to vector
         ↓
┌─────────────────────────────────────────────────┐
│ x = [0.2, 0.5, 0.1, 0.8, ..., 0.3]  (3072 values) │
└─────────────────────────────────────────────────┘
         │
         │ Multiply by W [10 × 3072]
         ↓
┌─────────────────────────────────────────────────┐
│ s = [2.1, -0.5, 3.2, ..., 0.8]  (10 class scores) │
│     airplane  auto  bird ... truck               │
└─────────────────────────────────────────────────┘
         │
         │ Pick highest score
         ↓
    Prediction: "bird" (score = 3.2)
```

**The Problem**: Linear classifiers can only learn **linear decision boundaries**.

```
Linear Classifier Decision Boundary:

    Feature 2
        │
        │    ○ ○ ○ ○
        │  ○ ○ ○ ○ ○     Class B
        │○ ○ ○ ○ ○ ○
        │─────────────── Decision boundary (a line)
        │● ● ● ● ● ● ●
        │  ● ● ● ● ●     Class A
        │    ● ● ●
        └──────────────── Feature 1

Works well when classes are linearly separable!
```

But real-world data often requires **non-linear boundaries**:

```
XOR Problem (Not Linearly Separable):

    Feature 2
        │
        │  ○           ●
        │
        │
        │  ●           ○
        │
        └──────────────── Feature 1

No single line can separate ○ from ●!

What we need:
        │
        │  ○     │     ●
        │       │
        │───────┼───────
        │       │
        │  ●    │      ○
        │
        └──────────────────

A non-linear decision boundary!
```

### Adding Non-linearity: The Neural Network Solution

A neural network adds intermediate transformations with non-linearities:

```
Two-Layer Network:

Input x                    Hidden Layer h              Output s
[3072]                        [100]                      [10]

  ┌───┐                      ┌───┐                      ┌───┐
  │x_1│─────┐    W_1         │h_1│─────┐    W_2         │s_1│ airplane
  ├───┤     │   [100×3072]   ├───┤     │   [10×100]     ├───┤
  │x_2│─────┼───────────────→│h_2│─────┼───────────────→│s_2│ auto
  ├───┤     │    + ReLU      ├───┤     │                ├───┤
  │...│─────┤                │...│─────┤                │...│ ...
  ├───┤     │                ├───┤     │                ├───┤
  │x_D│─────┘                │h_H│─────┘                │s_K│ truck
  └───┘                      └───┘                      └───┘

Mathematical form:
s = W_2 · ReLU(W_1 · x + b_1) + b_2
```

**Why the non-linearity is CRITICAL**:

Without it, the two matrices collapse to one:
```
s = W_2 · (W_1 · x)      (no non-linearity)
  = (W_2 · W_1) · x      (matrix multiplication is associative)
  = W_combined · x       (still just a linear transformation!)
```

With non-linearity (ReLU):
```
s = W_2 · ReLU(W_1 · x)  (non-linearity in the middle)
  ≠ any single matrix multiplication
  = Can learn non-linear decision boundaries!
```

---

## 1.2 The Neuron: Biological Inspiration and Mathematical Model

### Biological Neurons

The artificial neuron is loosely inspired by biological neurons in the brain:

```
Biological Neuron:

                    Dendrites
                    (inputs)
                       │
              ─────────┼─────────
             /    │    │    │    \
            ↓     ↓    ↓    ↓     ↓
         ┌─────────────────────────┐
         │                         │
         │     Cell Body (Soma)    │
         │     - Integrates        │
         │       signals           │
         │     - "Fires" if        │
         │       threshold met     │
         │                         │
         └───────────┬─────────────┘
                     │
                     │ Axon
                     │ (output)
                     │
                     ↓
              To other neurons
              (via synapses)
```

**Key biological concepts that inspired neural networks**:
- **Dendrites**: Receive signals from other neurons (like inputs)
- **Synapses**: Connections with varying strengths (like weights)
- **Soma**: Integrates signals (like weighted sum)
- **Axon**: Transmits output signal (like neuron output)
- **Firing threshold**: Only activates above threshold (like activation function)

### Mathematical Model of a Neuron

A single artificial neuron computes:

```
y = f(∑ᵢ wᵢxᵢ + b) = f(w·x + b)
```

**Detailed breakdown**:

```
Step 1: Weighted Sum
─────────────────────────────────────────────────────────────────

Input signals:    x = [x₁, x₂, x₃, ..., xₙ]
                      │   │   │       │
                      │   │   │       │
Weights:          w = [w₁, w₂, w₃, ..., wₙ]   (learnable)
                      │   │   │       │
                      ↓   ↓   ↓       ↓
                   w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ
                            │
                            │
                            ↓
                    Weighted sum: z = w·x


Step 2: Add Bias
─────────────────────────────────────────────────────────────────

                    z = w·x + b
                              │
                              │
                    Bias b allows shifting the activation
                    threshold (learnable)


Step 3: Apply Activation Function
─────────────────────────────────────────────────────────────────

                    z ──→ f(z) ──→ y
                          │
                          │
                    Non-linear transformation
                    (e.g., ReLU, sigmoid, tanh)


Complete Neuron Diagram:
─────────────────────────────────────────────────────────────────

        x₁ ───w₁──→╲
                    ╲
        x₂ ───w₂──→─╲
                     ╲    ┌─────────┐      ┌─────────┐
        x₃ ───w₃──→──●───→│ Σ (sum) │─────→│  f(·)   │───→ y
                     ╱    │  + b    │      │ (activ) │
        x₄ ───w₄──→─╱     └─────────┘      └─────────┘
                    ╱
        xₙ ───wₙ──→╱

        Inputs   Weights    Summation      Activation   Output
                           with bias        function
```

### Python Implementation of a Single Neuron

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        """
        Initialize a single neuron.

        Args:
            num_inputs: Number of input connections
        """
        # Initialize weights with small random values
        # Why small? Prevents saturation in sigmoid/tanh
        # Why random? Breaks symmetry between neurons
        self.weights = np.random.randn(num_inputs) * 0.01

        # Initialize bias to zero (common practice)
        self.bias = 0.0

    def forward(self, inputs):
        """
        Compute the neuron's output.

        Args:
            inputs: Input values [num_inputs]

        Returns:
            Output after activation
        """
        # Step 1: Weighted sum
        weighted_sum = np.dot(inputs, self.weights)

        # Step 2: Add bias
        z = weighted_sum + self.bias

        # Step 3: Activation (using sigmoid here)
        output = 1.0 / (1.0 + np.exp(-z))

        return output

# Example usage
neuron = Neuron(num_inputs=3)
inputs = np.array([0.5, -0.2, 0.8])
output = neuron.forward(inputs)
print(f"Output: {output:.4f}")  # e.g., Output: 0.5012
```

### A Neuron as a Linear Classifier

A single neuron with sigmoid activation performs **binary classification**:

```
Decision Boundary of a Single Neuron:
────────────────────────────────────────────────────────────────

The decision boundary is where w·x + b = 0

For 2D input (x₁, x₂) with weights (w₁, w₂) and bias b:
    w₁x₁ + w₂x₂ + b = 0

This is a LINE (hyperplane in higher dimensions):
    x₂ = -(w₁/w₂)x₁ - (b/w₂)


Example: w = [1, 1], b = -1.5

        x₂
         │
       2 │          ╱ Class 1 (y → 1)
         │        ╱  σ(w·x + b) > 0.5
         │      ╱
       1 │    ╱
         │  ╱   Decision boundary:
         │╱     x₁ + x₂ - 1.5 = 0
     ────┼─────────────────────── x₁
         │╲
         │  ╲   Class 0 (y → 0)
         │    ╲ σ(w·x + b) < 0.5
         │
         │

Points on the boundary: (1.5, 0), (0, 1.5), (0.75, 0.75)
```

---

## 1.3 Activation Functions

Activation functions introduce **non-linearity**, enabling networks to learn complex patterns. Without them, a neural network (regardless of depth) would only compute linear transformations.

### Sigmoid

```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Graph and Properties**:

```
Sigmoid Function: σ(x) = 1 / (1 + e^(-x))
────────────────────────────────────────────────────────────────

Output
  1.0 │                            _______________
      │                         __/
      │                       _/
  0.5 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      │                   _/│
      │                 _/  │
      │              __/    │
  0.0 │_____________/       │
      └─────────────────────┼─────────────────────→ Input
                   -4      0       4

Properties:
┌──────────────────────────────────────────────────────────────┐
│ • Range: (0, 1) - outputs can be interpreted as probability  │
│ • σ(0) = 0.5                                                 │
│ • Smooth and differentiable everywhere                       │
│ • Derivative: σ'(x) = σ(x)(1 - σ(x))                        │
│ • Maximum derivative at x=0: σ'(0) = 0.25                   │
└──────────────────────────────────────────────────────────────┘

Derivative (for backpropagation):
────────────────────────────────────────────────────────────────

σ'(x)
 0.25│              ╱╲
     │             ╱  ╲
     │            ╱    ╲
     │           ╱      ╲
     │          ╱        ╲
     │         ╱          ╲
     │        ╱            ╲
   0 │_______╱              ╲_______
     └─────────────────────────────→ x
            -4      0       4

Notice: Derivative → 0 for large |x| (VANISHING GRADIENT!)
```

**Problems with Sigmoid**:

```
Problem 1: VANISHING GRADIENTS
────────────────────────────────────────────────────────────────

When x is very positive or very negative:
    σ(10) ≈ 0.99995    →  σ'(10) ≈ 0.00005  (almost zero!)
    σ(-10) ≈ 0.00005   →  σ'(-10) ≈ 0.00005 (almost zero!)

In backpropagation, gradients multiply:
    ∂L/∂w₁ = ∂L/∂y × ∂y/∂z × ∂z/∂w₁
                      ↑
              σ'(z) ≈ 0 if z is large

Result: Gradients vanish → Early layers don't learn!


Problem 2: NOT ZERO-CENTERED
────────────────────────────────────────────────────────────────

Sigmoid outputs are always positive (0 to 1).

If all inputs to next layer are positive:
    ∂L/∂wᵢ = ∂L/∂z × xᵢ
                     ↑
                 Always positive!

All gradients have the SAME SIGN → zig-zag updates:

Weight space:                  Ideal path:
    w₂                             w₂
     │    ╱→                        │
     │   ╱                          │    →→→→
     │  ╱                           │        ↘
     │ ↗  zig-zag!                  │         →
     │↗                             │
     └──────── w₁                   └──────── w₁
```

### Tanh (Hyperbolic Tangent)

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ) = 2σ(2x) - 1
```

**Graph and Properties**:

```
Tanh Function:
────────────────────────────────────────────────────────────────

Output
  1.0 │                            _______________
      │                         __/
      │                       _/
    0 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      │                   _/│
      │                 _/  │
      │              __/    │
 -1.0 │_____________/       │
      └─────────────────────┼─────────────────────→ Input
                   -4      0       4

Properties:
┌──────────────────────────────────────────────────────────────┐
│ • Range: (-1, 1) - ZERO-CENTERED (better than sigmoid!)     │
│ • tanh(0) = 0                                               │
│ • Derivative: tanh'(x) = 1 - tanh²(x)                       │
│ • Maximum derivative at x=0: tanh'(0) = 1                   │
│ • Still suffers from vanishing gradients at extremes        │
└──────────────────────────────────────────────────────────────┘
```

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

**Graph and Properties**:

```
ReLU Function:
────────────────────────────────────────────────────────────────

Output
      │                    ╱
    4 │                  ╱
      │                ╱
    2 │              ╱
      │            ╱
    0 │__________●─────────────────────→ Input
      │         0│
                 │

      For x ≤ 0: output = 0
      For x > 0: output = x (identity)

Properties:
┌──────────────────────────────────────────────────────────────┐
│ • Range: [0, ∞)                                             │
│ • NO saturation for positive values (gradient = 1)          │
│ • Computationally efficient: just max(0, x)                 │
│ • Creates SPARSE activations (many zeros)                   │
│ • Default choice for hidden layers                          │
└──────────────────────────────────────────────────────────────┘

Derivative:
────────────────────────────────────────────────────────────────

ReLU'(x)
      │
    1 │              ●────────────────
      │              │
      │              │
    0 │──────────────●
      └─────────────────────────────→ x
                    0

      ReLU'(x) = 0 if x < 0
      ReLU'(x) = 1 if x > 0
      ReLU'(0) = undefined (use 0 or 1 in practice)
```

**The "Dying ReLU" Problem**:

```
Dying ReLU Problem:
────────────────────────────────────────────────────────────────

If a neuron's input is consistently negative:
    output = ReLU(w·x + b) = 0    (always!)
    gradient = 0                   (always!)

→ The neuron stops learning entirely ("dies")

Example scenario:
    Initial weights → large negative bias
    → w·x + b < 0 for all training examples
    → Neuron outputs 0 forever
    → No gradient signal to update weights
    → Neuron is "dead"

         Bad initialization                After training
         ─────────────────                 ────────────────
         [Active Neuron]   ────→          [Dead Neuron 💀]
         [Active Neuron]   ────→          [Active Neuron]
         [Active Neuron]   ────→          [Dead Neuron 💀]
         [Active Neuron]   ────→          [Active Neuron]

Up to 40% of neurons can die with bad initialization!
```

### Leaky ReLU and Variants

```
Leaky ReLU(x) = max(αx, x)  where α ≈ 0.01
```

**Graph**:

```
Leaky ReLU (α = 0.1 for visibility):
────────────────────────────────────────────────────────────────

Output
      │                    ╱
    4 │                  ╱
      │                ╱
    2 │              ╱
      │            ╱
    0 │──────────●──────────────────────→ Input
      │        ╱ 0
      │      ╱  (small negative slope)
   -1 │    ╱
      │  ╱
      │╱


Comparison at x = -5:
┌─────────────────────────────────────────┐
│ ReLU(-5)       = 0                      │
│ Leaky ReLU(-5) = -0.05  (with α=0.01)  │
│ → Non-zero gradient, neuron can learn! │
└─────────────────────────────────────────┘
```

**Variants**:

```
Parametric ReLU (PReLU):
    α is LEARNED during training (different per channel)

ELU (Exponential Linear Unit):
    f(x) = x              if x > 0
    f(x) = α(eˣ - 1)      if x ≤ 0
    → Smooth, negative values, zero-centered outputs

SELU (Scaled ELU):
    Self-normalizing properties for deep networks
```

### GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x)  where Φ is the standard Gaussian CDF
        ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Graph**:

```
GELU Function:
────────────────────────────────────────────────────────────────

Output
      │                      ╱
    4 │                    ╱
      │                  ╱
    2 │               _╱
      │            _─╱
    0 │_________─╯──────────────────────→ Input
      │        ╲ 0
      │         ╲_
  -0.2│           ‾‾──────
      │

Key properties:
• Smooth everywhere (unlike ReLU's kink at 0)
• Weights input by its probability under N(0,1)
• Used in BERT, GPT, and most modern transformers
• Small negative region (unlike ReLU)
```

### Softmax (for Output Layer)

```
softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ
```

**Visualization**:

```
Softmax converts scores to probabilities:
────────────────────────────────────────────────────────────────

Input scores (logits):        Softmax output (probabilities):
┌─────────────────┐           ┌─────────────────┐
│ Cat:    2.0     │           │ Cat:    0.659   │ ■■■■■■■
│ Dog:    1.0     │    →      │ Dog:    0.242   │ ■■■
│ Bird:   0.0     │           │ Bird:   0.089   │ ■
│ Fish:  -1.0     │           │ Fish:   0.033   │
└─────────────────┘           └─────────────────┘
                               Sum = 1.000 ✓

Calculation:
    e^2.0 = 7.39    → 7.39 / 11.22 = 0.659
    e^1.0 = 2.72    → 2.72 / 11.22 = 0.242
    e^0.0 = 1.00    → 1.00 / 11.22 = 0.089
    e^-1  = 0.37    → 0.37 / 11.22 = 0.033
    ─────────
    Sum = 11.22

Properties:
• All outputs in (0, 1)
• Sum of outputs = 1 (valid probability distribution)
• Preserves ranking: highest score → highest probability
• Amplifies differences: larger scores get much larger probs
```

### Activation Function Comparison and Recommendations

```
Summary Table:
┌──────────────┬─────────────┬───────────────┬──────────────────┐
│ Function     │ Range       │ Gradient      │ Best For         │
├──────────────┼─────────────┼───────────────┼──────────────────┤
│ Sigmoid      │ (0, 1)      │ Vanishes      │ Binary output    │
│ Tanh         │ (-1, 1)     │ Vanishes      │ RNNs (sometimes) │
│ ReLU         │ [0, ∞)      │ 0 or 1        │ Hidden layers    │
│ Leaky ReLU   │ (-∞, ∞)     │ α or 1        │ If dying ReLU    │
│ GELU         │ (-0.17, ∞)  │ Smooth        │ Transformers     │
│ Softmax      │ (0, 1)      │ Complex       │ Multi-class out  │
└──────────────┴─────────────┴───────────────┴──────────────────┘

Decision Tree:
────────────────────────────────────────────────────────────────

Is this the output layer?
    │
    ├── YES → Classification?
    │           │
    │           ├── Binary → Sigmoid
    │           └── Multi-class → Softmax
    │
    └── NO → Hidden layer
              │
              ├── Building a Transformer? → GELU
              │
              └── Otherwise → ReLU (try Leaky if dying neurons)
```

---

## 1.4 Neural Network Architectures

### Layer-wise Organization

Neural networks organize neurons into **layers**:

```
Fully-Connected (Dense) Neural Network:
────────────────────────────────────────────────────────────────

   Input Layer      Hidden Layer 1    Hidden Layer 2    Output Layer
   (features)        (learned)         (learned)        (predictions)

      ●─────────────────●─────────────────●─────────────────●
      │╲               ╱│╲               ╱│╲               ╱│
      │ ╲             ╱ │ ╲             ╱ │ ╲             ╱ │
      │  ╲           ╱  │  ╲           ╱  │  ╲           ╱  │
      ●───╲─────────╱───●───╲─────────╱───●───╲─────────╱───●
      │    ╲       ╱    │    ╲       ╱    │    ╲       ╱    │
      │     ╲     ╱     │     ╲     ╱     │     ╲     ╱     │
      │      ╲   ╱      │      ╲   ╱      │      ╲   ╱      │
      ●───────╲╱────────●───────╲╱────────●───────╲╱────────●
      │       ╱╲        │       ╱╲        │       ╱╲        │
      │      ╱  ╲       │      ╱  ╲       │      ╱  ╲       │
      │     ╱    ╲      │     ╱    ╲      │     ╱    ╲      │
      ●────╱──────╲─────●────╱──────╲─────●────╱──────╲─────●
      │   ╱        ╲    │   ╱        ╲    │   ╱        ╲    │
      │  ╱          ╲   │  ╱          ╲   │  ╱          ╲   │
      │ ╱            ╲  │ ╱            ╲  │ ╱            ╲  │
      ●───────────────╲─●───────────────╲─●───────────────╲─●

   D=5 inputs         4 neurons         4 neurons        K=3 outputs

Every neuron connects to EVERY neuron in adjacent layers
→ "Fully connected" or "Dense" layer

Layer counts:
• This is a "3-layer network" (3 layers with learnable weights)
• Input layer doesn't count (no parameters)
• Count = number of weight matrices
```

### Network Dimensions and Parameter Counting

```
Example: 3-Layer Network for CIFAR-10
────────────────────────────────────────────────────────────────

Input:  [3072]  (32×32×3 pixels flattened)
   │
   │  W₁: [100 × 3072], b₁: [100]
   ↓
Hidden1: [100] → ReLU
   │
   │  W₂: [100 × 100], b₂: [100]
   ↓
Hidden2: [100] → ReLU
   │
   │  W₃: [10 × 100], b₃: [10]
   ↓
Output: [10]  (class scores)


Parameter Count:
────────────────────────────────────────────────────────────────

Layer 1:
    W₁: 3072 × 100 = 307,200 parameters
    b₁: 100 parameters
    Subtotal: 307,300

Layer 2:
    W₂: 100 × 100 = 10,000 parameters
    b₂: 100 parameters
    Subtotal: 10,100

Layer 3:
    W₃: 100 × 10 = 1,000 parameters
    b₃: 10 parameters
    Subtotal: 1,010

────────────────────────────────────
TOTAL: 318,410 parameters


General formula for FC layer:
    Parameters = (input_size × output_size) + output_size
               = input_size × output_size + bias
```

### Forward Pass Computation

```
Forward Pass Step by Step:
────────────────────────────────────────────────────────────────

Input x: [3072] = [0.2, 0.5, 0.1, ..., 0.8]

Step 1: First Linear Transform
────────────────────────────────
    z₁ = W₁ · x + b₁

    [100×3072] · [3072×1] + [100×1] = [100×1]

    z₁ = [-2.1, 0.5, 1.3, ..., -0.8]   (100 values)

Step 2: First Activation (ReLU)
────────────────────────────────
    h₁ = ReLU(z₁) = max(0, z₁)

    h₁ = [0, 0.5, 1.3, ..., 0]   (negative values → 0)

Step 3: Second Linear Transform
────────────────────────────────
    z₂ = W₂ · h₁ + b₂

    [100×100] · [100×1] + [100×1] = [100×1]

    z₂ = [0.3, -1.2, 0.8, ..., 2.1]

Step 4: Second Activation (ReLU)
────────────────────────────────
    h₂ = ReLU(z₂)

    h₂ = [0.3, 0, 0.8, ..., 2.1]

Step 5: Output Linear Transform
────────────────────────────────
    s = W₃ · h₂ + b₃

    [10×100] · [100×1] + [10×1] = [10×1]

    s = [2.1, -0.5, 3.2, 0.1, ..., -1.2]   (10 class scores)

Step 6: (For training) Softmax
────────────────────────────────
    p = softmax(s)

    p = [0.15, 0.01, 0.45, 0.02, ..., 0.01]   (probabilities)


Prediction: Class with highest score = Class 2 (score 3.2)
```

### Python Implementation

```python
import numpy as np

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def softmax(x):
    """Softmax for numerical stability."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for stability
    return exp_x / np.sum(exp_x)

def forward_pass(x, W1, b1, W2, b2, W3, b3):
    """
    Forward pass through a 3-layer neural network.

    Args:
        x: Input vector [D]
        W1, b1: First layer weights [H1 × D] and biases [H1]
        W2, b2: Second layer weights [H2 × H1] and biases [H2]
        W3, b3: Output layer weights [K × H2] and biases [K]

    Returns:
        scores: Class scores [K]
        cache: Intermediate values for backprop
    """
    # Layer 1
    z1 = np.dot(W1, x) + b1      # Linear transform
    h1 = relu(z1)                 # Activation

    # Layer 2
    z2 = np.dot(W2, h1) + b2     # Linear transform
    h2 = relu(z2)                 # Activation

    # Output layer (no activation - raw scores)
    scores = np.dot(W3, h2) + b3

    # Cache for backpropagation
    cache = (x, z1, h1, z2, h2, scores)

    return scores, cache

# Example dimensions
D = 3072   # Input (CIFAR-10)
H1 = 100   # Hidden 1
H2 = 100   # Hidden 2
K = 10     # Output (10 classes)

# Initialize weights (we'll cover proper initialization in Module 2)
W1 = np.random.randn(H1, D) * 0.01
b1 = np.zeros(H1)
W2 = np.random.randn(H2, H1) * 0.01
b2 = np.zeros(H2)
W3 = np.random.randn(K, H2) * 0.01
b3 = np.zeros(K)

# Forward pass
x = np.random.randn(D)  # Random input
scores, cache = forward_pass(x, W1, b1, W2, b2, W3, b3)
probabilities = softmax(scores)

print(f"Scores: {scores}")
print(f"Probabilities: {probabilities}")
print(f"Predicted class: {np.argmax(scores)}")
```

---

## 1.5 Representational Power

### Universal Approximation Theorem

**Theorem** (Cybenko, 1989; Hornik, 1991): A neural network with **one hidden layer** and enough neurons can approximate any continuous function on a compact domain to arbitrary precision.

```
Universal Approximation Illustrated:
────────────────────────────────────────────────────────────────

Target function (any smooth function):

f(x)
  │      ╱╲
  │     ╱  ╲        ╱╲
  │    ╱    ╲      ╱  ╲
  │   ╱      ╲    ╱    ╲
  │  ╱        ╲  ╱      ╲
  │ ╱          ╲╱        ╲
  │╱                      ╲
  └─────────────────────────── x

Neural network approximation (with enough hidden neurons):

f(x)
  │      ╱╲              ← Each "bump" comes from a neuron
  │     ╱  ╲        ╱╲
  │    ╱    ╲      ╱  ╲   ← ReLU neurons create piecewise
  │   ╱      ╲    ╱    ╲    linear approximations
  │  ╱        ╲  ╱      ╲
  │ ╱          ╲╱        ╲
  │╱                      ╲
  └─────────────────────────── x

More neurons = finer approximation
```

**However**, this doesn't mean single-layer networks are optimal:
- May require exponentially many neurons
- Deeper networks are more **parameter-efficient**
- Depth enables **hierarchical feature learning**

### Why Depth Matters: Hierarchical Features

```
Deep Networks Learn Hierarchical Representations:
────────────────────────────────────────────────────────────────

Input (Image of a face)
        │
        ↓
┌───────────────────────────────────────────────────────────────┐
│ Layer 1: Edge Detectors                                       │
│                                                               │
│   ┌───┐ ┌───┐ ┌───┐ ┌───┐                                    │
│   │ / │ │ \ │ │ ─ │ │ │ │  Horizontal, vertical, diagonal    │
│   └───┘ └───┘ └───┘ └───┘  edges at various orientations     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ↓
┌───────────────────────────────────────────────────────────────┐
│ Layer 2: Textures and Simple Shapes                          │
│                                                               │
│   ┌───┐ ┌───┐ ┌───┐ ┌───┐                                    │
│   │╱╲╱│ │▓▓▓│ │ ○ │ │╲╱╲│  Corners, circles, patterns        │
│   └───┘ └───┘ └───┘ └───┘  from combinations of edges        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ↓
┌───────────────────────────────────────────────────────────────┐
│ Layer 3: Object Parts                                         │
│                                                               │
│   ┌───┐ ┌───┐ ┌───┐ ┌───┐                                    │
│   │👁 │ │👃 │ │👄 │ │👂│  Eyes, noses, mouths, ears           │
│   └───┘ └───┘ └───┘ └───┘  from combinations of textures     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ↓
┌───────────────────────────────────────────────────────────────┐
│ Layer 4: Objects                                              │
│                                                               │
│   ┌───────┐                                                   │
│   │  😊   │  Complete face from combinations of parts        │
│   └───────┘                                                   │
│                                                               │
└───────────────────────────────────────────────────────────────┘

This hierarchy emerges automatically from training!
No one programs "look for eyes" - the network learns it.
```

### Depth vs Width: Parameter Efficiency

```
Example: Approximating a complex function
────────────────────────────────────────────────────────────────

Shallow network (1 hidden layer):
    Might need 1,000,000 neurons to approximate function F

Deep network (10 hidden layers):
    Might need only 100 neurons per layer = 1,000 total

Why? Deep networks can REUSE features:
    Layer 1: Learns "horizontal edge"
    Layer 2: Uses "horizontal edge" to build "line pattern"
    Layer 3: Uses "line pattern" to build "grid texture"
    ...

Shallow networks must learn each pattern independently.


Intuition from circuits:
────────────────────────────────────────────────────────────────

Computing x₁ × x₂ × x₃ × ... × xₙ (product of n numbers)

Shallow (parallel):         Deep (tree structure):
    x₁ ─┐                       x₁ ─┐
    x₂ ─┤                       x₂ ─┴─●─┐
    x₃ ─┤                              │
    x₄ ─┼──→ [×] → result       x₃ ─┐  ├─●─┐
    ... │                       x₄ ─┴─●─┘  │
    xₙ ─┘                                  ├─● → result
                                x₅ ─┐      │
Needs n inputs to             x₆ ─┴─●─┐   │
one giant multiplier           ...    ├─●─┘
                                xₙ ─┴─●─┘

                              Only needs log(n) layers!
```

---

## 1.6 Network Sizing and Overfitting

### The Bias-Variance Tradeoff

```
Model Complexity vs. Error:
────────────────────────────────────────────────────────────────

Error
  │
  │╲
  │ ╲    Training Error
  │  ╲
  │   ╲
  │    ╲___
  │        ‾‾‾───___
  │                 ‾‾‾‾───────────────
  │
  │        ╱‾‾‾‾╲
  │       ╱      ╲
  │      ╱        ╲____
  │     ╱              ‾‾‾───___  Validation Error
  │    ╱                        ‾‾‾──
  │___╱                               ‾───
  │
  └────────────────────────────────────────────────────
         Simple                 Complex
     (few parameters)      (many parameters)
                    │
                    │ Sweet spot
                    ↓

UNDERFITTING              GOOD FIT              OVERFITTING
(high bias)                                     (high variance)
```

**Visualizing the Problem**:

```
Underfitting (model too simple):
────────────────────────────────────────────────────────────────
    │
    │  ○   ●   ○                True boundary is curved
    │    ●   ○   ●              but model only learns
    │  ○   ●   ○   ●            a straight line
    │──────────────────
    │●   ○   ●   ○
    │  ●   ○   ●
    │

Good Fit:
────────────────────────────────────────────────────────────────
    │
    │  ○   ●   ○
    │    ●   ○   ●   ╲         Model captures
    │  ○   ●╱○───●    ╲        the true pattern
    │      ╱‾‾‾‾‾╲─────
    │●   ○╱  ●   ○
    │  ●  ╱○   ●
    │

Overfitting (model too complex):
────────────────────────────────────────────────────────────────
    │
    │  ○   ●   ○              Model memorizes noise
    │    ●╱‾╲○╱╲●             and training points
    │  ○ ╱●  ○   ●            but won't generalize
    │   ╱╲  ╱╲  ╱╲─
    │● ╱○ ╲╱● ╲╱○
    │ ●   ○   ●
    │
```

### Key Principle: Don't Reduce Network Size to Prevent Overfitting

```
WRONG Approach:
────────────────────────────────────────────────────────────────

"Model is overfitting, let's make it smaller!"

    Large Network          →    Small Network
    (overfits)                  (can't learn complex patterns)

This limits what the network CAN learn.


CORRECT Approach:
────────────────────────────────────────────────────────────────

"Model is overfitting, let's add regularization!"

    Large Network          →    Large Network + Regularization
    (overfits)                  (learns well, generalizes well)

    Regularization techniques (covered in Module 2):
    • L2 regularization (weight decay)
    • Dropout
    • Data augmentation
    • Early stopping
    • Batch normalization

Keep capacity high, control complexity through regularization.


Why larger is often better:
────────────────────────────────────────────────────────────────

Loss Landscape Visualization:

Small Network:                Large Network:
        ╱╲    ╱╲                    ╲      ╱
       ╱  ╲  ╱  ╲                    ╲    ╱
      ╱    ╲╱    ╲                    ╲  ╱
     ╱             ╲                   ╲╱
    ╱               ╲             ╲         ╱
                                   ╲   ●   ╱  Many good
Many sharp local minima             ╲     ╱   local minima
(hard to optimize)                   ╲   ╱    (easier!)
                                      ╲ ╱
```

---

## 1.7 Tensors: The Data Structure of Deep Learning

### What is a Tensor?

A **tensor** is a multi-dimensional array—the fundamental data structure in deep learning:

```
Tensor Dimensions (also called "rank" or "order"):
────────────────────────────────────────────────────────────────

0D Tensor (Scalar):
    5
    Just a single number

1D Tensor (Vector):
    [1, 2, 3, 4, 5]
    A sequence of numbers
    Shape: (5,)

2D Tensor (Matrix):
    ┌           ┐
    │ 1  2  3   │
    │ 4  5  6   │
    └           ┘
    Rows and columns
    Shape: (2, 3)

3D Tensor:
    ┌─────────────┐
   ╱│ 1  2  3    │╱│
  ╱ │ 4  5  6   │╱ │
 ╱  └───────────┘  │
│   │           │  │
│   │           │  │
│   └───────────┼──┘
└───────────────┘
    Like a "stack of matrices"
    Shape: (depth, rows, cols)

4D Tensor:
    Common for batches of images
    Shape: (batch, channels, height, width)
           or (batch, height, width, channels)
```

### Common Tensor Shapes in Deep Learning

```
Image Data:
────────────────────────────────────────────────────────────────

Single grayscale image:
    Shape: (H, W) = (28, 28) for MNIST

    ┌──────────────────────┐
    │ ████████████████████ │
    │ ██                ██ │
    │ ██      ██        ██ │  28 pixels
    │ ██      ██        ██ │
    │ ██    ████████    ██ │
    │ ██                ██ │
    │ ████████████████████ │
    └──────────────────────┘
          28 pixels

Single RGB image:
    Shape: (C, H, W) = (3, 224, 224) or (H, W, C)

         R           G           B
    ┌────────┐  ┌────────┐  ┌────────┐
    │        │  │        │  │        │
    │  Red   │  │ Green  │  │  Blue  │
    │ Channel│  │ Channel│  │ Channel│
    │        │  │        │  │        │
    └────────┘  └────────┘  └────────┘

Batch of RGB images:
    Shape: (N, C, H, W) = (32, 3, 224, 224)

    32 images, each 3×224×224


Text Data:
────────────────────────────────────────────────────────────────

Token IDs:
    Shape: (L,) = (512,) for single sequence
    [101, 2023, 2003, 1037, 6251, 102, 0, 0, ...]
      │     │     │     │     │    │   │
     [CLS] This   is    a   sentence [SEP] [PAD]...

Batch of sequences:
    Shape: (N, L) = (16, 512)
    16 sequences, each up to 512 tokens

Embedded sequences:
    Shape: (N, L, D) = (16, 512, 768)
    16 sequences, 512 tokens each, 768-dim embeddings
```

### PyTorch Tensor Operations

```python
import torch

# Creating tensors
────────────────────────────────────────────────────────────────

# From data
x = torch.tensor([1, 2, 3, 4])           # 1D tensor
x = torch.tensor([[1, 2], [3, 4]])       # 2D tensor

# Random tensors
x = torch.randn(3, 4)                     # Normal distribution
x = torch.rand(3, 4)                      # Uniform [0, 1)
x = torch.zeros(3, 4)                     # All zeros
x = torch.ones(3, 4)                      # All ones

# Tensor properties
────────────────────────────────────────────────────────────────

x = torch.randn(32, 3, 224, 224)          # Batch of images

x.shape          # torch.Size([32, 3, 224, 224])
x.dtype          # torch.float32
x.device         # device(type='cpu') or device(type='cuda', index=0)
x.ndim           # 4 (number of dimensions)
x.numel()        # 32 * 3 * 224 * 224 = 4,816,896 (total elements)


# Reshaping operations
────────────────────────────────────────────────────────────────

x = torch.randn(32, 3, 224, 224)

# Flatten all but batch dimension
x.view(32, -1)                            # [32, 150528]
x.reshape(32, -1)                         # [32, 150528] (same but safer)

# Permute dimensions (reorder axes)
x.permute(0, 2, 3, 1)                     # [32, 224, 224, 3]
                                          # (NCHW → NHWC)

# Add/remove dimensions
x.unsqueeze(0)                            # Add dim at position 0
x.squeeze()                               # Remove dims of size 1

# Transpose (swap two dimensions)
x.transpose(1, 2)                         # Swap dims 1 and 2


# Indexing and slicing
────────────────────────────────────────────────────────────────

x = torch.randn(32, 3, 224, 224)

x[0]              # First image: [3, 224, 224]
x[:, 0]           # Red channel of all images: [32, 224, 224]
x[0, 0, :10, :10] # Top-left 10×10 of first image, red channel
x[::2]            # Every other image: [16, 3, 224, 224]


# Mathematical operations
────────────────────────────────────────────────────────────────

a = torch.randn(3, 4)
b = torch.randn(3, 4)

# Element-wise
a + b             # Addition
a * b             # Multiplication
a ** 2            # Square
torch.exp(a)      # Exponential
torch.relu(a)     # ReLU

# Matrix operations
c = torch.randn(4, 5)
a @ c             # Matrix multiplication [3, 5]
torch.mm(a, c)    # Same thing

# Reductions
a.sum()           # Sum all elements
a.mean()          # Mean of all elements
a.sum(dim=1)      # Sum along dimension 1: [3]
a.max(dim=1)      # Max along dimension 1 (returns values and indices)
```

---

## 1.8 Summary

### Key Concepts Covered

```
Module 1 Summary:
────────────────────────────────────────────────────────────────

1. LINEAR MODELS are limited to linear decision boundaries
   ↓
2. NEURAL NETWORKS add non-linear transformations
   ↓
3. NEURONS compute: y = f(w·x + b)
   • Weights (w): Control input importance
   • Bias (b): Shift activation threshold
   • Activation (f): Introduce non-linearity
   ↓
4. ACTIVATION FUNCTIONS enable complex learning
   • Sigmoid: (0,1), used for binary output
   • Tanh: (-1,1), zero-centered
   • ReLU: [0,∞), default for hidden layers
   • GELU: Used in transformers
   • Softmax: Probabilities for multi-class
   ↓
5. LAYERS organize neurons
   • Fully-connected: Every neuron connects to all previous
   • Parameters: (input × output) + output
   ↓
6. FORWARD PASS computes output from input
   • z = Wx + b (linear)
   • h = f(z) (activation)
   • Repeat for each layer
   ↓
7. DEPTH enables hierarchical feature learning
   • Edges → Textures → Parts → Objects
   • More parameter-efficient than width
   ↓
8. TENSORS are multi-dimensional arrays
   • Images: (N, C, H, W)
   • Text: (N, L, D)
```

### Glossary Terms Covered

| Term | Definition |
|------|------------|
| **Artificial Neural Network (ANN)** | Computing system inspired by biological neural networks |
| **Neuron (Node, Unit)** | Basic computational unit: y = f(w·x + b) |
| **Weight** | Learnable parameter controlling input strength |
| **Bias** | Learnable parameter shifting activation threshold |
| **Parameter** | Any learnable value in the network |
| **Layer** | Collection of neurons at the same depth |
| **Deep Learning** | Neural networks with multiple hidden layers |
| **Forward Pass** | Computing output from input through the network |
| **Activation Function** | Non-linear transformation (ReLU, sigmoid, etc.) |
| **ReLU** | max(0, x) - most common hidden layer activation |
| **Sigmoid** | 1/(1+e^(-x)) - squashes to (0,1) |
| **Tanh** | (e^x - e^(-x))/(e^x + e^(-x)) - squashes to (-1,1) |
| **Softmax** | Converts scores to probabilities |
| **GELU** | x·Φ(x) - used in transformers |
| **Tensor** | Multi-dimensional array |
| **Feedforward Neural Network** | Network where information flows one direction |

---

## Exercises

### Exercise 1: Parameter Counting
A network has layers with dimensions [784, 256, 128, 10]. Calculate the total number of parameters.

<details>
<summary>Solution</summary>

```
Layer 1: 784 → 256
    Weights: 784 × 256 = 200,704
    Biases: 256
    Subtotal: 200,960

Layer 2: 256 → 128
    Weights: 256 × 128 = 32,768
    Biases: 128
    Subtotal: 32,896

Layer 3: 128 → 10
    Weights: 128 × 10 = 1,280
    Biases: 10
    Subtotal: 1,290

Total: 200,960 + 32,896 + 1,290 = 235,146 parameters
```
</details>

### Exercise 2: ReLU Computation
Compute ReLU([-2, 0.5, 3, -1, 0]).

<details>
<summary>Solution</summary>

```
ReLU(x) = max(0, x)

ReLU(-2)  = max(0, -2)  = 0
ReLU(0.5) = max(0, 0.5) = 0.5
ReLU(3)   = max(0, 3)   = 3
ReLU(-1)  = max(0, -1)  = 0
ReLU(0)   = max(0, 0)   = 0

Result: [0, 0.5, 3, 0, 0]
```
</details>

### Exercise 3: Softmax Computation
Given scores [2.0, 1.0, 0.1], compute softmax probabilities.

<details>
<summary>Solution</summary>

```
softmax(x_i) = e^(x_i) / Σ e^(x_j)

e^2.0 = 7.389
e^1.0 = 2.718
e^0.1 = 1.105

Sum = 7.389 + 2.718 + 1.105 = 11.212

P(class 0) = 7.389 / 11.212 = 0.659
P(class 1) = 2.718 / 11.212 = 0.242
P(class 2) = 1.105 / 11.212 = 0.099

Result: [0.659, 0.242, 0.099]
Sum = 1.0 ✓
```
</details>

### Exercise 4: Implementation
Implement a 2-layer neural network forward pass in NumPy.

<details>
<summary>Solution</summary>

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def forward_2layer(x, W1, b1, W2, b2):
    """
    2-layer network forward pass.

    Args:
        x: Input [D]
        W1: [H, D], b1: [H]
        W2: [K, H], b2: [K]

    Returns:
        probabilities [K]
    """
    # Layer 1
    z1 = np.dot(W1, x) + b1
    h1 = relu(z1)

    # Layer 2
    z2 = np.dot(W2, h1) + b2

    # Output probabilities
    probs = softmax(z2)

    return probs

# Test
D, H, K = 10, 5, 3
x = np.random.randn(D)
W1 = np.random.randn(H, D) * 0.01
b1 = np.zeros(H)
W2 = np.random.randn(K, H) * 0.01
b2 = np.zeros(K)

probs = forward_2layer(x, W1, b1, W2, b2)
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum():.4f}")  # Should be 1.0
```
</details>

---

## References and Further Reading

### Lectures
- [CS231n Lecture 3: Loss Functions and Optimization](http://cs231n.stanford.edu/slides/2024/lecture_3.pdf)
- [CS231n Lecture 4: Neural Networks and Backpropagation](http://cs231n.stanford.edu/slides/2024/lecture_4.pdf)
- [CS224N Lecture 3: Backprop and Neural Networks](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture03-neuralnets.pdf)

### Course Notes
- [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- [CS231n: Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/)

### Papers
- [Cybenko (1989): Approximation by Superpositions of a Sigmoidal Function](https://doi.org/10.1007/BF02551274) - Universal approximation theorem
- [Hornik (1991): Approximation Capabilities of Multilayer Feedforward Networks](https://doi.org/10.1016/0893-6080(91)90009-T)
- [Glorot & Bengio (2010): Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html) - Xavier initialization
- [He et al. (2015): Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) - He initialization, PReLU
- [Hendrycks & Gimpel (2016): Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

### Books
- [Goodfellow, Bengio, Courville: Deep Learning, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) - Free online
- [Nielsen: Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online

### Interactive Resources
- [TensorFlow Playground](https://playground.tensorflow.org/) - Visualize neural networks
- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/topics/neural-networks) - Excellent visual explanations
- [Distill.pub](https://distill.pub/) - Research explanations with interactive visualizations
