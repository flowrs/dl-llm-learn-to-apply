# Module 0: Mathematical Foundations for Deep Learning

This module covers the essential mathematics you need to understand deep learning.
Each section maps to concepts used in later modules.

---

## Overview: Math You'll Need

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH FOUNDATIONS ROADMAP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LINEAR ALGEBRA                        CALCULUS                             │
│  ──────────────                        ────────                             │
│  • Vectors and matrices                • Derivatives                        │
│  • Matrix multiplication               • Chain rule                         │
│  • Transpose, inverse                  • Partial derivatives                │
│  • Dot products                        • Gradients                          │
│  • Norms                               • Jacobians                          │
│  • Eigenvalues/vectors                 • Optimization                       │
│  • Broadcasting                                                             │
│                                                                             │
│  PROBABILITY                           INFORMATION THEORY                   │
│  ───────────                           ──────────────────                   │
│  • Probability distributions           • Entropy                            │
│  • Bayes' theorem                      • Cross-entropy                      │
│  • Expectation, variance               • KL divergence                      │
│  • Sampling                            • Mutual information                 │
│  • Maximum likelihood                                                       │
│                                                                             │
│  MODULE DEPENDENCIES                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  Module 1-2 (Foundations):     Vectors, matrices, derivatives, chain rule   │
│  Module 3-4 (CNNs):            Convolutions, matrix ops, gradients          │
│  Module 5 (RNNs):              Matrix multiplication, gradients over time   │
│  Module 6 (Transformers/LLMs): Dot products, softmax, matrix factorization  │
│  Module 7 (Genertic Models):   Probability, KL divergence, sampling         │
│  Module 8 (Advanced):          All of the above + information theory        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Linear Algebra

### 1.1 Vectors

A vector is an ordered list of numbers. In deep learning, vectors represent:
- Input features (e.g., pixel values, word embeddings)
- Hidden states
- Gradients
- Model parameters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR BASICS                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NOTATION                                                                   │
│  ────────                                                                   │
│                                                                             │
│  Column vector (default in DL):        Row vector:                          │
│       ┌───┐                                                                 │
│       │ 1 │                            [1, 2, 3] or [1  2  3]               │
│  x =  │ 2 │  ∈ ℝ³                                                          │
│       │ 3 │                                                                 │
│       └───┘                                                                 │
│                                                                             │
│  VECTOR OPERATIONS                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  Addition (element-wise):              Scalar multiplication:               │
│  ┌───┐   ┌───┐   ┌───┐                      ┌───┐   ┌───┐                  │
│  │ 1 │ + │ 4 │ = │ 5 │                      │ 1 │   │ 2 │                  │
│  │ 2 │   │ 5 │   │ 7 │                 2 ×  │ 2 │ = │ 4 │                  │
│  │ 3 │   │ 6 │   │ 9 │                      │ 3 │   │ 6 │                  │
│  └───┘   └───┘   └───┘                      └───┘   └───┘                  │
│                                                                             │
│  DOT PRODUCT (Inner Product)                                                │
│  ───────────────────────────                                                │
│                                                                             │
│  x · y = Σᵢ xᵢyᵢ = x₁y₁ + x₂y₂ + ... + xₙyₙ                                │
│                                                                             │
│  Example:                                                                   │
│  [1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32                │
│                                                                             │
│  Geometric interpretation:                                                  │
│  x · y = ‖x‖ ‖y‖ cos(θ)                                                    │
│                                                                             │
│       x                                                                     │
│       ↗                                                                     │
│      θ                                                                      │
│     ──────→ y                                                               │
│                                                                             │
│  • If θ = 0° (same direction):  x · y = ‖x‖‖y‖  (maximum)                  │
│  • If θ = 90° (perpendicular):  x · y = 0                                  │
│  • If θ = 180° (opposite):      x · y = -‖x‖‖y‖ (minimum)                  │
│                                                                             │
│  WHY IT MATTERS: Attention scores are dot products!                         │
│  score(query, key) = query · key                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Vector Norms

Norms measure the "size" or "length" of a vector.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR NORMS                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  L2 NORM (Euclidean norm) - Most common in DL                               │
│  ─────────────────────────────────────────────                              │
│                                                                             │
│  ‖x‖₂ = √(x₁² + x₂² + ... + xₙ²) = √(Σᵢ xᵢ²)                               │
│                                                                             │
│  Example: x = [3, 4]                                                        │
│  ‖x‖₂ = √(3² + 4²) = √(9 + 16) = √25 = 5                                   │
│                                                                             │
│  Geometric meaning: Straight-line distance from origin                      │
│                                                                             │
│        y                                                                    │
│        ↑                                                                    │
│      4 │      • (3,4)                                                       │
│        │    ╱                                                               │
│        │  ╱  ‖x‖₂ = 5                                                       │
│        │╱                                                                   │
│        └──────────→ x                                                       │
│             3                                                               │
│                                                                             │
│  L1 NORM (Manhattan norm)                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  ‖x‖₁ = |x₁| + |x₂| + ... + |xₙ| = Σᵢ |xᵢ|                                 │
│                                                                             │
│  Example: x = [3, 4]                                                        │
│  ‖x‖₁ = |3| + |4| = 7                                                      │
│                                                                             │
│  Geometric meaning: "Taxicab" distance (moving along axes)                  │
│                                                                             │
│        y                                                                    │
│        ↑                                                                    │
│      4 │──────• (3,4)                                                       │
│        │      │                                                             │
│        │      │  ‖x‖₁ = 3 + 4 = 7                                          │
│        │      │                                                             │
│        └──────┴────→ x                                                      │
│             3                                                               │
│                                                                             │
│  L∞ NORM (Max norm)                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  ‖x‖∞ = max(|x₁|, |x₂|, ..., |xₙ|)                                         │
│                                                                             │
│  Example: x = [3, 4]                                                        │
│  ‖x‖∞ = max(3, 4) = 4                                                      │
│                                                                             │
│  USE IN DEEP LEARNING                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  • L2 norm: Weight decay regularization (‖W‖₂²)                            │
│  • L1 norm: Sparsity-inducing regularization                                │
│  • Gradient norms: Gradient clipping (‖∇‖ ≤ threshold)                     │
│  • Embedding norms: Cosine similarity (normalize to unit vectors)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Matrices

A matrix is a 2D array of numbers. In deep learning:
- **Weight matrices** connect layers
- **Input batches** are matrices (batch × features)
- **Images** are matrices (or 3D tensors with channels)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATRIX BASICS                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NOTATION                                                                   │
│  ────────                                                                   │
│                                                                             │
│  Matrix A ∈ ℝᵐˣⁿ  (m rows, n columns)                                      │
│                                                                             │
│       ┌─────────────┐                                                       │
│       │ a₁₁ a₁₂ a₁₃ │                                                       │
│  A =  │ a₂₁ a₂₂ a₂₃ │   A[i,j] = element at row i, column j                │
│       │ a₃₁ a₃₂ a₃₃ │                                                       │
│       └─────────────┘                                                       │
│                                                                             │
│  Example: 2×3 matrix                                                        │
│       ┌─────────┐                                                           │
│  A =  │ 1  2  3 │    A[0,1] = 2 (row 0, column 1)                          │
│       │ 4  5  6 │    A[1,2] = 6 (row 1, column 2)                          │
│       └─────────┘                                                           │
│                                                                             │
│  TRANSPOSE                                                                  │
│  ─────────                                                                  │
│                                                                             │
│  Swap rows and columns: (Aᵀ)ᵢⱼ = Aⱼᵢ                                       │
│                                                                             │
│       ┌─────────┐         ┌───────┐                                         │
│  A =  │ 1  2  3 │   Aᵀ =  │ 1  4 │                                         │
│       │ 4  5  6 │         │ 2  5 │                                         │
│       └─────────┘         │ 3  6 │                                         │
│       (2×3)               └───────┘                                         │
│                           (3×2)                                             │
│                                                                             │
│  Properties:                                                                │
│  • (Aᵀ)ᵀ = A                                                               │
│  • (AB)ᵀ = BᵀAᵀ                                                            │
│  • (A + B)ᵀ = Aᵀ + Bᵀ                                                      │
│                                                                             │
│  SPECIAL MATRICES                                                           │
│  ────────────────                                                           │
│                                                                             │
│  Identity matrix I:           Zero matrix 0:                                │
│  ┌─────────┐                  ┌─────────┐                                   │
│  │ 1  0  0 │                  │ 0  0  0 │                                   │
│  │ 0  1  0 │    AI = IA = A   │ 0  0  0 │    A + 0 = A                     │
│  │ 0  0  1 │                  │ 0  0  0 │                                   │
│  └─────────┘                  └─────────┘                                   │
│                                                                             │
│  Symmetric matrix:  A = Aᵀ                                                  │
│  Diagonal matrix:   Non-zero only on diagonal                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Matrix Multiplication

The most important operation in deep learning!

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATRIX MULTIPLICATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RULE: (m×n) × (n×p) = (m×p)                                               │
│        Inner dimensions must match!                                         │
│                                                                             │
│       A          ×        B         =        C                              │
│    (m × n)            (n × p)            (m × p)                            │
│                                                                             │
│  HOW IT WORKS                                                               │
│  ────────────                                                               │
│                                                                             │
│  C[i,j] = Σₖ A[i,k] × B[k,j]  (dot product of row i with column j)         │
│                                                                             │
│  Example:                                                                   │
│       ┌───────┐       ┌───────┐       ┌───────────┐                         │
│       │ 1  2 │       │ 5  6 │       │ 1×5+2×7  1×6+2×8 │   ┌─────────┐     │
│  A =  │ 3  4 │ × B = │ 7  8 │   =   │ 3×5+4×7  3×6+4×8 │ = │ 19  22 │     │
│       └───────┘       └───────┘       └───────────────────┘   │ 43  50 │     │
│       (2×2)           (2×2)                                   └─────────┘     │
│                                                               (2×2)         │
│                                                                             │
│  STEP BY STEP:                                                              │
│                                                                             │
│  C[0,0] = [1,2] · [5,7] = 1×5 + 2×7 = 5 + 14 = 19                          │
│  C[0,1] = [1,2] · [6,8] = 1×6 + 2×8 = 6 + 16 = 22                          │
│  C[1,0] = [3,4] · [5,7] = 3×5 + 4×7 = 15 + 28 = 43                         │
│  C[1,1] = [3,4] · [6,8] = 3×6 + 4×8 = 18 + 32 = 50                         │
│                                                                             │
│  VISUAL INTUITION                                                           │
│  ────────────────                                                           │
│                                                                             │
│        Column j of B                                                        │
│            ↓                                                                │
│       ┌─────────┐                                                           │
│       │    │    │                                                           │
│  A:   │────●────│──→ Row i of A                                            │
│       │    │    │                                                           │
│       └─────────┘                                                           │
│            ↓                                                                │
│       C[i,j] = dot product                                                  │
│                                                                             │
│  PROPERTIES (important!)                                                    │
│  ──────────────────────                                                     │
│                                                                             │
│  • NOT commutative: AB ≠ BA (usually)                                      │
│  • Associative: (AB)C = A(BC)                                               │
│  • Distributive: A(B + C) = AB + AC                                         │
│                                                                             │
│  DEEP LEARNING CONNECTION                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  Linear layer: y = Wx + b                                                   │
│                                                                             │
│  Input x:     (batch_size × input_features)                                 │
│  Weights W:   (input_features × output_features)                            │
│  Output y:    (batch_size × output_features)                                │
│                                                                             │
│  Example: batch=32, input=784, output=256                                   │
│  (32 × 784) × (784 × 256) = (32 × 256)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Matrix-Vector Multiplication

A special case that's everywhere in neural networks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATRIX-VECTOR MULTIPLICATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  y = Wx  where W is (m×n), x is (n×1), y is (m×1)                          │
│                                                                             │
│  TWO INTERPRETATIONS                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  1. ROW-WISE: Each output is dot product of W row with x                    │
│                                                                             │
│     ┌─────────┐   ┌───┐     ┌─────────────────────────┐                     │
│     │ w₁ᵀ    │   │   │     │ y₁ = w₁ · x = Σⱼ w₁ⱼxⱼ │                     │
│     │ w₂ᵀ    │ × │ x │  =  │ y₂ = w₂ · x = Σⱼ w₂ⱼxⱼ │                     │
│     │ w₃ᵀ    │   │   │     │ y₃ = w₃ · x = Σⱼ w₃ⱼxⱼ │                     │
│     └─────────┘   └───┘     └─────────────────────────┘                     │
│                                                                             │
│  2. COLUMN-WISE: Output is weighted sum of W columns                        │
│                                                                             │
│     y = x₁·(col 1 of W) + x₂·(col 2 of W) + ...                            │
│                                                                             │
│     ┌───┐       ┌───┐       ┌───┐                                           │
│     │ 1 │       │ 2 │       │ 5 │                                           │
│     │ 4 │ × 1 + │ 5 │ × 2 + │ 8 │ × 3 = ...                                │
│     │ 7 │       │ 8 │       │11 │                                           │
│     └───┘       └───┘       └───┘                                           │
│                                                                             │
│  EXAMPLE: Simple Neural Network Layer                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  Input: x = [1, 2, 3]ᵀ (3 features)                                        │
│  Weights: W = ┌─────────┐  (2×3, maps 3 features to 2)                     │
│               │ 1  2  3 │                                                   │
│               │ 4  5  6 │                                                   │
│               └─────────┘                                                   │
│                                                                             │
│  Output: y = Wx = ┌───────────────────────┐ = ┌────┐                       │
│                   │ 1×1 + 2×2 + 3×3 = 14  │   │ 14 │                       │
│                   │ 4×1 + 5×2 + 6×3 = 32  │   │ 32 │                       │
│                   └───────────────────────┘   └────┘                       │
│                                                                             │
│  This is exactly what a linear layer does (before activation)!              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.6 Eigenvalues and Eigenvectors

Important for understanding PCA, covariance, and some optimization concepts.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EIGENVALUES AND EIGENVECTORS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEFINITION                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  For a square matrix A, if:                                                 │
│                                                                             │
│      Av = λv                                                                │
│                                                                             │
│  Then:                                                                      │
│  • v is an eigenvector of A                                                 │
│  • λ is the corresponding eigenvalue                                        │
│                                                                             │
│  INTUITION                                                                  │
│  ─────────                                                                  │
│                                                                             │
│  Most vectors change direction when multiplied by A.                        │
│  Eigenvectors only get scaled (stretched or shrunk).                        │
│                                                                             │
│     Regular vector:              Eigenvector:                               │
│        ↗                            ↗                                       │
│      x    ───A───→   Ax              v   ───A───→   λv                     │
│                     ↘                              ↗                        │
│              (rotated + scaled)            (only scaled)                    │
│                                                                             │
│  EXAMPLE                                                                    │
│  ───────                                                                    │
│                                                                             │
│       ┌───────┐                                                             │
│  A =  │ 2  1 │                                                             │
│       │ 1  2 │                                                             │
│       └───────┘                                                             │
│                                                                             │
│  Eigenvalue λ₁ = 3, Eigenvector v₁ = [1, 1]ᵀ                               │
│  Eigenvalue λ₂ = 1, Eigenvector v₂ = [1, -1]ᵀ                              │
│                                                                             │
│  Check: Av₁ = [2+1, 1+2]ᵀ = [3, 3]ᵀ = 3[1, 1]ᵀ = λ₁v₁ ✓                   │
│                                                                             │
│  WHERE USED IN DL                                                           │
│  ────────────────                                                           │
│                                                                             │
│  • PCA: Eigenvectors of covariance matrix = principal components            │
│  • Spectral normalization: Constrain largest eigenvalue of W                │
│  • Understanding optimization: Eigenvalues of Hessian affect convergence    │
│  • Graph neural networks: Eigenvalues of graph Laplacian                    │
│                                                                             │
│  SPECTRAL DECOMPOSITION                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│  Symmetric matrix A can be written as:                                      │
│                                                                             │
│  A = QΛQᵀ                                                                  │
│                                                                             │
│  Where:                                                                     │
│  • Q = matrix of eigenvectors (columns)                                     │
│  • Λ = diagonal matrix of eigenvalues                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.7 Broadcasting

How NumPy/PyTorch handle operations on arrays of different shapes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BROADCASTING                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Broadcasting lets you do element-wise operations on arrays of              │
│  different shapes by "stretching" smaller arrays.                           │
│                                                                             │
│  RULES                                                                      │
│  ─────                                                                      │
│                                                                             │
│  1. Align shapes from the right                                             │
│  2. Dimensions are compatible if they're equal OR one of them is 1          │
│  3. Missing dimensions are treated as 1                                     │
│                                                                             │
│  EXAMPLE 1: Vector + Scalar                                                 │
│  ─────────────────────────────                                              │
│                                                                             │
│  [1, 2, 3] + 5 = [1, 2, 3] + [5, 5, 5] = [6, 7, 8]                         │
│    (3,)     ()     (3,)        (3,)        (3,)                             │
│                                                                             │
│  EXAMPLE 2: Matrix + Vector                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  ┌─────────┐       ┌─────────┐   ┌─────────────┐                            │
│  │ 1  2  3 │ + [10, 20, 30] = │ 11  22  33 │                               │
│  │ 4  5  6 │       ↓          │ 14  25  36 │                               │
│  └─────────┘   ┌─────────────┐ └─────────────┘                              │
│    (2, 3)      │ 10  20  30 │                                               │
│                │ 10  20  30 │   Vector broadcast to each row                │
│                └─────────────┘                                              │
│                    (2, 3)                                                   │
│                                                                             │
│  EXAMPLE 3: Adding Bias in Neural Networks                                  │
│  ─────────────────────────────────────────                                  │
│                                                                             │
│  Batch output: (batch_size, features) = (32, 256)                           │
│  Bias vector:  (features,)            = (256,)                              │
│                                                                             │
│  output + bias: Each row gets the same bias added                           │
│                                                                             │
│  ┌───────────────────┐         ┌───────────────────┐                        │
│  │ sample 1 features │         │ sample 1 + bias   │                        │
│  │ sample 2 features │ + [b] = │ sample 2 + bias   │                        │
│  │ ...               │         │ ...               │                        │
│  │ sample 32 features│         │ sample 32 + bias  │                        │
│  └───────────────────┘         └───────────────────┘                        │
│      (32, 256)        (256,)       (32, 256)                                │
│                                                                             │
│  SHAPE COMPATIBILITY TABLE                                                  │
│  ─────────────────────────                                                  │
│                                                                             │
│  Shape A    Shape B    Compatible?   Result Shape                           │
│  ─────────────────────────────────────────────────                          │
│  (3,)       (3,)       ✓             (3,)                                   │
│  (3,)       (1,)       ✓             (3,)                                   │
│  (3,)       (4,)       ✗             Error!                                 │
│  (2, 3)     (3,)       ✓             (2, 3)                                 │
│  (2, 3)     (2, 1)     ✓             (2, 3)                                 │
│  (2, 3)     (1, 3)     ✓             (2, 3)                                 │
│  (2, 3)     (2, 3)     ✓             (2, 3)                                 │
│  (4, 1, 3)  (2, 3)     ✓             (4, 2, 3)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Calculus

### 2.1 Derivatives

Derivatives measure how much a function's output changes when input changes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DERIVATIVES                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEFINITION                                                                 │
│  ──────────                                                                 │
│                                                                             │
│              f(x + h) - f(x)                                                │
│  f'(x) = lim ───────────────                                               │
│          h→0       h                                                        │
│                                                                             │
│  GEOMETRIC MEANING: Slope of tangent line at point x                        │
│                                                                             │
│       f(x)                                                                  │
│        │     ╱                                                              │
│        │   ╱   ← tangent line (slope = f'(x))                              │
│        │ ╱●───────                                                          │
│        │╱  ╲                                                                │
│        └──────────→ x                                                       │
│                                                                             │
│  COMMON DERIVATIVES                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  Function f(x)       Derivative f'(x)       Example                         │
│  ───────────────────────────────────────────────────                        │
│  c (constant)        0                      d/dx[5] = 0                     │
│  x                   1                      d/dx[x] = 1                     │
│  xⁿ                  nxⁿ⁻¹                  d/dx[x²] = 2x                   │
│  eˣ                  eˣ                     d/dx[eˣ] = eˣ                   │
│  ln(x)               1/x                    d/dx[ln(x)] = 1/x               │
│  sin(x)              cos(x)                                                 │
│  cos(x)              -sin(x)                                                │
│                                                                             │
│  DERIVATIVE RULES                                                           │
│  ────────────────                                                           │
│                                                                             │
│  Sum rule:      (f + g)' = f' + g'                                         │
│  Product rule:  (fg)' = f'g + fg'                                          │
│  Quotient rule: (f/g)' = (f'g - fg')/g²                                    │
│  Chain rule:    (f(g(x)))' = f'(g(x)) · g'(x)                              │
│                                                                             │
│  DEEP LEARNING DERIVATIVES                                                  │
│  ─────────────────────────                                                  │
│                                                                             │
│  ReLU: f(x) = max(0, x)                                                     │
│        f'(x) = { 1 if x > 0                                                │
│                { 0 if x < 0                                                │
│                { undefined at x = 0 (use 0 or 1)                           │
│                                                                             │
│  Sigmoid: σ(x) = 1/(1 + e⁻ˣ)                                               │
│           σ'(x) = σ(x)(1 - σ(x))   ← elegant form!                         │
│                                                                             │
│  Tanh: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)                                    │
│        tanh'(x) = 1 - tanh²(x)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Chain Rule

The most important rule for deep learning - it's how backpropagation works!

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE CHAIN RULE                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  If y = f(g(x)), then:                                                      │
│                                                                             │
│      dy     dy   dg                                                         │
│      ── = ── · ──                                                          │
│      dx    dg   dx                                                          │
│                                                                             │
│  Or: (f ∘ g)'(x) = f'(g(x)) · g'(x)                                        │
│                                                                             │
│  INTUITION                                                                  │
│  ─────────                                                                  │
│                                                                             │
│  If g doubles its input, and f triples its input:                           │
│  • g(x) = 2x, so g'(x) = 2                                                 │
│  • f(u) = 3u, so f'(u) = 3                                                 │
│  • f(g(x)) = f(2x) = 3(2x) = 6x                                            │
│  • (f ∘ g)'(x) = f'(g(x)) · g'(x) = 3 · 2 = 6 ✓                           │
│                                                                             │
│  COMPUTATIONAL GRAPH VIEW                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│    x ────→ [g] ────→ u=g(x) ────→ [f] ────→ y=f(u)                        │
│                                                                             │
│    ∂y/∂x = ∂y/∂u · ∂u/∂x                                                   │
│          = f'(u) · g'(x)                                                   │
│                                                                             │
│  EXTENDED CHAIN RULE                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  For y = f(g(h(x))):                                                        │
│                                                                             │
│    dy     df   dg   dh                                                      │
│    ── = ── · ── · ──                                                       │
│    dx    dg   dh   dx                                                       │
│                                                                             │
│  EXAMPLE: Neural Network Forward Pass                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│    x ───→ [Linear: z=Wx+b] ───→ [ReLU: a=max(0,z)] ───→ [Loss: L]          │
│                                                                             │
│  Backward (chain rule):                                                     │
│                                                                             │
│    ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W                                          │
│                                                                             │
│    Where:                                                                   │
│    • ∂L/∂a comes from loss function                                        │
│    • ∂a/∂z = 1 if z > 0, else 0 (ReLU derivative)                         │
│    • ∂z/∂W = x (since z = Wx + b)                                         │
│                                                                             │
│  THIS IS BACKPROPAGATION!                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  Forward: Compute all intermediate values                                   │
│  Backward: Apply chain rule, multiply gradients backward                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Partial Derivatives and Gradients

When functions have multiple inputs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARTIAL DERIVATIVES AND GRADIENTS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PARTIAL DERIVATIVE                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  For f(x, y), the partial derivative ∂f/∂x means:                          │
│  "How does f change when x changes, holding y constant?"                    │
│                                                                             │
│  Example: f(x, y) = x² + 3xy + y²                                          │
│                                                                             │
│  ∂f/∂x = 2x + 3y    (treat y as constant)                                  │
│  ∂f/∂y = 3x + 2y    (treat x as constant)                                  │
│                                                                             │
│  GRADIENT                                                                   │
│  ────────                                                                   │
│                                                                             │
│  The gradient ∇f is a vector of all partial derivatives:                   │
│                                                                             │
│           ┌────────┐                                                        │
│           │ ∂f/∂x₁ │                                                       │
│           │ ∂f/∂x₂ │                                                       │
│  ∇f(x) = │   ⋮    │                                                       │
│           │ ∂f/∂xₙ │                                                       │
│           └────────┘                                                        │
│                                                                             │
│  KEY PROPERTY: Gradient points in direction of steepest increase            │
│                                                                             │
│            ∇f                                                               │
│            ↑                                                                │
│       ╱────────╲                                                            │
│      ╱  ●────→  ╲   The gradient at ● points "uphill"                      │
│     ╱           ╲                                                           │
│    ╱─────────────╲                                                          │
│                                                                             │
│  GRADIENT DESCENT                                                           │
│  ────────────────                                                           │
│                                                                             │
│  To minimize f, go in opposite direction of gradient:                       │
│                                                                             │
│  x_new = x_old - α · ∇f(x_old)                                             │
│                                                                             │
│  Where α is the learning rate.                                              │
│                                                                             │
│  EXAMPLE: Linear Regression Loss                                            │
│  ────────────────────────────────                                           │
│                                                                             │
│  Loss L(w) = (1/n) Σᵢ (yᵢ - wᵀxᵢ)²                                        │
│                                                                             │
│  Gradient: ∇L = (2/n) Σᵢ (wᵀxᵢ - yᵢ) · xᵢ                                │
│                                                                             │
│  Update: w = w - α · ∇L                                                    │
│                                                                             │
│  GRADIENT OF NEURAL NETWORK                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  Parameters: θ = {W₁, b₁, W₂, b₂, ...}                                     │
│  Loss: L(θ)                                                                 │
│                                                                             │
│           ┌─────────┐                                                       │
│           │ ∂L/∂W₁  │                                                       │
│           │ ∂L/∂b₁  │                                                       │
│  ∇L(θ) = │ ∂L/∂W₂  │   ← This is what backprop computes!                  │
│           │ ∂L/∂b₂  │                                                       │
│           │   ⋮     │                                                       │
│           └─────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Jacobians

For functions with multiple inputs AND multiple outputs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    JACOBIAN MATRIX                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For f: ℝⁿ → ℝᵐ (n inputs, m outputs):                                     │
│                                                                             │
│  f(x) = [f₁(x), f₂(x), ..., fₘ(x)]ᵀ                                        │
│                                                                             │
│  The Jacobian J is an m×n matrix:                                           │
│                                                                             │
│         ┌─────────────────────────────────┐                                 │
│         │ ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ │                                │
│         │ ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ │                                │
│  J_f = │    ⋮        ⋮      ⋱      ⋮     │                                │
│         │ ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ │                                │
│         └─────────────────────────────────┘                                 │
│                                                                             │
│  J[i,j] = ∂fᵢ/∂xⱼ                                                          │
│                                                                             │
│  EXAMPLE                                                                    │
│  ───────                                                                    │
│                                                                             │
│  f(x, y) = [x² + y, xy]ᵀ                                                   │
│                                                                             │
│  f₁ = x² + y     f₂ = xy                                                   │
│                                                                             │
│       ┌─────────────────┐     ┌──────────┐                                  │
│  J = │ ∂f₁/∂x  ∂f₁/∂y │ = │ 2x    1  │                                   │
│       │ ∂f₂/∂x  ∂f₂/∂y │   │  y    x  │                                   │
│       └─────────────────┘     └──────────┘                                  │
│                                                                             │
│  At point (2, 3):                                                           │
│       ┌──────────┐                                                          │
│  J = │ 4    1  │                                                           │
│       │ 3    2  │                                                           │
│       └──────────┘                                                          │
│                                                                             │
│  CHAIN RULE WITH JACOBIANS                                                  │
│  ─────────────────────────                                                  │
│                                                                             │
│  If h = g(f(x)), then:                                                      │
│                                                                             │
│  J_h = J_g · J_f   (matrix multiplication!)                                │
│                                                                             │
│  This is how gradients flow through layers in a neural network.             │
│                                                                             │
│  NEURAL NETWORK EXAMPLE                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│    x ∈ ℝ³ ───→ [Layer 1] ───→ h ∈ ℝ⁴ ───→ [Layer 2] ───→ y ∈ ℝ²          │
│                 W₁: 4×3                     W₂: 2×4                         │
│                                                                             │
│  Jacobian of Layer 1 (linear): J₁ = W₁  (4×3)                              │
│  Jacobian of Layer 2 (linear): J₂ = W₂  (2×4)                              │
│                                                                             │
│  Total Jacobian: J = J₂ · J₁ = W₂ · W₁  (2×3)                              │
│                                                                             │
│  For gradient of loss w.r.t. x:                                             │
│  ∂L/∂x = (∂L/∂y)ᵀ · J = (∂L/∂y)ᵀ · W₂ · W₁                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Probability

### 3.1 Probability Basics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROBABILITY FUNDAMENTALS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BASIC RULES                                                                │
│  ───────────                                                                │
│                                                                             │
│  • 0 ≤ P(A) ≤ 1                                                            │
│  • P(Ω) = 1  (something must happen)                                        │
│  • P(not A) = 1 - P(A)                                                      │
│  • P(A or B) = P(A) + P(B) - P(A and B)                                    │
│                                                                             │
│  CONDITIONAL PROBABILITY                                                    │
│  ───────────────────────                                                    │
│                                                                             │
│                P(A and B)                                                   │
│  P(A|B) = ─────────────    "Probability of A given B"                      │
│               P(B)                                                          │
│                                                                             │
│  Example: P(rain | cloudy) = P(rain and cloudy) / P(cloudy)                │
│                                                                             │
│  BAYES' THEOREM                                                             │
│  ──────────────                                                             │
│                                                                             │
│              P(B|A) · P(A)                                                  │
│  P(A|B) = ─────────────────                                                │
│                 P(B)                                                        │
│                                                                             │
│  In ML terms:                                                               │
│                                                                             │
│  posterior = (likelihood × prior) / evidence                                │
│                                                                             │
│              P(data|model) · P(model)                                       │
│  P(model|data) = ─────────────────────                                     │
│                      P(data)                                                │
│                                                                             │
│  INDEPENDENCE                                                               │
│  ────────────                                                               │
│                                                                             │
│  A and B are independent if:                                                │
│  P(A and B) = P(A) · P(B)                                                  │
│                                                                             │
│  Equivalently: P(A|B) = P(A)  (knowing B doesn't change A)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Probability Distributions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROBABILITY DISTRIBUTIONS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DISCRETE DISTRIBUTIONS                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│  Probability Mass Function (PMF): P(X = x)                                  │
│                                                                             │
│  Bernoulli (coin flip):                                                     │
│  P(X = 1) = p,  P(X = 0) = 1 - p                                           │
│                                                                             │
│  Categorical (dice, classification):                                        │
│  P(X = k) = pₖ,  where Σₖ pₖ = 1                                           │
│                                                                             │
│  This is what softmax outputs!                                              │
│                                                                             │
│  CONTINUOUS DISTRIBUTIONS                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  Probability Density Function (PDF): p(x)                                   │
│  P(a ≤ X ≤ b) = ∫ₐᵇ p(x) dx                                                │
│                                                                             │
│  Gaussian (Normal): N(μ, σ²)                                               │
│                                                                             │
│                    1              (x - μ)²                                  │
│  p(x) = ───────────────── exp(- ─────────)                                 │
│          σ√(2π)                   2σ²                                       │
│                                                                             │
│        │                                                                    │
│        │      ╱╲                                                            │
│        │     ╱  ╲                                                           │
│        │    ╱    ╲                                                          │
│        │  ╱        ╲                                                        │
│        │╱            ╲                                                      │
│        └──────┬───────→                                                     │
│               μ                                                             │
│                                                                             │
│  • μ = mean (center)                                                        │
│  • σ² = variance (spread)                                                   │
│  • σ = standard deviation                                                   │
│                                                                             │
│  UNIFORM DISTRIBUTION: U(a, b)                                              │
│                                                                             │
│  p(x) = 1/(b-a) for x ∈ [a, b], else 0                                     │
│                                                                             │
│  Used for random initialization                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Expectation and Variance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPECTATION AND VARIANCE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EXPECTATION (Mean)                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  Discrete:  E[X] = Σₓ x · P(X = x)                                         │
│  Continuous: E[X] = ∫ x · p(x) dx                                          │
│                                                                             │
│  Properties:                                                                │
│  • E[aX + b] = aE[X] + b  (linearity)                                      │
│  • E[X + Y] = E[X] + E[Y]                                                  │
│  • E[XY] = E[X]E[Y]  (if X, Y independent)                                 │
│                                                                             │
│  Example: E[dice] = 1·(1/6) + 2·(1/6) + ... + 6·(1/6) = 3.5               │
│                                                                             │
│  VARIANCE                                                                   │
│  ────────                                                                   │
│                                                                             │
│  Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²                                 │
│                                                                             │
│  "Average squared deviation from the mean"                                  │
│                                                                             │
│  Properties:                                                                │
│  • Var(aX + b) = a²Var(X)                                                  │
│  • Var(X + Y) = Var(X) + Var(Y)  (if independent)                          │
│                                                                             │
│  Standard deviation: σ = √Var(X)                                           │
│                                                                             │
│  COVARIANCE                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  Cov(X, Y) = E[(X - E[X])(Y - E[Y])]                                       │
│            = E[XY] - E[X]E[Y]                                               │
│                                                                             │
│  • Cov > 0: X and Y tend to increase together                               │
│  • Cov < 0: When X increases, Y tends to decrease                           │
│  • Cov = 0: X and Y are uncorrelated                                        │
│                                                                             │
│  WHY IT MATTERS IN DL                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  • Loss is an expectation: L = E[loss(model, data)]                        │
│  • Batch normalization uses mean and variance                               │
│  • VAEs model latent variables with Gaussian (μ, σ²)                       │
│  • Gradient variance affects optimization stability                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Maximum Likelihood Estimation (MLE)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MAXIMUM LIKELIHOOD ESTIMATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  IDEA                                                                       │
│  ────                                                                       │
│                                                                             │
│  Given data D, find parameters θ that make D most likely:                   │
│                                                                             │
│  θ* = argmax P(D | θ)                                                      │
│          θ                                                                  │
│                                                                             │
│  LIKELIHOOD                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  For independent data points {x₁, x₂, ..., xₙ}:                            │
│                                                                             │
│  L(θ) = P(D|θ) = ∏ᵢ P(xᵢ|θ)                                               │
│                                                                             │
│  LOG-LIKELIHOOD (more practical)                                            │
│  ───────────────────────────────                                            │
│                                                                             │
│  log L(θ) = Σᵢ log P(xᵢ|θ)                                                 │
│                                                                             │
│  • Products become sums (easier math)                                       │
│  • Avoids numerical underflow                                               │
│  • Maximizing log L is same as maximizing L                                 │
│                                                                             │
│  NEGATIVE LOG-LIKELIHOOD (NLL) = LOSS                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  NLL = -log L(θ) = -Σᵢ log P(xᵢ|θ)                                        │
│                                                                             │
│  Minimizing NLL = Maximizing likelihood                                     │
│                                                                             │
│  EXAMPLE: Binary Classification                                             │
│  ───────────────────────────────                                            │
│                                                                             │
│  Model predicts: p = P(y=1|x)                                               │
│  True label: y ∈ {0, 1}                                                    │
│                                                                             │
│  P(y|x) = pʸ(1-p)¹⁻ʸ                                                       │
│                                                                             │
│  NLL = -[y log p + (1-y) log(1-p)]                                         │
│                                                                             │
│  This is Binary Cross-Entropy Loss!                                         │
│                                                                             │
│  EXAMPLE: Multi-class Classification                                        │
│  ───────────────────────────────────                                        │
│                                                                             │
│  Model predicts: p = softmax(logits)                                        │
│  True label: y (one-hot vector)                                             │
│                                                                             │
│  NLL = -Σₖ yₖ log pₖ                                                       │
│                                                                             │
│  This is Cross-Entropy Loss!                                                │
│                                                                             │
│  KEY INSIGHT: Most classification losses are NLL in disguise!               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Information Theory

### 4.1 Entropy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTROPY                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEFINITION                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  Entropy measures "uncertainty" or "information content":                   │
│                                                                             │
│  H(X) = -Σₓ P(x) log P(x) = E[-log P(X)]                                   │
│                                                                             │
│  (Usually use log base 2 for "bits" or natural log for "nats")              │
│                                                                             │
│  INTUITION                                                                  │
│  ─────────                                                                  │
│                                                                             │
│  Low entropy = Predictable (less surprise)                                  │
│  High entropy = Unpredictable (more surprise)                               │
│                                                                             │
│  Example: Coin flip                                                         │
│                                                                             │
│  Fair coin (p=0.5):      H = -0.5 log 0.5 - 0.5 log 0.5 = 1 bit            │
│  Biased coin (p=0.9):    H = -0.9 log 0.9 - 0.1 log 0.1 ≈ 0.47 bits        │
│  Certain (p=1.0):        H = -1 log 1 = 0 bits                              │
│                                                                             │
│         H                                                                   │
│         │                                                                   │
│       1 │    ●───────────────●                                              │
│         │   ╱                 ╲                                             │
│         │  ╱                   ╲                                            │
│         │ ╱                     ╲                                           │
│       0 ●───────────────────────●                                           │
│         0        0.5            1    p                                      │
│                                                                             │
│  Maximum entropy when uniform (most uncertain)                              │
│                                                                             │
│  MAXIMUM ENTROPY                                                            │
│  ───────────────                                                            │
│                                                                             │
│  For K classes:  H_max = log K  (when uniform distribution)                 │
│                                                                             │
│  Example: 10 classes, uniform → H = log₂(10) ≈ 3.32 bits                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Cross-Entropy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-ENTROPY                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEFINITION                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  Cross-entropy between distributions P and Q:                               │
│                                                                             │
│  H(P, Q) = -Σₓ P(x) log Q(x) = E_P[-log Q(X)]                              │
│                                                                             │
│  "Expected surprise when using Q to encode samples from P"                  │
│                                                                             │
│  CLASSIFICATION LOSS                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  • P = true distribution (one-hot: all mass on correct class)               │
│  • Q = predicted distribution (softmax output)                              │
│                                                                             │
│  H(P, Q) = -Σₖ P(k) log Q(k)                                               │
│                                                                             │
│  If true class is c (P(c)=1, others=0):                                    │
│  H(P, Q) = -log Q(c)                                                       │
│                                                                             │
│  This is exactly cross-entropy loss!                                        │
│                                                                             │
│  EXAMPLE                                                                    │
│  ───────                                                                    │
│                                                                             │
│  True label: class 2 (one-hot: [0, 0, 1])                                  │
│  Prediction: [0.1, 0.2, 0.7]                                               │
│                                                                             │
│  Loss = -log(0.7) ≈ 0.36                                                   │
│                                                                             │
│  If prediction were [0.1, 0.1, 0.8]:                                       │
│  Loss = -log(0.8) ≈ 0.22  (better prediction → lower loss)                 │
│                                                                             │
│  If prediction were [0.1, 0.1, 0.01]:                                      │
│  Loss = -log(0.01) ≈ 4.6  (confident and wrong → high loss!)               │
│                                                                             │
│  RELATIONSHIP                                                               │
│  ────────────                                                               │
│                                                                             │
│  H(P, Q) = H(P) + D_KL(P || Q)                                             │
│                                                                             │
│  Cross-entropy = Entropy + KL Divergence                                    │
│                                                                             │
│  Since H(P) is constant (true distribution fixed),                          │
│  minimizing cross-entropy = minimizing KL divergence                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 KL Divergence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KL DIVERGENCE                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEFINITION                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  Kullback-Leibler divergence measures "distance" between distributions:     │
│                                                                             │
│                         P(x)                                                │
│  D_KL(P || Q) = Σₓ P(x) log ────                                           │
│                         Q(x)                                                │
│                                                                             │
│              = E_P[log P(X) - log Q(X)]                                    │
│                                                                             │
│  PROPERTIES                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  • D_KL(P || Q) ≥ 0  (always non-negative)                                 │
│  • D_KL(P || Q) = 0  iff P = Q                                             │
│  • NOT symmetric: D_KL(P || Q) ≠ D_KL(Q || P)                              │
│                                                                             │
│  ASYMMETRY                                                                  │
│  ─────────                                                                  │
│                                                                             │
│  D_KL(P || Q): "How bad is Q as an approximation to P?"                    │
│                                                                             │
│  Forward KL: D_KL(P_true || Q_model)                                       │
│  • Penalizes Q putting low probability where P has high probability         │
│  • Leads to "mean-seeking" behavior                                         │
│                                                                             │
│  Reverse KL: D_KL(Q_model || P_true)                                       │
│  • Penalizes Q putting high probability where P has low probability         │
│  • Leads to "mode-seeking" behavior                                         │
│                                                                             │
│  USE IN VAEs                                                                │
│  ──────────                                                                 │
│                                                                             │
│  VAE loss = Reconstruction loss + β · D_KL(q(z|x) || p(z))                 │
│                                                                             │
│  The KL term regularizes the latent space to be close to prior N(0, I):    │
│                                                                             │
│  D_KL(N(μ, σ²) || N(0, 1)) = ½(μ² + σ² - log σ² - 1)                      │
│                                                                             │
│  This has a closed-form solution for Gaussians!                             │
│                                                                             │
│  USE IN RLHF                                                                │
│  ──────────                                                                 │
│                                                                             │
│  RLHF adds KL penalty to prevent model from deviating too far from base:   │
│                                                                             │
│  Reward_final = Reward - β · D_KL(π_new || π_base)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Softmax and Cross-Entropy (Putting It Together)

This is the most important application of math in classification.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SOFTMAX + CROSS-ENTROPY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SOFTMAX FUNCTION                                                           │
│  ────────────────                                                           │
│                                                                             │
│  Converts logits (raw scores) to probabilities:                             │
│                                                                             │
│                    exp(zᵢ)                                                  │
│  softmax(z)ᵢ = ─────────────                                               │
│                 Σⱼ exp(zⱼ)                                                  │
│                                                                             │
│  Properties:                                                                │
│  • Output is always positive                                                │
│  • Outputs sum to 1 (valid probability distribution)                        │
│  • Preserves ordering (larger logit → larger probability)                   │
│  • Differentiable                                                           │
│                                                                             │
│  EXAMPLE                                                                    │
│  ───────                                                                    │
│                                                                             │
│  Logits: z = [2.0, 1.0, 0.1]                                               │
│                                                                             │
│  exp(z) = [e², e¹, e⁰·¹] = [7.39, 2.72, 1.11]                             │
│  sum = 7.39 + 2.72 + 1.11 = 11.22                                          │
│                                                                             │
│  softmax(z) = [7.39/11.22, 2.72/11.22, 1.11/11.22]                        │
│             = [0.659, 0.242, 0.099]                                        │
│                                                                             │
│  NUMERICAL STABILITY TRICK                                                  │
│  ─────────────────────────                                                  │
│                                                                             │
│  Problem: exp(1000) = overflow!                                             │
│                                                                             │
│  Solution: Subtract max before exp                                          │
│                                                                             │
│  softmax(z)ᵢ = exp(zᵢ - max(z)) / Σⱼ exp(zⱼ - max(z))                     │
│                                                                             │
│  This doesn't change the result (cancels out) but prevents overflow.        │
│                                                                             │
│  CROSS-ENTROPY LOSS                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  For true class y (one-hot) and predictions p = softmax(z):                 │
│                                                                             │
│  L = -Σₖ yₖ log pₖ = -log p_y  (only true class contributes)               │
│                                                                             │
│  GRADIENT OF SOFTMAX + CROSS-ENTROPY                                        │
│  ───────────────────────────────────                                        │
│                                                                             │
│  The gradient has a beautifully simple form:                                │
│                                                                             │
│  ∂L/∂zᵢ = pᵢ - yᵢ                                                         │
│                                                                             │
│  For true class c:  ∂L/∂zc = p_c - 1  (push down if p_c < 1)               │
│  For other classes: ∂L/∂zᵢ = pᵢ      (push down proportional to p)         │
│                                                                             │
│  Example:                                                                   │
│  True label: class 0                                                        │
│  Predictions: [0.7, 0.2, 0.1]                                              │
│  Gradient: [0.7-1, 0.2-0, 0.1-0] = [-0.3, 0.2, 0.1]                        │
│                                                                             │
│  This says:                                                                 │
│  • Increase logit for class 0 (true class)                                 │
│  • Decrease logits for classes 1 and 2                                     │
│                                                                             │
│  IMPLEMENTATION                                                             │
│  ──────────────                                                             │
│                                                                             │
│  ```python                                                                  │
│  import numpy as np                                                         │
│                                                                             │
│  def softmax(z):                                                            │
│      z = z - np.max(z, axis=-1, keepdims=True)  # stability                 │
│      exp_z = np.exp(z)                                                      │
│      return exp_z / np.sum(exp_z, axis=-1, keepdims=True)                   │
│                                                                             │
│  def cross_entropy_loss(logits, labels):                                    │
│      probs = softmax(logits)                                                │
│      return -np.log(probs[range(len(labels)), labels]).mean()               │
│                                                                             │
│  def softmax_cross_entropy_gradient(logits, labels):                        │
│      probs = softmax(logits)                                                │
│      grad = probs.copy()                                                    │
│      grad[range(len(labels)), labels] -= 1                                  │
│      return grad / len(labels)                                              │
│  ```                                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Module-Specific Math

### Math for Module 1-2 (Neural Network Foundations)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH FOR MODULES 1-2                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KEY CONCEPTS NEEDED                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  1. Matrix-vector multiplication (for linear layers)                        │
│     y = Wx + b                                                              │
│                                                                             │
│  2. Derivatives and chain rule (for backpropagation)                        │
│     ∂L/∂W = ∂L/∂y · ∂y/∂W                                                  │
│                                                                             │
│  3. Common derivatives:                                                     │
│     • ReLU:    f'(x) = 1 if x > 0, else 0                                  │
│     • Sigmoid: σ'(x) = σ(x)(1 - σ(x))                                      │
│     • Tanh:    tanh'(x) = 1 - tanh²(x)                                     │
│                                                                             │
│  4. Loss function gradients:                                                │
│     • MSE: ∂L/∂ŷ = 2(ŷ - y)                                                │
│     • Cross-entropy: ∂L/∂z = softmax(z) - y                                │
│                                                                             │
│  BACKPROP EXAMPLE                                                           │
│  ────────────────                                                           │
│                                                                             │
│  Forward:                                                                   │
│  z₁ = W₁x + b₁                                                             │
│  a₁ = ReLU(z₁)                                                             │
│  z₂ = W₂a₁ + b₂                                                            │
│  ŷ = softmax(z₂)                                                           │
│  L = CrossEntropy(ŷ, y)                                                    │
│                                                                             │
│  Backward (chain rule):                                                     │
│  ∂L/∂z₂ = ŷ - y                              (softmax + CE gradient)       │
│  ∂L/∂W₂ = ∂L/∂z₂ · a₁ᵀ                       (gradient of matmul)         │
│  ∂L/∂a₁ = W₂ᵀ · ∂L/∂z₂                       (backprop through matmul)     │
│  ∂L/∂z₁ = ∂L/∂a₁ ⊙ ReLU'(z₁)                (element-wise, ReLU grad)     │
│  ∂L/∂W₁ = ∂L/∂z₁ · xᵀ                        (gradient of first matmul)   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Math for Module 3-4 (CNNs)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH FOR MODULES 3-4 (CNNs)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CONVOLUTION OPERATION                                                      │
│  ─────────────────────                                                      │
│                                                                             │
│  2D convolution (cross-correlation, actually):                              │
│                                                                             │
│  (I * K)[i,j] = Σₘ Σₙ I[i+m, j+n] · K[m, n]                                │
│                                                                             │
│  Visual:                                                                    │
│  ┌───────────┐     ┌─────┐     ┌─────┐                                      │
│  │ 1 2 3 4 5 │     │ 1 0 │     │     │                                      │
│  │ 6 7 8 9 0 │  *  │ 0 1 │  =  │     │  output[0,0] = 1×1+2×0+6×0+7×1 = 8  │
│  │ 1 2 3 4 5 │     └─────┘     │     │                                      │
│  │ 6 7 8 9 0 │                 └─────┘                                      │
│  └───────────┘                                                              │
│     Input       Kernel        Output                                        │
│                                                                             │
│  OUTPUT SIZE                                                                │
│  ───────────                                                                │
│                                                                             │
│  output_size = (input_size - kernel_size + 2×padding) / stride + 1         │
│                                                                             │
│  Example: 32×32 input, 3×3 kernel, padding=1, stride=1                     │
│  output = (32 - 3 + 2×1) / 1 + 1 = 32×32                                   │
│                                                                             │
│  POOLING                                                                    │
│  ───────                                                                    │
│                                                                             │
│  Max pooling: Take maximum in each window                                   │
│  Avg pooling: Take average in each window                                   │
│                                                                             │
│  ┌───────┐              ┌───┐                                               │
│  │ 1 3 │ 2 4 │   2×2    │ 3 │ 4 │                                          │
│  │ 5 2 │ 1 6 │   max    │ 5 │ 6 │                                          │
│  ├─────┼─────┤  pool →  └───┴───┘                                          │
│  │ 7 1 │ 8 2 │                                                              │
│  │ 3 4 │ 9 1 │          │ 7 │ 9 │                                          │
│  └─────┴─────┘          └───┴───┘                                          │
│                                                                             │
│  RECEPTIVE FIELD                                                            │
│  ───────────────                                                            │
│                                                                             │
│  After L layers of 3×3 convolutions:                                        │
│  receptive_field = 1 + L × 2                                                │
│                                                                             │
│  With pooling/stride, grows faster.                                         │
│                                                                             │
│  CONVOLUTION GRADIENT                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  ∂L/∂K = input * ∂L/∂output  (convolution of input with output gradient)   │
│  ∂L/∂input = ∂L/∂output * K  (full convolution with flipped kernel)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Math for Module 5 (RNNs)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH FOR MODULE 5 (RNNs)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RNN EQUATIONS                                                              │
│  ─────────────                                                              │
│                                                                             │
│  hₜ = tanh(Wₕₕhₜ₋₁ + Wₓₕxₜ + b)                                           │
│  yₜ = Wₕᵧhₜ                                                                │
│                                                                             │
│  BACKPROPAGATION THROUGH TIME (BPTT)                                        │
│  ───────────────────────────────────                                        │
│                                                                             │
│  Gradient flows backward through time steps:                                │
│                                                                             │
│  ∂L/∂Wₕₕ = Σₜ ∂L/∂hₜ · ∂hₜ/∂Wₕₕ                                          │
│                                                                             │
│  ∂hₜ/∂hₜ₋₁ = Wₕₕᵀ · diag(1 - hₜ²)  (tanh derivative)                      │
│                                                                             │
│  VANISHING GRADIENT PROBLEM                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  ∂hₜ/∂h₀ = ∏ₛ₌₁ᵗ ∂hₛ/∂hₛ₋₁                                               │
│                                                                             │
│  If |∂hₛ/∂hₛ₋₁| < 1 consistently, product → 0 (vanishing)                  │
│  If |∂hₛ/∂hₛ₋₁| > 1 consistently, product → ∞ (exploding)                  │
│                                                                             │
│  LSTM EQUATIONS                                                             │
│  ──────────────                                                             │
│                                                                             │
│  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)         Forget gate                           │
│  iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)         Input gate                            │
│  c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)     Candidate cell state                  │
│  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ          Cell state update                     │
│  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)         Output gate                           │
│  hₜ = oₜ ⊙ tanh(cₜ)                  Hidden state                          │
│                                                                             │
│  KEY INSIGHT: Cell state cₜ has additive updates, avoiding vanishing       │
│  gradients (similar to ResNet skip connections)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Math for Module 6 (Attention and Transformers)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH FOR MODULE 6 (ATTENTION/TRANSFORMERS)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SCALED DOT-PRODUCT ATTENTION                                               │
│  ────────────────────────────                                               │
│                                                                             │
│                            QKᵀ                                              │
│  Attention(Q, K, V) = softmax(────) V                                      │
│                            √dₖ                                              │
│                                                                             │
│  Where:                                                                     │
│  • Q: Queries (seq_len × d_k)                                              │
│  • K: Keys (seq_len × d_k)                                                 │
│  • V: Values (seq_len × d_v)                                               │
│  • dₖ: Key dimension (scaling factor)                                      │
│                                                                             │
│  WHY SCALE BY √dₖ?                                                         │
│  ─────────────────                                                          │
│                                                                             │
│  Dot products grow with dimension. If q, k ~ N(0, 1):                       │
│  E[q·k] = 0                                                                 │
│  Var[q·k] = dₖ                                                             │
│                                                                             │
│  Large variance → softmax becomes very peaked (near one-hot)                │
│  → Gradients become very small                                              │
│                                                                             │
│  Dividing by √dₖ normalizes variance to ~1                                 │
│                                                                             │
│  MULTI-HEAD ATTENTION                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ                        │
│                                                                             │
│  headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)                                      │
│                                                                             │
│  • h heads, each with dimension d_k/h                                       │
│  • Allows attending to different "aspects"                                  │
│  • Same computation cost as single-head with full dimension                 │
│                                                                             │
│  POSITIONAL ENCODING                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  Attention is permutation-invariant, so we add position info:               │
│                                                                             │
│  PE(pos, 2i) = sin(pos / 10000^(2i/d))                                     │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))                                   │
│                                                                             │
│  Properties:                                                                │
│  • Each position gets unique encoding                                       │
│  • PE(pos+k) can be represented as linear function of PE(pos)               │
│  • Bounded values [-1, 1]                                                   │
│                                                                             │
│  SELF-ATTENTION COMPLEXITY                                                  │
│  ─────────────────────────                                                  │
│                                                                             │
│  For sequence length n:                                                     │
│  • Compute QKᵀ: O(n² · d)                                                  │
│  • Memory for attention matrix: O(n²)                                      │
│                                                                             │
│  This is why long sequences are expensive!                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Math for Module 7 (Generative Models)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH FOR MODULE 7 (GENERATIVE MODELS)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VAE: EVIDENCE LOWER BOUND (ELBO)                                           │
│  ────────────────────────────────                                           │
│                                                                             │
│  Goal: Maximize log p(x)                                                    │
│  Problem: p(x) = ∫ p(x|z)p(z)dz is intractable                             │
│                                                                             │
│  Solution: Optimize lower bound instead                                     │
│                                                                             │
│  log p(x) ≥ E_q[log p(x|z)] - D_KL(q(z|x) || p(z))                        │
│             \_____________/   \____________________/                        │
│             Reconstruction     KL regularization                            │
│                                                                             │
│  REPARAMETERIZATION TRICK                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  Problem: Can't backprop through sampling z ~ q(z|x)                        │
│                                                                             │
│  Solution: z = μ + σ ⊙ ε,  where ε ~ N(0, I)                               │
│                                                                             │
│  Now gradients flow through μ and σ (deterministic operations)              │
│                                                                             │
│  GAN: MINIMAX GAME                                                          │
│  ────────────────────                                                       │
│                                                                             │
│  min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]               │
│                                                                             │
│  Generator minimizes:    E_z[log(1 - D(G(z)))]                             │
│  Discriminator maximizes: E_x[log D(x)] + E_z[log(1 - D(G(z)))]            │
│                                                                             │
│  Optimal discriminator: D*(x) = p_data(x) / (p_data(x) + p_g(x))           │
│                                                                             │
│  DIFFUSION: FORWARD PROCESS                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)                                    │
│                                                                             │
│  Directly sample any timestep:                                              │
│  q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)                                       │
│                                                                             │
│  where ᾱₜ = ∏ₛ₌₁ᵗ (1 - βₛ)                                                │
│                                                                             │
│  DIFFUSION: REVERSE PROCESS                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  p_θ(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ, t), Σ_θ(xₜ, t))                          │
│                                                                             │
│  Network predicts noise: ε_θ(xₜ, t)                                        │
│  Loss: E[‖ε - ε_θ(xₜ, t)‖²]                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Quick Reference

### Common Derivatives Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DERIVATIVES CHEAT SHEET                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ACTIVATION FUNCTIONS                                                       │
│  ────────────────────                                                       │
│  Function              Derivative                                           │
│  ─────────────────────────────────────                                      │
│  ReLU(x)               1 if x > 0, else 0                                  │
│  LeakyReLU(x)          1 if x > 0, else α                                  │
│  Sigmoid σ(x)          σ(x)(1 - σ(x))                                      │
│  Tanh(x)               1 - tanh²(x)                                        │
│  Softmax               (see special section above)                          │
│  GELU(x)               (complex, usually use autograd)                      │
│                                                                             │
│  LOSS FUNCTIONS                                                             │
│  ──────────────                                                             │
│  Function              Gradient w.r.t. prediction                           │
│  ─────────────────────────────────────                                      │
│  MSE: (ŷ - y)²        2(ŷ - y)                                             │
│  MAE: |ŷ - y|          sign(ŷ - y)                                          │
│  Cross-Entropy         softmax(logits) - one_hot(y)                        │
│  Binary CE             sigmoid(logit) - y                                   │
│                                                                             │
│  REGULARIZATION                                                             │
│  ──────────────                                                             │
│  L2: λ‖W‖²            2λW                                                  │
│  L1: λ‖W‖₁            λ·sign(W)                                            │
│                                                                             │
│  LAYER OPERATIONS                                                           │
│  ────────────────                                                           │
│  y = Wx + b            ∂L/∂W = (∂L/∂y)xᵀ,  ∂L/∂x = Wᵀ(∂L/∂y)             │
│  y = x₁ + x₂           ∂L/∂x₁ = ∂L/∂y,  ∂L/∂x₂ = ∂L/∂y                    │
│  y = x₁ ⊙ x₂          ∂L/∂x₁ = x₂ ⊙ ∂L/∂y,  ∂L/∂x₂ = x₁ ⊙ ∂L/∂y          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Notation Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NOTATION GUIDE                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VECTORS AND MATRICES                                                       │
│  ────────────────────                                                       │
│  x, y, z          Vectors (lowercase bold in textbooks)                     │
│  W, A, B          Matrices (uppercase)                                      │
│  xᵢ               i-th element of vector x                                  │
│  Wᵢⱼ              Element at row i, column j                                │
│  ‖x‖              Norm (usually L2)                                        │
│  xᵀ               Transpose                                                 │
│  x · y or xᵀy     Dot product                                               │
│  ⊙                Element-wise (Hadamard) product                           │
│                                                                             │
│  CALCULUS                                                                   │
│  ────────                                                                   │
│  f'(x) or df/dx   Derivative                                                │
│  ∂f/∂x            Partial derivative                                        │
│  ∇f               Gradient vector                                           │
│  J or ∂f/∂x       Jacobian matrix                                           │
│  H or ∇²f         Hessian matrix                                            │
│                                                                             │
│  PROBABILITY                                                                │
│  ───────────                                                                │
│  P(X)             Probability of X                                          │
│  P(X|Y)           Conditional probability                                   │
│  p(x)             Probability density function                              │
│  E[X]             Expectation                                               │
│  Var(X)           Variance                                                  │
│  N(μ, σ²)         Normal distribution                                       │
│                                                                             │
│  DEEP LEARNING                                                              │
│  ─────────────                                                              │
│  θ                All model parameters                                       │
│  L or J           Loss function                                              │
│  α or η           Learning rate                                             │
│  x⁽ⁱ⁾             i-th training example                                     │
│  ŷ                Prediction                                                │
│  h or a           Hidden activations                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Exercises

### Exercise 1: Vector Operations
```python
# Implement these without using numpy's built-in functions
import numpy as np

def dot_product(a, b):
    """Compute dot product of two vectors."""
    # Your code here
    pass

def l2_norm(x):
    """Compute L2 norm of a vector."""
    # Your code here
    pass

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    # Your code here
    pass

# Test
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Dot product: {dot_product(a, b)}")  # Should be 32
print(f"L2 norm of a: {l2_norm(a)}")  # Should be ~3.74
print(f"Cosine similarity: {cosine_similarity(a, b)}")  # Should be ~0.97
```

### Exercise 2: Matrix Multiplication
```python
def matmul(A, B):
    """Matrix multiplication without numpy.dot or @"""
    # Your code here
    pass

# Test
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(matmul(A, B))  # Should be [[19, 22], [43, 50]]
```

### Exercise 3: Softmax and Cross-Entropy
```python
def softmax(z):
    """Numerically stable softmax."""
    # Your code here
    pass

def cross_entropy_loss(logits, labels):
    """Cross-entropy loss for batch of logits and integer labels."""
    # Your code here
    pass

def softmax_gradient(logits, labels):
    """Gradient of cross-entropy loss w.r.t. logits."""
    # Your code here
    pass

# Test
logits = np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 3.0]])
labels = np.array([0, 2])
print(f"Softmax: {softmax(logits)}")
print(f"Loss: {cross_entropy_loss(logits, labels)}")
print(f"Gradient: {softmax_gradient(logits, labels)}")
```

### Exercise 4: Chain Rule
```python
# Compute gradients for a simple network:
# z = Wx + b
# a = ReLU(z)
# loss = sum(a)

def forward_and_backward(W, x, b):
    """
    Returns: (loss, dW, dx, db)
    """
    # Forward
    z = # ?
    a = # ReLU
    loss = # ?

    # Backward
    da = # ?
    dz = # ?
    dW = # ?
    dx = # ?
    db = # ?

    return loss, dW, dx, db

# Test with simple values and verify with numerical gradient
```

---

## References

### Textbooks
- [Mathematics for Machine Learning](https://mml-book.github.io/) - Free online book
- [Deep Learning](https://www.deeplearningbook.org/) - Goodfellow et al., Part I
- [Linear Algebra Done Right](https://linear.axler.net/) - Axler
- [Calculus](https://www.whitman.edu/mathematics/calculus/) - Free online

### Video Courses
- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- [Khan Academy: Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)

### Quick References
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - Matrix derivative formulas
- [Probability Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)

---

*This module provides the mathematical foundation for the rest of the course.
Refer back to it whenever you encounter unfamiliar notation or need to
understand why an algorithm works the way it does.*
