# Exercises and Practice Problems

A comprehensive collection of exercises for each module, ranging from conceptual understanding to hands-on implementation.

**Difficulty Levels:**
- **[Basic]** - Conceptual understanding, definitions
- **[Intermediate]** - Calculations, short implementations
- **[Advanced]** - Complex implementations, design problems
- **[Challenge]** - Research-level, open-ended problems

---

## Table of Contents

1. [Module 1: Neural Network Foundations](#module-1-neural-network-foundations)
2. [Module 2: Training & Optimization](#module-2-training--optimization)
3. [Module 3: Convolutional Neural Networks](#module-3-convolutional-neural-networks)
4. [Module 4: Sequence Models](#module-4-sequence-models)
5. [Module 5: Attention & Transformers](#module-5-attention--transformers)
6. [Module 6: Large Language Models](#module-6-large-language-models)
7. [Module 7: Generative Models](#module-7-generative-models)
8. [Module 8: Advanced Topics](#module-8-advanced-topics)
9. [Integrated Projects](#integrated-projects)
10. [Solutions Guide](#solutions-guide)

---

## Module 1: Neural Network Foundations

### Conceptual Questions

**Exercise 1.1 [Basic]**: Linear vs Non-linear Decision Boundaries

Consider the XOR problem with four data points:
```
(0, 0) -> 0
(0, 1) -> 1
(1, 0) -> 1
(1, 1) -> 0
```

a) Prove that no single linear classifier (one line) can correctly classify all four points.
b) Draw two lines that together can solve the XOR problem.
c) How does a 2-layer neural network use these two lines to solve XOR?

---

**Exercise 1.2 [Basic]**: Activation Functions

Compare the following activation functions by filling in the table:

| Property | ReLU | Sigmoid | Tanh | Leaky ReLU |
|----------|------|---------|------|------------|
| Output range | | | | |
| Vanishing gradient problem? | | | | |
| Dead neuron problem? | | | | |
| Zero-centered? | | | | |
| Computational cost | | | | |

---

**Exercise 1.3 [Intermediate]**: Manual Forward Pass

Given the following 2-layer network:

```
Input: x = [1, 2]

Layer 1:
W1 = [[0.1, 0.2],
      [0.3, 0.4],
      [0.5, 0.6]]
b1 = [0.1, 0.1, 0.1]

Activation: ReLU

Layer 2:
W2 = [[0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6]]
b2 = [0, 0]
```

Calculate:
a) The output of Layer 1 before activation: z1 = W1 @ x + b1
b) The output of Layer 1 after ReLU: h1 = ReLU(z1)
c) The final output: y = W2 @ h1 + b2

Show all steps.

---

**Exercise 1.4 [Intermediate]**: Parameter Counting

For each network architecture, calculate the total number of trainable parameters:

a) Fully connected: Input(784) -> Hidden(256) -> Hidden(128) -> Output(10)

b) Same as (a) but with batch normalization after each hidden layer

c) Input(32x32x3) -> Conv(3x3, 64 filters) -> Conv(3x3, 128 filters) -> FC(10)
   (Assume no padding, stride 1)

---

**Exercise 1.5 [Advanced]**: Implement a Neuron Class

```python
import numpy as np

class Neuron:
    """
    Implement a single neuron with:
    - Configurable activation function
    - Forward pass
    - Backward pass (gradient computation)
    """

    def __init__(self, n_inputs, activation='relu'):
        """
        Initialize weights and bias.
        Use Xavier initialization for sigmoid/tanh, He initialization for ReLU.
        """
        # YOUR CODE HERE
        pass

    def forward(self, x):
        """
        Compute output: activation(w @ x + b)
        Store intermediate values for backward pass.
        """
        # YOUR CODE HERE
        pass

    def backward(self, grad_output):
        """
        Compute gradients with respect to:
        - weights (self.grad_w)
        - bias (self.grad_b)
        - inputs (return value)
        """
        # YOUR CODE HERE
        pass

# Test your implementation
np.random.seed(42)
neuron = Neuron(3, activation='relu')
x = np.array([1.0, 2.0, 3.0])
y = neuron.forward(x)
grad = neuron.backward(1.0)
print(f"Output: {y}")
print(f"Gradient w.r.t input: {grad}")
```

---

### Coding Exercises

**Exercise 1.6 [Intermediate]**: Universal Approximation Visualization

Write code to demonstrate that a neural network with one hidden layer can approximate any continuous function.

```python
import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    """A complex function to approximate"""
    return np.sin(x) + 0.5 * np.sin(3*x) + 0.3 * np.cos(5*x)

# TODO: Implement a simple neural network with one hidden layer
# TODO: Train it to approximate target_function on [-3, 3]
# TODO: Plot the original function and the network's approximation
# TODO: Experiment with different numbers of hidden neurons (5, 20, 100)

# YOUR CODE HERE
```

---

## Module 2: Training & Optimization

### Conceptual Questions

**Exercise 2.1 [Basic]**: Loss Function Selection

For each task, select the most appropriate loss function and explain why:

a) Multi-class classification (mutually exclusive classes)
b) Multi-label classification (multiple labels per sample)
c) Regression with outliers
d) Binary classification with class imbalance
e) Predicting probability distributions

---

**Exercise 2.2 [Basic]**: Gradient Descent Variants

Compare the following optimization methods:

| Method | Update Rule | Key Hyperparameters | When to Use |
|--------|-------------|---------------------|-------------|
| SGD | | | |
| SGD + Momentum | | | |
| RMSprop | | | |
| Adam | | | |
| AdamW | | | |

---

**Exercise 2.3 [Intermediate]**: Manual Backpropagation

Given:
- Forward pass: y = sigmoid(w2 * relu(w1 * x + b1) + b2)
- Loss: L = (y - target)^2 / 2
- x = 1, w1 = 0.5, b1 = 0.1, w2 = 0.8, b2 = 0.2, target = 1

Calculate all gradients: dL/dw2, dL/db2, dL/dw1, dL/db1

Show the complete chain rule derivation for each.

---

**Exercise 2.4 [Intermediate]**: Learning Rate Analysis

```python
import numpy as np

def rosenbrock(x, y):
    """Rosenbrock function - a classic optimization test function"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Starting point
x0 = np.array([-1.0, 1.0])

# TODO: Implement gradient descent with different learning rates
# TODO: Compare lr = 0.0001, 0.001, 0.01 (this one will diverge)
# TODO: Plot the optimization trajectory on contour plot of Rosenbrock function
# TODO: Implement Adam optimizer and compare

# YOUR CODE HERE
```

---

**Exercise 2.5 [Intermediate]**: Regularization Effects

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a dataset with more features than samples (overfitting prone)
X, y = make_classification(n_samples=100, n_features=50, n_informative=10,
                          n_redundant=20, n_clusters_per_class=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# TODO: Train a neural network without regularization
# TODO: Train with L2 regularization (weight decay)
# TODO: Train with dropout
# TODO: Train with both L2 and dropout
# TODO: Compare training and test accuracy for each

# YOUR CODE HERE
```

---

**Exercise 2.6 [Advanced]**: Implement Optimizers from Scratch

```python
import numpy as np

class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p['grad'] = np.zeros_like(p['value'])

class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent"""
    def step(self):
        # YOUR CODE HERE
        pass

class SGDMomentum(Optimizer):
    """SGD with Momentum"""
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p['value']) for p in parameters]

    def step(self):
        # YOUR CODE HERE
        pass

class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p['value']) for p in parameters]
        self.v = [np.zeros_like(p['value']) for p in parameters]

    def step(self):
        # YOUR CODE HERE
        # Remember to implement bias correction!
        pass

# Test on a simple quadratic function
# f(x) = x^2 + y^2, minimum at (0, 0)
```

---

### Debugging Exercises

**Exercise 2.7 [Intermediate]**: Debug the Training Loop

The following training code has several bugs. Find and fix them:

```python
import numpy as np

class BuggyNetwork:
    def __init__(self):
        self.w1 = np.random.randn(10, 784)  # Bug 1
        self.w2 = np.random.randn(10, 10)

    def forward(self, x):
        self.z1 = self.w1 @ x
        self.h1 = np.maximum(self.z1, 0)  # ReLU
        self.z2 = self.w2 @ self.h1
        return self.softmax(self.z2)

    def softmax(self, x):
        exp_x = np.exp(x)  # Bug 2
        return exp_x / np.sum(exp_x)

    def backward(self, y_pred, y_true):
        grad = y_pred - y_true
        self.grad_w2 = grad @ self.h1.T  # Bug 3
        grad_h1 = self.w2.T @ grad
        grad_z1 = grad_h1 * (self.z1 > 0)
        self.grad_w1 = grad_z1 @ self.x.T

    def train_step(self, x, y, lr=0.01):
        self.x = x
        y_pred = self.forward(x)
        self.backward(y_pred, y)
        self.w1 = self.w1 - lr * self.grad_w1  # Bug 4
        self.w2 = self.w2 - lr * self.grad_w2
        return -np.sum(y * np.log(y_pred))  # Bug 5

# Bugs to find:
# 1. Weight initialization issue
# 2. Numerical stability in softmax
# 3. Gradient dimension mismatch
# 4. Missing something in update
# 5. Potential numerical issue
```

---

## Module 3: Convolutional Neural Networks

### Conceptual Questions

**Exercise 3.1 [Basic]**: Convolution Output Size

For each configuration, calculate the output spatial dimensions:

a) Input: 32x32, Kernel: 3x3, Padding: 0, Stride: 1
b) Input: 32x32, Kernel: 5x5, Padding: 2, Stride: 1
c) Input: 32x32, Kernel: 3x3, Padding: 1, Stride: 2
d) Input: 224x224, Kernel: 7x7, Padding: 3, Stride: 2

Formula: Output = floor((Input + 2*Padding - Kernel) / Stride) + 1

---

**Exercise 3.2 [Basic]**: Receptive Field Calculation

Calculate the receptive field after each layer:

```
Input: 224x224
Layer 1: Conv 3x3, stride 1
Layer 2: Conv 3x3, stride 1
Layer 3: MaxPool 2x2, stride 2
Layer 4: Conv 3x3, stride 1
Layer 5: Conv 3x3, stride 1
```

Draw a diagram showing how a pixel in the final feature map relates to the input.

---

**Exercise 3.3 [Intermediate]**: Manual Convolution

Compute the output of convolving this input with the given kernel:

```
Input (5x5):               Kernel (3x3):
[1, 2, 3, 4, 5]           [1,  0, -1]
[2, 3, 4, 5, 6]           [2,  0, -2]
[3, 4, 5, 6, 7]           [1,  0, -1]
[4, 5, 6, 7, 8]
[5, 6, 7, 8, 9]

Padding: 0, Stride: 1
```

What type of edge does this kernel detect?

---

**Exercise 3.4 [Intermediate]**: Architecture Analysis

For VGG-16, answer the following:

a) Why does VGG use multiple 3x3 convolutions instead of larger kernels?
   - What is the receptive field of two 3x3 convolutions?
   - What is the receptive field of one 5x5 convolution?
   - Compare the number of parameters

b) How many parameters are in VGG-16? Break down by layer type.

c) Where does most of the memory go during training?

---

**Exercise 3.5 [Advanced]**: Implement Convolution from Scratch

```python
import numpy as np

def conv2d_naive(input, kernel, stride=1, padding=0):
    """
    Naive implementation of 2D convolution.

    Args:
        input: (H, W) or (H, W, C) array
        kernel: (kH, kW) or (kH, kW, C) array
        stride: int
        padding: int

    Returns:
        output: (H_out, W_out) array
    """
    # YOUR CODE HERE
    pass

def conv2d_im2col(input, kernel, stride=1, padding=0):
    """
    Efficient implementation using im2col.

    The im2col trick converts convolution to matrix multiplication:
    1. Extract all patches from input as columns of a matrix
    2. Flatten kernel to a row vector
    3. Matrix multiply
    4. Reshape to output
    """
    # YOUR CODE HERE
    pass

# Test both implementations
input = np.random.randn(28, 28)
kernel = np.random.randn(3, 3)
out_naive = conv2d_naive(input, kernel)
out_im2col = conv2d_im2col(input, kernel)
assert np.allclose(out_naive, out_im2col)
```

---

**Exercise 3.6 [Advanced]**: Build a CNN from Scratch

```python
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # Initialize weights using He initialization
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # x: (N, C_in, H, W)
        # output: (N, C_out, H_out, W_out)
        # YOUR CODE HERE
        pass

    def backward(self, grad_output):
        # Compute gradients w.r.t weights and input
        # YOUR CODE HERE
        pass

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        # YOUR CODE HERE
        pass

    def backward(self, grad_output):
        # YOUR CODE HERE
        pass

class Flatten:
    def forward(self, x):
        # YOUR CODE HERE
        pass

    def backward(self, grad_output):
        # YOUR CODE HERE
        pass

# Build a simple CNN
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 16, 3, padding=1)
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(16, 32, 3, padding=1)
        self.pool2 = MaxPool2D(2)
        self.flatten = Flatten()
        # Add FC layers...

    def forward(self, x):
        # YOUR CODE HERE
        pass
```

---

### Visualization Exercises

**Exercise 3.7 [Intermediate]**: Filter Visualization

```python
import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Load a pretrained VGG model
vgg = models.vgg16(pretrained=True)

# TODO: Visualize the first layer filters (64 filters of size 3x3x3)
# TODO: What patterns do these filters detect?

# TODO: Implement gradient-based filter visualization for deeper layers
# (maximize activation of a specific filter by optimizing an input image)

# YOUR CODE HERE
```

---

## Module 4: Sequence Models

### Conceptual Questions

**Exercise 4.1 [Basic]**: RNN Unrolling

Draw the unrolled computation graph for an RNN processing the sequence "cat":

```
Input: ["c", "a", "t"]

Show:
- Hidden states h0, h1, h2, h3
- Weight matrices U (input-to-hidden), W (hidden-to-hidden), V (hidden-to-output)
- Where parameters are shared
```

---

**Exercise 4.2 [Basic]**: Vanishing Gradient Analysis

Explain why gradients vanish in vanilla RNNs:

a) Write the equation for gradient flow through time steps
b) What happens when we multiply the same matrix W many times?
c) Under what conditions on W's eigenvalues do gradients vanish vs explode?
d) How do LSTMs address this problem?

---

**Exercise 4.3 [Intermediate]**: LSTM Gate Analysis

For each LSTM gate, explain:

a) **Forget gate**: What does it forget? When would it be ~0 vs ~1?
b) **Input gate**: What information does it control? How?
c) **Output gate**: How does it differ from the cell state?

Give a concrete example: processing "The cat, which was very fluffy, sat"
- When should the forget gate activate?
- When should the input gate be high?

---

**Exercise 4.4 [Intermediate]**: Manual LSTM Forward Pass

Given:
```
Input sequence: [x1, x2] where x1 = [1, 0], x2 = [0, 1]
Hidden size: 2
Initial: h0 = [0, 0], c0 = [0, 0]

Weight matrices (simplified, all same):
Wf = Wi = Wo = Wc = [[0.5, 0.5],
                      [0.5, 0.5]]
Uf = Ui = Uo = Uc = [[0.5, 0.5],
                      [0.5, 0.5]]
bf = bi = bo = bc = [0, 0]
```

Calculate h1, c1, h2, c2 step by step.

---

**Exercise 4.5 [Advanced]**: Implement RNN from Scratch

```python
import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        """
        Args:
            inputs: list of input vectors, each (input_size, 1)
            h_prev: initial hidden state (hidden_size, 1)
        Returns:
            outputs: list of output vectors
            hidden_states: list of hidden states
        """
        # YOUR CODE HERE
        pass

    def backward(self, d_outputs, hidden_states, inputs):
        """
        Backpropagation through time (BPTT).

        Args:
            d_outputs: gradients of loss w.r.t outputs
            hidden_states: cached hidden states from forward
            inputs: original inputs
        Returns:
            Gradients for all parameters
        """
        # YOUR CODE HERE
        pass

# Test: Character-level language model
text = "hello world"
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

rnn = RNN(len(chars), 50, len(chars))
# Train on text and generate samples
```

---

**Exercise 4.6 [Advanced]**: Implement LSTM from Scratch

```python
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        # Initialize weights for all four gates
        # YOUR CODE HERE
        pass

    def forward(self, x, h_prev, c_prev):
        """
        Single step forward pass.

        Args:
            x: input (input_size,)
            h_prev: previous hidden state (hidden_size,)
            c_prev: previous cell state (hidden_size,)
        Returns:
            h_next, c_next, cache (for backward)
        """
        # YOUR CODE HERE
        # Compute: f, i, o, c_tilde, c_next, h_next
        pass

    def backward(self, dh_next, dc_next, cache):
        """
        Single step backward pass.
        """
        # YOUR CODE HERE
        pass

# Implement gradient checking to verify your implementation
def gradient_check(lstm, x, h, c, eps=1e-5):
    """
    Numerically verify gradients.
    """
    # YOUR CODE HERE
    pass
```

---

## Module 5: Attention & Transformers

### Conceptual Questions

**Exercise 5.1 [Basic]**: Attention Intuition

Explain the Query-Key-Value metaphor using a library analogy:
- What is the query?
- What are the keys?
- What are the values?
- What does the attention score represent?
- What does the weighted sum of values represent?

---

**Exercise 5.2 [Basic]**: Self-Attention vs Cross-Attention

Compare and contrast:

| Aspect | Self-Attention | Cross-Attention |
|--------|---------------|-----------------|
| Where Q comes from | | |
| Where K, V come from | | |
| Use case | | |
| Example in Transformer | | |

---

**Exercise 5.3 [Intermediate]**: Manual Attention Calculation

Given:
```
Q = [[1, 0],    (2 queries, dim=2)
     [0, 1]]

K = [[1, 0],    (3 keys, dim=2)
     [0, 1],
     [1, 1]]

V = [[1, 2],    (3 values, dim=2)
     [3, 4],
     [5, 6]]
```

Calculate:
a) Attention scores: Q @ K^T
b) Scaled attention scores (divide by sqrt(d_k))
c) Attention weights (softmax)
d) Output (weights @ V)

---

**Exercise 5.4 [Intermediate]**: Multi-Head Attention

If single-head attention uses d_model = 512:
- With 8 heads, what is the dimension of each head?
- How many parameters in the Q, K, V projections?
- How many parameters in the output projection?
- Why might multiple heads be better than one large head?

---

**Exercise 5.5 [Intermediate]**: Positional Encoding Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_len, d_model):
    """
    Implement sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    # YOUR CODE HERE
    pass

# TODO: Generate PE for max_len=100, d_model=64
# TODO: Visualize the encoding as a heatmap
# TODO: Plot PE[0] and PE[1] to see the pattern for different dimensions
# TODO: Compute dot product between PE[i] and PE[j] for various i,j
#       What pattern do you observe?

# YOUR CODE HERE
```

---

**Exercise 5.6 [Advanced]**: Implement Scaled Dot-Product Attention

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: queries (batch, seq_len_q, d_k)
        K: keys (batch, seq_len_k, d_k)
        V: values (batch, seq_len_k, d_v)
        mask: optional mask (batch, seq_len_q, seq_len_k)

    Returns:
        output: (batch, seq_len_q, d_v)
        attention_weights: (batch, seq_len_q, seq_len_k)
    """
    # YOUR CODE HERE
    pass

def multi_head_attention(Q, K, V, num_heads, d_model, mask=None):
    """
    Multi-head attention.

    Args:
        Q, K, V: (batch, seq_len, d_model)
        num_heads: number of attention heads
        d_model: model dimension
        mask: optional mask

    Returns:
        output: (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    # 1. Project Q, K, V for each head
    # 2. Reshape to (batch, num_heads, seq_len, d_k)
    # 3. Apply attention
    # 4. Concatenate heads
    # 5. Final projection
    pass

# Test with some data
batch_size = 2
seq_len = 10
d_model = 64
num_heads = 8

Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

output = multi_head_attention(Q, K, V, num_heads, d_model)
print(f"Output shape: {output.shape}")  # Should be (2, 10, 64)
```

---

**Exercise 5.7 [Advanced]**: Implement a Transformer Block

```python
import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        # YOUR CODE HERE
        pass

class FeedForward:
    def __init__(self, d_model, d_ff):
        # Two linear layers with ReLU
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        Pre-LN Transformer block:
        x -> LN -> Attention -> + -> LN -> FF -> + -> output
              └────────────────┘     └─────────┘
        """
        # YOUR CODE HERE
        pass

# Build a small Transformer
class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len):
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.layers = [TransformerBlock(d_model, num_heads, d_ff)
                       for _ in range(num_layers)]
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, x):
        # x: (batch, seq_len) token ids
        # YOUR CODE HERE
        pass
```

---

**Exercise 5.8 [Challenge]**: KV Cache Implementation

```python
class KVCache:
    """
    Implement KV caching for efficient autoregressive generation.

    During generation, we don't need to recompute attention for
    all previous tokens - we can cache their K and V vectors.
    """

    def __init__(self, num_layers, max_seq_len, num_heads, head_dim):
        # YOUR CODE HERE
        pass

    def update(self, layer_idx, new_k, new_v):
        """Add new K, V vectors to the cache"""
        # YOUR CODE HERE
        pass

    def get(self, layer_idx):
        """Get cached K, V for a layer"""
        # YOUR CODE HERE
        pass

def generate_with_kv_cache(model, prompt_ids, max_new_tokens):
    """
    Generate text using KV caching.

    Compare the number of operations with vs without caching.
    """
    # YOUR CODE HERE
    pass
```

---

## Module 6: Large Language Models

### Conceptual Questions

**Exercise 6.1 [Basic]**: Tokenization

Given the sentence: "Hello, I'm learning about tokenization!"

a) How would character-level tokenization split this?
b) How would word-level tokenization split this?
c) Why is subword tokenization (BPE) often better?
d) What are the tradeoffs of vocabulary size?

---

**Exercise 6.2 [Basic]**: Scaling Laws

Explain the Chinchilla scaling laws:
a) What is the relationship between model size and training tokens?
b) Why was GPT-3 "undertrained" according to these laws?
c) How do these laws affect practical training decisions?

---

**Exercise 6.3 [Intermediate]**: Fine-tuning vs Prompting

For each task, decide whether to use fine-tuning, few-shot prompting, or zero-shot prompting. Justify your choice.

a) Sentiment analysis on product reviews (10,000 labeled examples available)
b) Translating between English and a rare language (100 examples available)
c) Classifying emails into categories (no labeled data, but categories are clear)
d) Extracting structured information from legal documents (1,000 examples)
e) Answering questions about a company's internal documentation

---

**Exercise 6.4 [Intermediate]**: LoRA Analysis

```python
# Original weight matrix
W = np.random.randn(4096, 4096)  # ~16M parameters

# LoRA with rank 8
A = np.random.randn(4096, 8)     # How many parameters?
B = np.random.randn(8, 4096)     # How many parameters?

# Questions:
# a) What is the total number of trainable parameters in LoRA?
# b) What percentage of the original parameters is this?
# c) What is the rank of the update matrix A @ B?
# d) Why might this low-rank approximation work well for fine-tuning?
```

---

**Exercise 6.5 [Intermediate]**: RLHF Components

Draw a diagram showing the RLHF pipeline:
1. Supervised fine-tuning (SFT)
2. Reward model training
3. PPO optimization

For each stage, specify:
- What data is needed
- What is being optimized
- What the outputs are

---

**Exercise 6.6 [Advanced]**: Implement BPE Tokenizer

```python
class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def train(self, texts):
        """
        Train BPE on a corpus.

        Algorithm:
        1. Initialize vocabulary with all characters
        2. Count frequency of adjacent pairs
        3. Merge most frequent pair
        4. Repeat until vocab_size reached
        """
        # YOUR CODE HERE
        pass

    def encode(self, text):
        """Encode text to token ids"""
        # YOUR CODE HERE
        pass

    def decode(self, ids):
        """Decode token ids to text"""
        # YOUR CODE HERE
        pass

# Train on a small corpus
corpus = [
    "low lower lowest",
    "new newer newest",
    "the quick brown fox",
    "jumped over the lazy dog"
]

tokenizer = BPETokenizer(vocab_size=50)
tokenizer.train(corpus)
print(tokenizer.encode("the newer fox"))
```

---

**Exercise 6.7 [Advanced]**: Implement Reward Model Training

```python
import numpy as np

class RewardModel:
    """
    A reward model that scores responses.

    Training data: pairs of (prompt, chosen_response, rejected_response)
    Objective: r(chosen) > r(rejected)
    """

    def __init__(self, base_model):
        self.base_model = base_model
        # Add a reward head
        self.reward_head = np.random.randn(base_model.hidden_size, 1) * 0.02

    def forward(self, prompt, response):
        """Compute reward score for a response"""
        # YOUR CODE HERE
        pass

    def compute_loss(self, prompt, chosen, rejected):
        """
        Bradley-Terry loss:
        loss = -log(sigmoid(r_chosen - r_rejected))
        """
        # YOUR CODE HERE
        pass

    def train_step(self, batch):
        """One training step"""
        # YOUR CODE HERE
        pass

# Generate synthetic preference data and train
```

---

**Exercise 6.8 [Challenge]**: Implement DPO

```python
class DPO:
    """
    Direct Preference Optimization.

    Instead of training a reward model then doing RL,
    DPO directly optimizes the policy from preferences.

    Loss = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) -
                                log(pi(y_l|x)/pi_ref(y_l|x)))))

    Where:
    - y_w is the preferred response
    - y_l is the rejected response
    - pi is the policy being trained
    - pi_ref is the reference policy (frozen)
    - beta is a temperature parameter
    """

    def __init__(self, policy, reference_policy, beta=0.1):
        self.policy = policy
        self.reference_policy = reference_policy
        self.beta = beta

    def compute_loss(self, prompts, chosen, rejected):
        """Compute DPO loss"""
        # YOUR CODE HERE
        pass

# Compare DPO vs PPO:
# - When is DPO preferable?
# - What are the computational differences?
```

---

## Module 7: Generative Models

### Conceptual Questions

**Exercise 7.1 [Basic]**: Generative Model Comparison

Fill in the comparison table:

| Aspect | VAE | GAN | Diffusion |
|--------|-----|-----|-----------|
| Training objective | | | |
| Can compute likelihood? | | | |
| Sample quality | | | |
| Training stability | | | |
| Mode coverage | | | |
| Generation speed | | | |
| Best use case | | | |

---

**Exercise 7.2 [Basic]**: VAE Intuition

Explain in your own words:
a) Why do we need the "reparameterization trick" in VAEs?
b) What does the KL divergence term in the ELBO do?
c) What happens if beta (the KL weight) is too high? Too low?

---

**Exercise 7.3 [Intermediate]**: GAN Training Dynamics

```
Consider a GAN with:
- Generator G(z) that maps noise z to images
- Discriminator D(x) that outputs probability of real

Training alternates between:
1. Train D to maximize: E[log D(real)] + E[log(1 - D(G(z)))]
2. Train G to minimize: E[log(1 - D(G(z)))]

Questions:
a) What should D output for perfect real images?
b) What should D output for generated images (initially)?
c) What gradient does G receive when D is confident generated images are fake?
d) This is called "vanishing gradients for the generator" - explain why.
e) How does the "non-saturating" GAN loss address this?
```

---

**Exercise 7.4 [Intermediate]**: Diffusion Process Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_diffusion(x0, t, beta_schedule):
    """
    Implement the forward diffusion process:
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Args:
        x0: original image
        t: timestep
        beta_schedule: noise schedule
    """
    # YOUR CODE HERE
    pass

# TODO: Visualize the forward process
# 1. Start with a clear image (e.g., MNIST digit)
# 2. Show x_t at t = 0, 250, 500, 750, 1000
# 3. Plot the noise schedule (linear vs cosine)

# YOUR CODE HERE
```

---

**Exercise 7.5 [Advanced]**: Implement VAE from Scratch

```python
import numpy as np

class VAE:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        # Encoder
        self.encoder_fc1 = Linear(input_dim, hidden_dim)
        self.encoder_mu = Linear(hidden_dim, latent_dim)
        self.encoder_logvar = Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc1 = Linear(latent_dim, hidden_dim)
        self.decoder_out = Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Return mu and logvar of q(z|x)"""
        # YOUR CODE HERE
        pass

    def reparameterize(self, mu, logvar):
        """Sample z using reparameterization trick"""
        # YOUR CODE HERE
        pass

    def decode(self, z):
        """Decode z to reconstruction"""
        # YOUR CODE HERE
        pass

    def forward(self, x):
        """Full forward pass"""
        # YOUR CODE HERE
        pass

    def loss(self, x, x_recon, mu, logvar):
        """
        ELBO loss = Reconstruction loss + KL divergence

        Reconstruction: E[log p(x|z)]
        KL: KL(q(z|x) || p(z)) where p(z) = N(0, I)

        KL has closed form for Gaussians:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        # YOUR CODE HERE
        pass

# Train on MNIST
# Visualize:
# 1. Reconstructions
# 2. Samples from prior
# 3. Latent space (if 2D)
```

---

**Exercise 7.6 [Advanced]**: Implement GAN from Scratch

```python
import numpy as np

class Generator:
    def __init__(self, latent_dim, hidden_dim, output_dim):
        # YOUR CODE HERE
        pass

    def forward(self, z):
        # YOUR CODE HERE
        pass

class Discriminator:
    def __init__(self, input_dim, hidden_dim):
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE (output probability)
        pass

class GAN:
    def __init__(self, generator, discriminator):
        self.G = generator
        self.D = discriminator

    def train_discriminator_step(self, real_batch, lr=0.0002):
        """
        Update D to maximize:
        E[log D(x)] + E[log(1 - D(G(z)))]
        """
        # YOUR CODE HERE
        pass

    def train_generator_step(self, batch_size, lr=0.0002):
        """
        Update G to maximize:
        E[log D(G(z))]  # non-saturating loss
        """
        # YOUR CODE HERE
        pass

    def train(self, data, epochs, batch_size):
        """Full training loop"""
        # YOUR CODE HERE
        # Include: loss plotting, sample generation every N epochs
        pass

# Train on 2D Gaussian mixture and visualize
```

---

**Exercise 7.7 [Challenge]**: Implement DDPM

```python
class DDPM:
    """
    Denoising Diffusion Probabilistic Model.

    Forward process: q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)

    We train a network to predict the noise:
    L = E_{t,x_0,eps}[||eps - eps_theta(x_t, t)||^2]

    For sampling, we reverse the process using the predicted noise.
    """

    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model  # U-Net or similar
        self.num_timesteps = num_timesteps

        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def q_sample(self, x0, t, noise=None):
        """
        Sample from q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        # YOUR CODE HERE
        pass

    def training_loss(self, x0):
        """
        Compute training loss.

        1. Sample t uniformly
        2. Sample noise eps
        3. Create x_t using q_sample
        4. Predict noise using model
        5. Return MSE between predicted and true noise
        """
        # YOUR CODE HERE
        pass

    def p_sample(self, x_t, t):
        """
        Sample from p(x_{t-1} | x_t).

        Uses the model to predict noise, then samples.
        """
        # YOUR CODE HERE
        pass

    def sample(self, shape):
        """
        Generate samples by running the reverse process.

        Start from x_T ~ N(0, I) and iterate to x_0.
        """
        # YOUR CODE HERE
        pass

# Train on MNIST and generate samples
```

---

## Module 8: Advanced Topics

### Conceptual Questions

**Exercise 8.1 [Basic]**: RAG vs Fine-tuning

When should you use RAG vs fine-tuning? Consider:

| Scenario | RAG | Fine-tuning | Both |
|----------|-----|-------------|------|
| Adding company documentation | | | |
| Changing the model's tone/style | | | |
| Real-time information needed | | | |
| Specialized domain (e.g., legal) | | | |
| Cost is primary concern | | | |

---

**Exercise 8.2 [Basic]**: Agent Design

Design an agent for the following task:
"Book a flight from NYC to Paris for next Friday, under $500"

a) What tools does the agent need?
b) What information must it gather?
c) Draw the decision tree for the agent
d) What could go wrong? How to handle it?

---

**Exercise 8.3 [Intermediate]**: RAG Evaluation

```python
# You built a RAG system. How do you evaluate it?

# Retrieval metrics:
# - Recall@K: What fraction of relevant documents are in top K?
# - MRR: Mean Reciprocal Rank of first relevant document

# Generation metrics:
# - Faithfulness: Does the answer use only retrieved context?
# - Relevance: Does the answer address the question?
# - Correctness: Is the answer factually correct?

# TODO: Implement evaluation functions
def recall_at_k(retrieved_docs, relevant_docs, k):
    """Compute Recall@K"""
    # YOUR CODE HERE
    pass

def mrr(retrieved_docs, relevant_docs):
    """Compute Mean Reciprocal Rank"""
    # YOUR CODE HERE
    pass

def faithfulness_score(answer, context, llm):
    """
    Use an LLM to check if answer is grounded in context.
    Returns score 0-1.
    """
    # YOUR CODE HERE
    pass

# Create a test dataset and evaluate your RAG system
```

---

**Exercise 8.4 [Intermediate]**: LLM Evaluation

```python
# Evaluate an LLM on different capabilities:

evaluation_suite = {
    "factual_accuracy": [
        {"question": "What year did World War 2 end?", "answer": "1945"},
        {"question": "What is the capital of France?", "answer": "Paris"},
    ],
    "reasoning": [
        {
            "question": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
            "answer": "No"
        }
    ],
    "math": [
        {"question": "What is 17 * 23?", "answer": "391"},
    ],
    "coding": [
        {
            "question": "Write a function to check if a string is a palindrome",
            "test_cases": [("racecar", True), ("hello", False)]
        }
    ]
}

# TODO: Implement evaluation
# TODO: Handle different answer formats (exact match, fuzzy match, code execution)
# TODO: Report results by category
```

---

**Exercise 8.5 [Advanced]**: Implement a RAG System

```python
import numpy as np
from typing import List, Dict

class SimpleRAG:
    def __init__(self, embedding_model, llm, chunk_size=500, chunk_overlap=50):
        self.embedding_model = embedding_model
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: List[str]):
        """
        Process and index documents.

        1. Chunk documents
        2. Embed chunks
        3. Store in vector index
        """
        # YOUR CODE HERE
        pass

    def chunk_document(self, doc: str) -> List[str]:
        """Split document into overlapping chunks"""
        # YOUR CODE HERE
        pass

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant chunks for a query.

        1. Embed query
        2. Compute similarity with all chunks
        3. Return top K
        """
        # YOUR CODE HERE
        pass

    def generate(self, query: str) -> str:
        """
        Full RAG pipeline.

        1. Retrieve relevant chunks
        2. Build prompt with context
        3. Generate answer
        """
        context = self.retrieve(query)
        prompt = self._build_prompt(query, context)
        return self.llm.generate(prompt)

    def _build_prompt(self, query: str, context: List[str]) -> str:
        """Build the augmented prompt"""
        # YOUR CODE HERE
        pass

# Test with some documents
documents = [
    "The Eiffel Tower is located in Paris, France. It was constructed in 1889.",
    "Machine learning is a subset of artificial intelligence...",
    # ... more documents
]

rag = SimpleRAG(embedding_model, llm)
rag.add_documents(documents)
answer = rag.generate("When was the Eiffel Tower built?")
```

---

**Exercise 8.6 [Advanced]**: Implement a ReAct Agent

```python
class Tool:
    def __init__(self, name: str, description: str, function):
        self.name = name
        self.description = description
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

class ReActAgent:
    """
    ReAct: Reasoning + Acting

    The agent follows a loop:
    1. Thought: Reason about what to do
    2. Action: Choose and execute a tool
    3. Observation: Process the result
    4. Repeat until done
    """

    def __init__(self, llm, tools: List[Tool], max_steps: int = 10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        """
        Execute the ReAct loop.
        """
        prompt = self._build_initial_prompt(task)
        history = []

        for step in range(self.max_steps):
            # Get next thought and action from LLM
            response = self.llm.generate(prompt)
            thought, action, action_input = self._parse_response(response)

            history.append({"thought": thought, "action": action, "input": action_input})

            # Check if done
            if action == "finish":
                return action_input

            # Execute action
            if action in self.tools:
                observation = self.tools[action](action_input)
            else:
                observation = f"Unknown tool: {action}"

            history.append({"observation": observation})
            prompt = self._update_prompt(prompt, history[-2:])

        return "Max steps reached without completion"

    def _build_initial_prompt(self, task: str) -> str:
        """Build the initial prompt with tool descriptions"""
        # YOUR CODE HERE
        pass

    def _parse_response(self, response: str):
        """Parse LLM response to extract thought, action, action_input"""
        # YOUR CODE HERE
        pass

# Create tools
tools = [
    Tool("search", "Search the web for information", web_search),
    Tool("calculator", "Perform mathematical calculations", calculate),
    Tool("weather", "Get current weather for a location", get_weather),
]

agent = ReActAgent(llm, tools)
result = agent.run("What is the weather in Paris and convert the temperature from Celsius to Fahrenheit?")
```

---

**Exercise 8.7 [Challenge]**: Model Serving Optimization

```python
# Implement various inference optimizations

class OptimizedInference:
    def __init__(self, model):
        self.model = model
        self.kv_cache = None

    def generate_baseline(self, input_ids, max_new_tokens):
        """
        Baseline generation without optimization.
        Measure: tokens per second, memory usage
        """
        # YOUR CODE HERE
        pass

    def generate_with_kv_cache(self, input_ids, max_new_tokens):
        """
        Generation with KV caching.
        Compare performance to baseline.
        """
        # YOUR CODE HERE
        pass

    def continuous_batching(self, requests):
        """
        Implement continuous batching for multiple concurrent requests.

        Instead of waiting for all sequences to finish,
        start new sequences as soon as others complete.
        """
        # YOUR CODE HERE
        pass

    def speculative_decoding(self, input_ids, max_new_tokens, draft_model):
        """
        Use a smaller model to draft tokens, verify with main model.

        1. Draft K tokens with small model
        2. Verify all K tokens in parallel with large model
        3. Accept prefix of correct tokens
        4. Repeat
        """
        # YOUR CODE HERE
        pass

# Benchmark all methods
# Report: tokens/second, memory usage, latency
```

---

## Integrated Projects

These projects combine concepts from multiple modules.

### Project 1: Build a Complete Classification Pipeline

```
Task: Build an image classification system from scratch

Requirements:
1. Data loading and augmentation (Module 3)
2. Implement CNN architecture (Module 3)
3. Training loop with proper optimization (Module 2)
4. Regularization (dropout, weight decay, early stopping)
5. Hyperparameter tuning
6. Model evaluation and visualization
7. Export model for inference

Dataset: CIFAR-10 or similar

Deliverables:
- Working training script
- Trained model achieving >85% test accuracy
- Training curves and analysis
- Confusion matrix and per-class metrics
```

---

### Project 2: Sequence-to-Sequence Translation

```
Task: Build a neural machine translation system

Requirements:
1. Implement tokenizer (Module 6)
2. Build encoder-decoder with attention (Modules 4, 5)
3. Training with teacher forcing
4. Inference with beam search
5. BLEU score evaluation

Dataset: English-French pairs (small subset of WMT)

Deliverables:
- Working translation system
- Attention visualization
- Analysis of translation quality
```

---

### Project 3: Build Your Own GPT

```
Task: Train a character-level language model

Requirements:
1. Implement Transformer from scratch (Module 5)
2. Training loop with proper batching
3. Text generation with temperature sampling
4. Perplexity evaluation

Dataset: Shakespeare, or any text corpus >1MB

Deliverables:
- Working language model
- Generated samples at different temperatures
- Loss curves and analysis
```

---

### Project 4: RAG-Powered QA System

```
Task: Build a question-answering system over your own documents

Requirements:
1. Document processing and chunking (Module 8)
2. Embedding and vector storage
3. Retrieval with semantic search
4. Answer generation with context
5. Evaluation on test questions

Dataset: Any collection of documents (PDFs, web pages, etc.)

Deliverables:
- Working QA system
- Retrieval quality metrics
- Example Q&A sessions
- Analysis of failure cases
```

---

### Project 5: Train and Align Your Own LLM

```
Task: Fine-tune and align a small LLM

Requirements:
1. Start with a small pre-trained model (TinyLlama, GPT-2)
2. Supervised fine-tuning on instruction data
3. Implement LoRA for efficient training
4. RLHF or DPO for alignment
5. Evaluation on safety and helpfulness

Dataset: Alpaca, Dolly, or custom instruction data

Deliverables:
- Fine-tuned model
- Training metrics and analysis
- Evaluation results
- Example conversations
```

---

## Solutions Guide

### Solution 1.3: Manual Forward Pass

```
Layer 1 (before activation):
z1 = W1 @ x + b1
   = [[0.1, 0.2],    @ [1]  + [0.1]
      [0.3, 0.4],      [2]    [0.1]
      [0.5, 0.6]]              [0.1]

   = [[0.1*1 + 0.2*2],   + [0.1]
      [0.3*1 + 0.4*2],     [0.1]
      [0.5*1 + 0.6*2]]     [0.1]

   = [[0.5],    + [0.1]    = [0.6]
      [1.1],      [0.1]      [1.2]
      [1.7]]      [0.1]      [1.8]

Layer 1 (after ReLU):
h1 = ReLU(z1) = [0.6, 1.2, 1.8]  (all positive, unchanged)

Layer 2:
y = W2 @ h1 + b2
  = [[0.1, 0.2, 0.3],   @ [0.6]  + [0]
     [0.4, 0.5, 0.6]]     [1.2]    [0]
                          [1.8]

  = [[0.1*0.6 + 0.2*1.2 + 0.3*1.8],  + [0]
     [0.4*0.6 + 0.5*1.2 + 0.6*1.8]]    [0]

  = [[0.06 + 0.24 + 0.54],  = [0.84]
     [0.24 + 0.60 + 1.08]]    [1.92]
```

---

### Solution 3.1: Convolution Output Size

Formula: Output = floor((Input + 2*Padding - Kernel) / Stride) + 1

a) (32 + 0 - 3) / 1 + 1 = 30
b) (32 + 4 - 5) / 1 + 1 = 32  (same padding)
c) (32 + 2 - 3) / 2 + 1 = 16
d) (224 + 6 - 7) / 2 + 1 = 112

---

### Solution 5.3: Manual Attention

```python
import numpy as np

Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1], [1, 1]])
V = np.array([[1, 2], [3, 4], [5, 6]])

# a) Attention scores
scores = Q @ K.T
# [[1*1+0*0, 1*0+0*1, 1*1+0*1],
#  [0*1+1*0, 0*0+1*1, 0*1+1*1]]
# = [[1, 0, 1],
#    [0, 1, 1]]

# b) Scaled scores (d_k = 2)
scaled = scores / np.sqrt(2)
# = [[0.707, 0, 0.707],
#    [0, 0.707, 0.707]]

# c) Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

weights = softmax(scaled)
# Row 1: [0.422, 0.156, 0.422]
# Row 2: [0.156, 0.422, 0.422]

# d) Output
output = weights @ V
# Row 1: 0.422*[1,2] + 0.156*[3,4] + 0.422*[5,6] = [3.0, 4.0]
# Row 2: 0.156*[1,2] + 0.422*[3,4] + 0.422*[5,6] = [3.53, 4.53]
```

---

*More solutions available in the instructor's guide.*

---

## References

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (He et al., 2015)
- [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, 1997)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
- [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [RAG](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)

### Books
- [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow, Bengio, Courville)
- [Dive into Deep Learning](https://d2l.ai/)
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) (Jurafsky & Martin)

### Online Courses
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [CS224n: NLP with Deep Learning](http://cs224n.stanford.edu/)
- [fast.ai Practical Deep Learning](https://course.fast.ai/)

---

*Last updated: January 2026*
