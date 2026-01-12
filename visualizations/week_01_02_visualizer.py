"""
Week 1-2 Visualizer: Deep Learning Foundations
Interactive ASCII visualizations for neural network concepts.
Run: python week_01_02_visualizer.py
"""

import time
import os
import random
import math

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
# VISUALIZATION 1: Single Neuron / Perceptron
# ============================================================

def visualize_neuron():
    print_header("VISUALIZATION 1: The Neuron (Perceptron)")

    print("""
The neuron is the basic unit of a neural network.
It takes inputs, multiplies by weights, adds bias, applies activation.

    Formula: output = activation(w1*x1 + w2*x2 + w3*x3 + bias)
    """)
    pause()

    print("""
Step 1: INPUTS (x1, x2, x3)
=========================================

These are the features or data coming into the neuron.

        x1 = 0.5   ──┐
                     │
        x2 = 0.8   ──┼──▶ [NEURON]
                     │
        x3 = 0.2   ──┘
    """)
    pause()

    print("""
Step 2: WEIGHTS (w1, w2, w3)
=========================================

Each connection has a weight - how important is this input?

        x1 = 0.5  ──(w1=0.4)──┐
                               │
        x2 = 0.8  ──(w2=0.6)──┼──▶ [NEURON]
                               │
        x3 = 0.2  ──(w3=0.2)──┘

Calculation:
  0.5 * 0.4 = 0.20
  0.8 * 0.6 = 0.48
  0.2 * 0.2 = 0.04
             ─────
  Sum       = 0.72
    """)
    pause()

    print("""
Step 3: BIAS (b)
=========================================

Bias shifts the activation threshold.

        Weighted Sum = 0.72
        Bias         = 0.1
                      ─────
        Total        = 0.82

        ┌──────────────────────────────────────┐
        │  x1──(w1)──┐                         │
        │            ├─→ Σ ──(+b)──→ σ ──→ y  │
        │  x2──(w2)──┤                         │
        │            │                         │
        │  x3──(w3)──┘                         │
        └──────────────────────────────────────┘
    """)
    pause()

    print("""
Step 4: ACTIVATION FUNCTION (σ)
=========================================

Non-linear function that determines output.

Common activations:

    Sigmoid:  σ(x) = 1 / (1 + e^(-x))

         1 │      ┌───────────
           │     ╱
       0.5│────╱────────────
           │  ╱
         0 │──────────────────
           -4   0   4

    ReLU:    f(x) = max(0, x)

           │       ╱
           │      ╱
         0 │─────╱
           │
           └──────────────────
              0

For our neuron:
  σ(0.82) = 1 / (1 + e^(-0.82)) = 0.69

OUTPUT: 0.69
    """)
    pause()

# ============================================================
# VISUALIZATION 2: Multi-Layer Network (Forward Pass)
# ============================================================

def visualize_forward_pass():
    print_header("VISUALIZATION 2: Forward Pass Through a Network")

    print("""
A neural network is layers of neurons connected together.

Network Architecture: 2 inputs → 3 hidden → 2 outputs

    INPUT       HIDDEN         OUTPUT
    LAYER       LAYER          LAYER

     x1 ────┬────○─────┬────── y1
            │    ○     │
     x2 ────┴────○─────┴────── y2

     [2]        [3]          [2]
    """)
    pause()

    frames = [
        """
FORWARD PASS: Step 1 - Data enters input layer
══════════════════════════════════════════════════════════

     INPUT       HIDDEN         OUTPUT
     LAYER       LAYER          LAYER

    [0.5]───┬────○─────┬────── ?
            │    ○     │
    [0.8]───┴────○─────┴────── ?

    ▲▲▲▲
    Data enters here
        """,
        """
FORWARD PASS: Step 2 - Compute hidden layer activations
══════════════════════════════════════════════════════════

     INPUT       HIDDEN         OUTPUT
     LAYER       LAYER          LAYER

    [0.5]═══╦═══[0.62]═══╦════ ?
            ║   [0.71]   ║
    [0.8]═══╩═══[0.45]═══╩════ ?

            ▲▲▲▲▲▲▲▲▲
            σ(Wx + b) computed
        """,
        """
FORWARD PASS: Step 3 - Compute output layer
══════════════════════════════════════════════════════════

     INPUT       HIDDEN         OUTPUT
     LAYER       LAYER          LAYER

    [0.5]═══╦═══[0.62]═══╦════[0.73]
            ║   [0.71]   ║
    [0.8]═══╩═══[0.45]═══╩════[0.28]

                         ▲▲▲▲▲▲▲▲
                         Final output!
        """
    ]

    for frame in frames:
        clear_screen()
        print("=" * 60)
        print(f"{'FORWARD PASS ANIMATION':^60}")
        print("=" * 60)
        print(frame)
        time.sleep(1.5)

    pause()

# ============================================================
# VISUALIZATION 3: Gradient Descent
# ============================================================

def visualize_gradient_descent():
    print_header("VISUALIZATION 3: Gradient Descent Optimization")

    print("""
Gradient Descent: Finding the minimum of the loss function.

Imagine a ball rolling down a hill to find the lowest point.

                    Loss
                      │
                  ○   │      ← Start here (high loss)
                   ╲  │
                    ╲ │
                     ╲│
                      ╲
                       ╲
                        ●    ← Goal: minimum loss
                      ──┴────────────────
                              weights
    """)
    pause()

    # Animated gradient descent
    print_header("VISUALIZATION 3: Gradient Descent (Animated)")

    steps = [
        (0, 10.0, "Start: random weights, high loss"),
        (1, 7.2, "Step 1: compute gradient, update weights"),
        (2, 5.1, "Step 2: loss decreasing..."),
        (3, 3.4, "Step 3: getting closer..."),
        (4, 2.1, "Step 4: approaching minimum..."),
        (5, 1.2, "Step 5: almost there..."),
        (6, 0.5, "Step 6: converged!"),
    ]

    for step, loss, msg in steps:
        clear_screen()
        print("=" * 60)
        print(f"{'GRADIENT DESCENT ANIMATION':^60}")
        print("=" * 60)

        # Draw loss landscape
        print("\n    Loss")
        print("      │")

        height = 10
        for h in range(height, 0, -1):
            line = "      │"
            # Draw the curve
            for x in range(40):
                # Parabola shape: loss = (x-20)^2 / 40
                y = ((x - 20) ** 2) / 40
                if abs(y - h) < 0.5:
                    line += "·"
                else:
                    line += " "

            # Draw the ball position
            ball_x = int(step * 5 + 5)
            ball_h = ((ball_x - 20) ** 2) / 40
            if abs(ball_h - h) < 0.5 and ball_x < 35:
                line = line[:7 + ball_x] + "○" + line[8 + ball_x:]

            print(line)

        print("      └" + "─" * 40 + "→ weights")
        print(f"\n    Step: {step}  |  Loss: {loss:.2f}")
        print(f"    {msg}")

        time.sleep(0.8)

    pause()

    print("""
KEY CONCEPTS:

1. LEARNING RATE (α):
   - Too small: slow convergence
   - Too large: overshooting

   weights_new = weights_old - α * gradient

                Small α           Large α
              ○→○→○→○→●         ○──────○
                                      ╲╱
                                       ○ (oscillating)

2. THE GRADIENT:
   - Direction of steepest increase
   - We go OPPOSITE direction (descent)
   - ∂Loss/∂weight tells us how to adjust

3. STOCHASTIC vs BATCH:
   - Batch: use all data (smooth but slow)
   - Stochastic: one sample (noisy but fast)
   - Mini-batch: best of both worlds
    """)
    pause()

# ============================================================
# VISUALIZATION 4: Backpropagation
# ============================================================

def visualize_backprop():
    print_header("VISUALIZATION 4: Backpropagation")

    print("""
Backpropagation: Computing gradients layer by layer.

The chain rule allows us to compute how each weight
affects the final loss.

    ∂Loss   ∂Loss   ∂output   ∂hidden
    ───── = ───── × ─────── × ────────
    ∂w1     ∂output ∂hidden   ∂w1
    """)
    pause()

    frames = [
        """
BACKPROP Step 1: Compute output error
══════════════════════════════════════════════════════════

     INPUT       HIDDEN         OUTPUT        TARGET

    [0.5]───┬────[h1]────┬────[0.73]   vs   [1.0]
            │    [h2]    │
    [0.8]───┴────[h3]────┴────[0.28]   vs   [0.0]

                                  ◄═══════════
                                  Error = Target - Output
                                  e1 = 1.0 - 0.73 = 0.27
                                  e2 = 0.0 - 0.28 = -0.28
        """,
        """
BACKPROP Step 2: Propagate error to hidden layer
══════════════════════════════════════════════════════════

     INPUT       HIDDEN         OUTPUT        TARGET

    [0.5]───┬────[h1]◄═══┬════[0.73]   vs   [1.0]
            │    [h2]◄═══│
    [0.8]───┴────[h3]◄═══┴════[0.28]   vs   [0.0]

                 ◄═══════════════════
                 δ_hidden = δ_output × W × σ'(z)
        """,
        """
BACKPROP Step 3: Compute weight gradients
══════════════════════════════════════════════════════════

     INPUT       HIDDEN         OUTPUT        TARGET

    [0.5]◄══╦════[h1]◄═══╦════[0.73]   vs   [1.0]
            ║    [h2]◄═══║
    [0.8]◄══╩════[h3]◄═══╩════[0.28]   vs   [0.0]

    ◄════════════════════════════════
    ∂L/∂W = δ × input
    Now we know how to update ALL weights!
        """,
        """
BACKPROP Step 4: Update weights
══════════════════════════════════════════════════════════

     INPUT       HIDDEN         OUTPUT

    [0.5]═══╦════[h1]════╦════ UPDATED
            ║    [h2]    ║     WEIGHTS!
    [0.8]═══╩════[h3]════╩════

    W_new = W_old - learning_rate × gradient

    ✓ All weights adjusted to reduce loss
    ✓ Repeat for next batch of data
        """
    ]

    for frame in frames:
        clear_screen()
        print("=" * 60)
        print(f"{'BACKPROPAGATION ANIMATION':^60}")
        print("=" * 60)
        print(frame)
        time.sleep(2)

    pause()

# ============================================================
# VISUALIZATION 5: Activation Functions
# ============================================================

def visualize_activations():
    print_header("VISUALIZATION 5: Activation Functions")

    print("""
Why do we need activation functions?

Without them, neural networks are just linear transformations!

    Layer1: y = W1*x + b1
    Layer2: y = W2*(W1*x + b1) + b2
          = W2*W1*x + W2*b1 + b2
          = W'*x + b'  ← Still linear!

Activation functions add non-linearity.
    """)
    pause()

    print("""
SIGMOID: σ(x) = 1 / (1 + e^(-x))
═══════════════════════════════════════════════════════════

Range: (0, 1)  |  Good for: probability outputs

    1.0 │                    ╭───────────
        │                  ╱
        │                ╱
    0.5 │──────────────╱────────────────
        │            ╱
        │          ╱
    0.0 │─────────╯
        └────────────────────────────────
          -6    -4    -2    0    2    4    6

Problems:
  - Vanishing gradients (saturates at 0 and 1)
  - Outputs not centered at zero
  - exp() is expensive
    """)
    pause()

    print("""
TANH: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
═══════════════════════════════════════════════════════════

Range: (-1, 1)  |  Good for: centered outputs

    1.0 │                    ╭───────────
        │                  ╱
    0.0 │────────────────╱──────────────
        │              ╱
        │            ╱
   -1.0 │───────────╯
        └────────────────────────────────
          -6    -4    -2    0    2    4    6

Better than sigmoid:
  ✓ Zero-centered
  ✗ Still has vanishing gradient problem
    """)
    pause()

    print("""
ReLU: f(x) = max(0, x)
═══════════════════════════════════════════════════════════

Range: [0, ∞)  |  Good for: hidden layers (most popular!)

        │                    ╱
        │                  ╱
        │                ╱
        │              ╱
    0   │────────────╱
        │
        └────────────────────────────────
          -4    -2    0    2    4    6

Advantages:
  ✓ No vanishing gradient for positive values
  ✓ Computationally cheap
  ✓ Sparse activation (biological plausibility)

Problems:
  ✗ "Dead ReLU" - neurons can get stuck at 0
    """)
    pause()

    print("""
COMPARISON TABLE
═══════════════════════════════════════════════════════════

Function   │ Range     │ Centered │ Gradient │ Speed
───────────┼───────────┼──────────┼──────────┼────────
Sigmoid    │ (0, 1)    │ No       │ Vanishes │ Slow
Tanh       │ (-1, 1)   │ Yes      │ Vanishes │ Slow
ReLU       │ [0, ∞)    │ No       │ OK*      │ Fast
LeakyReLU  │ (-∞, ∞)   │ No       │ Good     │ Fast
GELU       │ (-0.17,∞) │ ~Yes     │ Good     │ Medium

*ReLU gradient = 0 for negative inputs (dead neurons)

RECOMMENDATION:
  - Hidden layers: ReLU or GELU
  - Output (classification): Softmax
  - Output (probability): Sigmoid
  - Output (regression): None/Linear
    """)
    pause()

# ============================================================
# VISUALIZATION 6: Loss Functions
# ============================================================

def visualize_loss_functions():
    print_header("VISUALIZATION 6: Loss Functions")

    print("""
Loss functions measure how wrong our predictions are.

REGRESSION: Mean Squared Error (MSE)
═══════════════════════════════════════════════════════════

    MSE = (1/n) Σ (y_true - y_pred)²

    Example:
    ┌──────────┬──────────┬─────────────┬────────────┐
    │ y_true   │ y_pred   │ Error       │ Error²     │
    ├──────────┼──────────┼─────────────┼────────────┤
    │ 3.0      │ 2.5      │ 0.5         │ 0.25       │
    │ 5.0      │ 4.8      │ 0.2         │ 0.04       │
    │ 2.0      │ 2.3      │ -0.3        │ 0.09       │
    └──────────┴──────────┴─────────────┴────────────┘
                                    MSE = 0.38/3 = 0.127

    Why squared?
    - Penalizes large errors more
    - Always positive
    - Smooth gradient
    """)
    pause()

    print("""
CLASSIFICATION: Cross-Entropy Loss
═══════════════════════════════════════════════════════════

Binary Cross-Entropy:
    L = -[y*log(p) + (1-y)*log(1-p)]

    where y = true label (0 or 1)
          p = predicted probability

    ┌────────────────────────────────────────────────┐
    │  True: 1  │  If p = 0.9  │  L = -log(0.9) = 0.1│
    │  True: 1  │  If p = 0.1  │  L = -log(0.1) = 2.3│ ← Bad!
    │  True: 0  │  If p = 0.1  │  L = -log(0.9) = 0.1│
    └────────────────────────────────────────────────┘

    Loss explodes when confident AND wrong!

    Loss │
         │╲
         │ ╲
         │  ╲
         │   ╲____
       0 │        ────────────
         └────────────────────
           0    0.5    1.0
                Correct class probability
    """)
    pause()

    print("""
MULTI-CLASS: Categorical Cross-Entropy
═══════════════════════════════════════════════════════════

    L = -Σ y_i * log(p_i)   for all classes

    Example: 3-class classification (cat, dog, bird)

    True label:    [1, 0, 0]  (it's a cat)
    Predictions:   [0.7, 0.2, 0.1]

    Loss = -[1*log(0.7) + 0*log(0.2) + 0*log(0.1)]
         = -log(0.7)
         = 0.36

    Only the TRUE class matters in the sum!

    ┌─────────────────────────────────────────────────┐
    │  Good prediction (cat=0.9):    L = 0.11        │
    │  Bad prediction (cat=0.3):     L = 1.20        │
    │  Terrible prediction (cat=0.01): L = 4.61      │
    └─────────────────────────────────────────────────┘
    """)
    pause()

# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print_header("WEEK 1-2: DEEP LEARNING FOUNDATIONS")
        print("""
Choose a visualization:

    [1] The Neuron (Perceptron)
        - Inputs, weights, bias, activation

    [2] Forward Pass
        - Data flowing through the network

    [3] Gradient Descent
        - Finding the minimum loss (animated)

    [4] Backpropagation
        - Computing gradients layer by layer

    [5] Activation Functions
        - Sigmoid, Tanh, ReLU comparison

    [6] Loss Functions
        - MSE, Cross-Entropy explained

    [A] Run ALL visualizations

    [Q] Quit
        """)

        choice = input("Enter choice: ").strip().upper()

        if choice == '1':
            visualize_neuron()
        elif choice == '2':
            visualize_forward_pass()
        elif choice == '3':
            visualize_gradient_descent()
        elif choice == '4':
            visualize_backprop()
        elif choice == '5':
            visualize_activations()
        elif choice == '6':
            visualize_loss_functions()
        elif choice == 'A':
            visualize_neuron()
            visualize_forward_pass()
            visualize_gradient_descent()
            visualize_backprop()
            visualize_activations()
            visualize_loss_functions()
        elif choice == 'Q':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
