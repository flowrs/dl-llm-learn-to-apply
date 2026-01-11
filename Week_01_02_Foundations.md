# Week 1-2: Deep Learning Foundations
## From Novice to Practitioner: Classification, Linear Models, and Optimization

---

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Image Classification Fundamentals](#image-classification-fundamentals)
3. [K-Nearest Neighbors](#k-nearest-neighbors)
4. [Linear Classifiers](#linear-classifiers)
5. [Loss Functions](#loss-functions)
6. [Optimization & Gradient Descent](#optimization--gradient-descent)
7. [Neural Networks](#neural-networks)
8. [Backpropagation](#backpropagation)
9. [Coding Exercises](#coding-exercises)
10. [Business Applications](#business-applications)

---

## The Big Picture

### What is Deep Learning?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE MACHINE LEARNING SPECTRUM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Traditional Programming          Machine Learning          Deep Learning  │
│                                                                             │
│   ┌──────────┐                    ┌──────────┐              ┌──────────┐   │
│   │  Rules   │                    │   Data   │              │   Data   │   │
│   │    +     │ ──► Output         │    +     │ ──► Rules    │    +     │   │
│   │   Data   │                    │  Output  │              │  Output  │   │
│   └──────────┘                    └──────────┘              └──────────┘   │
│                                                                    │        │
│   Human writes                    Algorithm learns               ▼        │
│   explicit rules                  simple patterns         ┌──────────┐    │
│                                                           │  Neural  │    │
│   Example:                        Example:                │ Networks │    │
│   if pixel>128:                   Decision Trees          │ learn    │    │
│     white                         Linear Regression       │ complex  │    │
│   else:                           SVM                     │ features │    │
│     black                                                 └──────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Deep Learning Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DEEP LEARNING PIPELINE                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │  DATA   │───►│ FORWARD │───►│  LOSS   │───►│BACKWARD │───►│ UPDATE  │ │
│  │         │    │  PASS   │    │         │    │  PASS   │    │ WEIGHTS │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│       │              │              │              │              │       │
│       ▼              ▼              ▼              ▼              ▼       │
│   Images &      Predictions     How wrong      Gradients      Improve    │
│   Labels        from model      are we?        (blame)        model      │
│                                                                           │
│  ◄──────────────────── REPEAT FOR MANY ITERATIONS ─────────────────────► │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Image Classification Fundamentals

### The Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMAGE CLASSIFICATION TASK                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: Raw Pixels                              OUTPUT: Category Label     │
│                                                                             │
│   ┌───────────────────┐                          ┌───────────────────┐     │
│   │ 142  98  67  ...  │                          │                   │     │
│   │ 156 102  71  ...  │      ───────────►        │      "CAT"        │     │
│   │ 163 108  78  ...  │       Classifier         │   Confidence: 92% │     │
│   │ ...  ...  ... ... │                          │                   │     │
│   └───────────────────┘                          └───────────────────┘     │
│        32x32x3 = 3072 numbers                                               │
│                                                                             │
│   CHALLENGES:                                                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │  Viewpoint  │  │   Scale     │  │ Deformation │  │  Occlusion  │      │
│   │  Variation  │  │  Variation  │  │             │  │             │      │
│   │    ┌─┐      │  │   ┌─┐ ┌───┐│  │    /\       │  │   ┌─┐       │      │
│   │   /  \      │  │   │ │ │   ││  │   /  \      │  │   │X│       │      │
│   │  front/side │  │   │ │ │   ││  │  (bent)     │  │   └─┘       │      │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data-Driven Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA-DRIVEN APPROACH                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TRAINING PHASE:                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │  Training Data                                                      │  │
│   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                          │  │
│   │  │ 🐱  │ │ 🐕  │ │ 🚗  │ │ 🐱  │ │ 🚗  │  ...  (50,000 images)    │  │
│   │  │ cat │ │ dog │ │ car │ │ cat │ │ car │                          │  │
│   │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                          │  │
│   │           │                                                        │  │
│   │           ▼                                                        │  │
│   │    ┌─────────────┐                                                 │  │
│   │    │   TRAIN     │ ──────► Model (learned parameters)              │  │
│   │    │  FUNCTION   │                                                 │  │
│   │    └─────────────┘                                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   PREDICTION PHASE:                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │  New Image      Model                   Prediction                  │  │
│   │  ┌─────┐    ┌─────────┐               ┌─────────┐                  │  │
│   │  │ 🐱  │ ──►│ PREDICT │ ─────────────►│  "cat"  │                  │  │
│   │  │  ?  │    │FUNCTION │               │  (87%)  │                  │  │
│   │  └─────┘    └─────────┘               └─────────┘                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## K-Nearest Neighbors

### Concept

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        K-NEAREST NEIGHBORS (k-NN)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IDEA: Find the k most similar training images, let them vote              │
│                                                                             │
│   2D Visualization (each point = image in feature space):                   │
│                                                                             │
│        k=1                        k=3                       k=5             │
│                                                                             │
│     ▲                          ▲                          ▲                 │
│     │  ○ ○                     │  ○ ○                     │  ○ ○            │
│     │    ○ ●                   │    ○ ●                   │    ○ ●          │
│     │  ○   ╳ ●                 │  ○   ╳ ●                 │  ○   ╳ ●        │
│     │      ● ●                 │      ● ●                 │      ● ●        │
│     │        ●                 │        ●                 │        ●        │
│     └──────────►               └──────────►               └──────────►      │
│                                                                             │
│   ╳ = test point               ╳ = test point             ╳ = test point    │
│   Nearest: ●                   3 nearest: ●●○             5 nearest: ●●●○○  │
│   Predict: ●                   Vote: ● wins (2-1)         Vote: ● wins (3-2)│
│                                                                             │
│   ○ = class A (e.g., cats)                                                  │
│   ● = class B (e.g., dogs)                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Distance Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DISTANCE METRICS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   L1 Distance (Manhattan):              L2 Distance (Euclidean):            │
│                                                                             │
│   d₁(I₁,I₂) = Σ|I₁ᵖ - I₂ᵖ|             d₂(I₁,I₂) = √(Σ(I₁ᵖ - I₂ᵖ)²)       │
│                                                                             │
│        ┌───────B                              ┌───────B                     │
│        │       │                              │      /                      │
│        │       │ L1 path                      │     / L2 path               │
│        │       │ (along grid)                 │    /  (straight)            │
│        A───────┘                              A───┘                         │
│                                                                             │
│   L1 = |x₂-x₁| + |y₂-y₁|                 L2 = √((x₂-x₁)² + (y₂-y₁)²)       │
│                                                                             │
│   PIXEL COMPARISON EXAMPLE:                                                 │
│   ┌─────────────┐    ┌─────────────┐                                       │
│   │ 56  128  34 │    │ 60  120  40 │                                       │
│   │ 200  45  89 │    │ 195  50  85 │                                       │
│   └─────────────┘    └─────────────┘                                       │
│       Image 1            Image 2                                            │
│                                                                             │
│   L1 = |56-60| + |128-120| + |34-40| + |200-195| + |45-50| + |89-85|       │
│      = 4 + 8 + 6 + 5 + 5 + 4 = 32                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
#==============================================================================
# EXERCISE 1.1: K-Nearest Neighbors Implementation
#==============================================================================

import numpy as np

class KNearestNeighbor:
    """
    K-Nearest Neighbor classifier with L1 and L2 distance metrics.

    This is a non-parametric, instance-based learning algorithm.
    - Non-parametric: No assumptions about data distribution
    - Instance-based: Stores all training data, no explicit training phase
    """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Train the classifier (just memorize the training data).

        Args:
            X: Training data of shape (N, D) where N is number of samples
               and D is the dimensionality (e.g., 3072 for 32x32x3 images)
            y: Training labels of shape (N,)
        """
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):
        """
        Compute distances using explicit loops (slow but clear).

        EXERCISE: Understand this first before optimizing.

        Args:
            X: Test data of shape (num_test, D)

        Returns:
            dists: Array of shape (num_test, num_train) where dists[i,j]
                   is the L2 distance between test point i and train point j
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                # L2 distance: sqrt(sum((x - y)^2))
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute distances with one loop (vectorized over training data).

        EXERCISE: Fill in this function to match compute_distances_two_loops
        but using only one explicit loop.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            #===================================================================
            # TODO: Compute the L2 distance between the i-th test point and
            # all training points without using a second loop.
            # Hint: Use broadcasting. X[i] has shape (D,), self.X_train has
            # shape (num_train, D). Their difference broadcasts to (num_train, D)
            #===================================================================
            dists[i, :] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
            #===================================================================

        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute distances with no explicit loops (fully vectorized).

        EXERCISE: This is the most challenging. Use the identity:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y

        This allows us to use matrix multiplication which is highly optimized.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        #=======================================================================
        # TODO: Compute the L2 distance between all test points and all training
        # points without using any explicit loops.
        #
        # Hint: Expand (x - y)^2 = x^2 + y^2 - 2xy
        # - Compute ||X_test||^2 for each test point: shape (num_test, 1)
        # - Compute ||X_train||^2 for each train point: shape (1, num_train)
        # - Compute X_test @ X_train.T: shape (num_test, num_train)
        #=======================================================================

        # ||test||^2: sum of squares for each test point
        test_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (num_test, 1)

        # ||train||^2: sum of squares for each training point
        train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T  # (1, num_train)

        # -2 * test · train
        cross_term = -2 * X @ self.X_train.T  # (num_test, num_train)

        # Combine: ||test - train||^2 = ||test||^2 + ||train||^2 - 2*test·train
        dists = np.sqrt(test_sq + train_sq + cross_term)

        #=======================================================================

        return dists

    def predict(self, X, k=1, distance_metric='L2'):
        """
        Predict labels for test data.

        Args:
            X: Test data of shape (num_test, D)
            k: Number of nearest neighbors to use
            distance_metric: 'L1' or 'L2'

        Returns:
            y_pred: Predicted labels of shape (num_test,)
        """
        if distance_metric == 'L2':
            dists = self.compute_distances_no_loops(X)
        else:
            # L1 distance implementation
            dists = np.sum(np.abs(X[:, np.newaxis] - self.X_train), axis=2)

        return self.predict_labels(dists, k)

    def predict_labels(self, dists, k=1):
        """
        Given distance matrix, predict labels using k nearest neighbors.

        EXERCISE: Implement the voting mechanism.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            #===================================================================
            # TODO:
            # 1. Find the k nearest neighbors (indices with smallest distances)
            # 2. Get their labels
            # 3. Find the most common label (majority vote)
            #===================================================================

            # Get indices of k smallest distances
            nearest_indices = np.argsort(dists[i])[:k]

            # Get labels of k nearest neighbors
            nearest_labels = self.y_train[nearest_indices]

            # Majority vote: find most common label
            # np.bincount counts occurrences of each integer
            # np.argmax returns the integer with highest count
            y_pred[i] = np.bincount(nearest_labels).argmax()

            #===================================================================

        return y_pred


#==============================================================================
# EXERCISE 1.2: Cross-Validation for Hyperparameter Tuning
#==============================================================================

def cross_validation_knn(X_train, y_train, k_values, num_folds=5):
    """
    Perform k-fold cross-validation to find the best k for k-NN.

    Args:
        X_train: Training data of shape (N, D)
        y_train: Training labels of shape (N,)
        k_values: List of k values to try
        num_folds: Number of folds for cross-validation

    Returns:
        k_to_accuracies: Dictionary mapping k values to list of accuracies
    """
    N = X_train.shape[0]
    fold_size = N // num_folds

    # Shuffle data (optional but good practice)
    indices = np.random.permutation(N)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    k_to_accuracies = {k: [] for k in k_values}

    for fold in range(num_folds):
        # Split into validation and training for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        X_val_fold = X_shuffled[val_start:val_end]
        y_val_fold = y_shuffled[val_start:val_end]

        # Training data is everything else
        X_train_fold = np.concatenate([
            X_shuffled[:val_start],
            X_shuffled[val_end:]
        ])
        y_train_fold = np.concatenate([
            y_shuffled[:val_start],
            y_shuffled[val_end:]
        ])

        # Train classifier
        classifier = KNearestNeighbor()
        classifier.train(X_train_fold, y_train_fold)

        # Evaluate for each k
        for k in k_values:
            y_pred = classifier.predict(X_val_fold, k=k)
            accuracy = np.mean(y_pred == y_val_fold)
            k_to_accuracies[k].append(accuracy)

    # Print results
    print("Cross-validation results:")
    print("-" * 40)
    for k in k_values:
        accuracies = k_to_accuracies[k]
        print(f"k = {k:2d}: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")

    return k_to_accuracies
```

---

## Linear Classifiers

### The Score Function

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LINEAR CLASSIFIER: SCORE FUNCTION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   f(x, W, b) = Wx + b                                                       │
│                                                                             │
│   For CIFAR-10 (10 classes, 32x32x3 images):                               │
│                                                                             │
│   ┌─────────────┐   ┌───────────────────────┐   ┌─────┐   ┌─────────────┐  │
│   │             │   │                       │   │     │   │             │  │
│   │   Image x   │   │      Weights W        │   │  b  │   │   Scores    │  │
│   │             │   │                       │   │     │   │             │  │
│   │  (3072x1)   │ × │     (10 x 3072)       │ + │(10) │ = │    (10)     │  │
│   │             │   │                       │   │     │   │             │  │
│   │ [0.2, 0.1,  │   │ [w₀₀ w₀₁ ... w₀,₃₀₇₁]│   │[b₀] │   │ [cat: 3.2] │  │
│   │  0.4, ...]  │   │ [w₁₀ w₁₁ ... w₁,₃₀₇₁]│   │[b₁] │   │ [dog: 1.5] │  │
│   │             │   │ [... ... ... ...     ]│   │[...│   │ [car: 4.1] │  │
│   │             │   │ [w₉₀ w₉₁ ... w₉,₃₀₇₁]│   │[b₉] │   │ [... ]     │  │
│   └─────────────┘   └───────────────────────┘   └─────┘   └─────────────┘  │
│                                                                             │
│   Each row of W is a TEMPLATE for one class:                               │
│                                                                             │
│   Row 0 (cat template):    Row 1 (dog template):   Row 2 (car template):   │
│   ┌─────────────────┐      ┌─────────────────┐     ┌─────────────────┐     │
│   │  Learned "cat"  │      │  Learned "dog"  │     │  Learned "car"  │     │
│   │  pattern from   │      │  pattern from   │     │  pattern from   │     │
│   │  training data  │      │  training data  │     │  training data  │     │
│   └─────────────────┘      └─────────────────┘     └─────────────────┘     │
│                                                                             │
│   Score = How well image matches each template (dot product similarity)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Geometric Interpretation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              LINEAR CLASSIFIER: GEOMETRIC INTERPRETATION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Each class defines a HYPERPLANE in pixel space:                          │
│                                                                             │
│   2D Example (2 pixels, 3 classes):                                        │
│                                                                             │
│        pixel 2                                                              │
│           ▲                                                                 │
│           │     CLASS C                                                     │
│           │    ╱                                                            │
│           │   ╱                                                             │
│           │  ╱  W·x + b = 0 (decision boundary)                            │
│       ────┼─╱──────────────────►                                           │
│           │╱        CLASS A     pixel 1                                     │
│          ╱│                                                                 │
│         ╱ │                                                                 │
│        ╱  │  CLASS B                                                        │
│       ╱   │                                                                 │
│                                                                             │
│   - Weight vector W determines the ORIENTATION of the boundary             │
│   - Bias b determines the OFFSET from origin                               │
│   - Points on one side: positive score, other side: negative score         │
│                                                                             │
│   LIMITATION: Linear classifiers can only draw straight lines!             │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │  ● ● │ ○ ○      │  │  ●     ○        │  │    ●            │           │
│   │  ● ● │ ○ ○      │  │    ○ ● ○        │  │  ●   ●          │           │
│   │  ● ● │ ○ ○      │  │  ●     ○        │  │    ●            │           │
│   │      │          │  │                 │  │                 │           │
│   │   SOLVABLE!     │  │  NOT SOLVABLE   │  │  NOT SOLVABLE   │           │
│   │                 │  │  (need curves)  │  │  (need curves)  │           │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Loss Functions

### SVM Loss (Hinge Loss)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SVM LOSS (HINGE LOSS)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GOAL: Correct class score should be HIGHER than others by a MARGIN       │
│                                                                             │
│   Loss for one example:                                                     │
│   L_i = Σ_{j≠y_i} max(0, s_j - s_{y_i} + Δ)                                │
│                                                                             │
│   where:                                                                    │
│   - s_j = score for class j                                                │
│   - s_{y_i} = score for correct class                                      │
│   - Δ = margin (usually 1)                                                 │
│                                                                             │
│   VISUAL EXPLANATION:                                                       │
│                                                                             │
│   Scores:   [cat: 3.2]  [dog: 5.1]  [car: -1.7]   (correct class = cat)    │
│                                                                             │
│   Loss from dog:   max(0, 5.1 - 3.2 + 1) = max(0, 2.9) = 2.9              │
│   Loss from car:   max(0, -1.7 - 3.2 + 1) = max(0, -3.9) = 0              │
│   Total L_i = 2.9 + 0 = 2.9                                                │
│                                                                             │
│                                                                             │
│   Score Line Visualization:                                                 │
│                                                                             │
│   ◄──────────────────────────────────────────────────────────────────────► │
│       -2    -1     0     1     2     3     4     5     6                   │
│                                 ├─────────────────┤                        │
│                                 │    MARGIN Δ=1   │                        │
│                                 │                 │                        │
│        car               correct class    dog                              │
│       (-1.7)                (cat: 3.2)   (5.1)                             │
│         │                       │           │                              │
│         ▼                       ▼           ▼                              │
│        ─●───────────────────────●───────────●──                            │
│                                 │←─loss=0──→│←loss=2.9→│                   │
│                                                                             │
│   Dog score VIOLATES margin → contributes loss                             │
│   Car score is SAFE → contributes 0                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Softmax Loss (Cross-Entropy)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SOFTMAX LOSS (CROSS-ENTROPY)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1: Convert scores to probabilities                                  │
│                                                                             │
│   P(class=k | x) = exp(s_k) / Σ_j exp(s_j)   (softmax function)           │
│                                                                             │
│   STEP 2: Compute negative log probability of correct class                │
│                                                                             │
│   L_i = -log(P(correct class))                                             │
│                                                                             │
│                                                                             │
│   EXAMPLE:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Scores:        [cat: 3.2]   [dog: 5.1]   [car: -1.7]             │  │
│   │                     │            │             │                    │  │
│   │                     ▼            ▼             ▼                    │  │
│   │   exp(scores):   [24.5]       [164.0]       [0.18]                 │  │
│   │                     │            │             │                    │  │
│   │                     └────────────┴─────────────┘                   │  │
│   │                               │                                     │  │
│   │                          sum = 188.7                                │  │
│   │                               │                                     │  │
│   │                     ┌─────────┴─────────┐                          │  │
│   │                     ▼                   ▼                          │  │
│   │   Probabilities: [cat: 0.13]  [dog: 0.87]  [car: 0.001]            │  │
│   │                                                                     │  │
│   │   If correct class is CAT:                                         │  │
│   │   L_i = -log(0.13) = 2.04                                          │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   KEY INSIGHT: Unlike SVM, softmax ALWAYS wants to improve                 │
│   - Even if correct class has highest score, loss > 0                      │
│   - Pushes probabilities toward 1.0 for correct class                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Regularization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            REGULARIZATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Full Loss = Data Loss + Regularization Loss                              │
│                                                                             │
│   L = (1/N) Σᵢ Lᵢ(f(xᵢ, W), yᵢ) + λR(W)                                   │
│       └──────────────┬──────────────┘   └──┬──┘                            │
│              Data Loss                 Regularization                       │
│         (fit training data)        (prevent overfitting)                   │
│                                                                             │
│   L2 REGULARIZATION (Weight Decay):                                        │
│   R(W) = Σₖ Σₗ W²ₖₗ                                                        │
│                                                                             │
│   INTUITION:                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Suppose:  x = [1, 1, 1, 1]                                       │  │
│   │                                                                     │  │
│   │   W₁ = [1, 0, 0, 0]  →  W₁·x = 1    L2 norm: √(1+0+0+0) = 1       │  │
│   │   W₂ = [0.25, 0.25, 0.25, 0.25] →  W₂·x = 1    L2 norm: √4×0.0625 = 0.5 │
│   │                                                                     │  │
│   │   Both give same score, but L2 regularization PREFERS W₂           │  │
│   │   because it spreads weights across features (more robust)         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   λ (lambda) controls the trade-off:                                       │
│                                                                             │
│      λ = 0              λ = optimal            λ = large                   │
│   ┌───────────┐       ┌───────────┐          ┌───────────┐                │
│   │ ●  ●  ●   │       │ ●  ●  ●   │          │  ●  ●  ●  │                │
│   │   ╱╲      │       │   ───     │          │   ───     │                │
│   │  ╱  ╲     │       │           │          │           │                │
│   │ ╱    ╲    │       │           │          │           │                │
│   │ Overfit   │       │ Good fit  │          │ Underfit  │                │
│   └───────────┘       └───────────┘          └───────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
#==============================================================================
# EXERCISE 2.1: Linear Classifier with SVM and Softmax Loss
#==============================================================================

import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Args:
        W: Weight matrix of shape (D, C) where D is dimension and C is classes
        X: Training data of shape (N, D)
        y: Training labels of shape (N,) with values 0..C-1
        reg: Regularization strength

    Returns:
        loss: Scalar loss value
        dW: Gradient of loss with respect to W, same shape as W
    """
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        # Compute scores for all classes
        scores = X[i] @ W  # Shape: (C,)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue  # Skip correct class

            margin = scores[j] - correct_class_score + 1  # Δ = 1

            if margin > 0:
                loss += margin
                # Gradient: when margin > 0, we have a contribution
                dW[:, j] += X[i]      # Incorrect class gets positive gradient
                dW[:, y[i]] -= X[i]   # Correct class gets negative gradient

    # Average over all training examples
    loss /= num_train
    dW /= num_train

    # Add regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    EXERCISE: This is much faster than the naive version.
    Understand how to eliminate loops using matrix operations.
    """
    num_train = X.shape[0]

    # Compute all scores at once: (N, C)
    scores = X @ W

    # Get scores of correct classes: (N,)
    correct_class_scores = scores[np.arange(num_train), y]

    # Compute margins for all classes: (N, C)
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)

    # Don't count the correct class
    margins[np.arange(num_train), y] = 0

    # Sum all margin violations
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # Gradient computation
    binary = margins > 0  # Indicator for margin violations
    binary = binary.astype(float)

    # Count how many classes contributed to loss for each example
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum

    dW = X.T @ binary / num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops).

    Args:
        W: Weight matrix of shape (D, C)
        X: Training data of shape (N, D)
        y: Training labels of shape (N,)
        reg: Regularization strength

    Returns:
        loss: Scalar loss value
        dW: Gradient of loss with respect to W
    """
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i] @ W  # Shape: (C,)

        # Numerical stability: subtract max score
        scores -= np.max(scores)

        # Compute softmax probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        # Cross-entropy loss
        loss += -np.log(probs[y[i]])

        # Gradient
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized implementation.

    EXERCISE: Implement this without loops.
    """
    num_train = X.shape[0]

    # Compute scores
    scores = X @ W  # (N, C)

    # Numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)

    # Softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Cross-entropy loss
    correct_log_probs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_log_probs) / num_train
    loss += reg * np.sum(W * W)

    # Gradient
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dscores /= num_train

    dW = X.T @ dscores
    dW += 2 * reg * W

    return loss, dW
```

---

## Optimization & Gradient Descent

### Gradient Descent Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GRADIENT DESCENT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Goal: Find weights W that minimize loss L(W)                             │
│                                                                             │
│   LOSS LANDSCAPE (simplified 2D visualization):                            │
│                                                                             │
│        Loss                                                                 │
│          ▲                                                                  │
│          │    ╱╲                                                            │
│          │   ╱  ╲                                                           │
│          │  ╱    ╲        Start here (random W)                            │
│          │ ╱      ╲           ●                                             │
│          │╱        ╲         ↓                                              │
│          │          ╲       ● ← Follow negative gradient                   │
│          │           ╲     ↓                                                │
│          │            ╲   ●                                                 │
│          │             ╲ ↓                                                  │
│          │              ●  ← Minimum (optimal W)                           │
│          └──────────────────────────────────────────► W                    │
│                                                                             │
│   UPDATE RULE:  W_new = W_old - learning_rate × gradient                   │
│                                                                             │
│   The gradient ∇L(W) points UPHILL (direction of steepest increase)        │
│   We move in the NEGATIVE gradient direction (steepest decrease)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Learning Rate Effects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEARNING RATE EFFECTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TOO SMALL                    JUST RIGHT                   TOO LARGE      │
│                                                                             │
│   Loss                         Loss                         Loss           │
│    ▲                           ▲                            ▲              │
│    │\                          │\                           │\   /\        │
│    │ \                         │ \                          │ \ /  \       │
│    │  \                        │  \                         │  X    \      │
│    │   \                       │   \                        │ / \    \     │
│    │    \                      │    \_____                  │/   \____     │
│    │     \                     │                            │              │
│    │      \___                 │                            │              │
│    └──────────► iter           └──────────► iter            └──────────► iter
│                                                                             │
│   - Converges slowly           - Converges well             - Oscillates   │
│   - May get stuck              - Reaches minimum            - May diverge  │
│   - Wastes computation         - Efficient learning         - Unstable     │
│                                                                             │
│   TYPICAL VALUES: Start with 1e-3, try {1e-4, 1e-3, 1e-2, 1e-1}           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mini-Batch Gradient Descent

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MINI-BATCH GRADIENT DESCENT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FULL BATCH:                                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │ Compute gradient using ALL 50,000 training examples                  │ │
│   │ ✗ Very slow per update                                               │ │
│   │ ✓ Accurate gradient estimate                                         │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   STOCHASTIC (batch_size = 1):                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │ Compute gradient using ONE training example                          │ │
│   │ ✓ Very fast per update                                               │ │
│   │ ✗ Very noisy gradient                                                │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   MINI-BATCH (batch_size = 32-256):                                        │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │ Compute gradient using SUBSET of training examples                   │ │
│   │ ✓ Good balance of speed and accuracy                                 │ │
│   │ ✓ Can utilize GPU parallelism efficiently                            │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   TRAINING LOOP:                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Epoch 1:                                                           │  │
│   │  [████████████████████████████████████████████████] 50,000 examples │  │
│   │   │batch│batch│batch│batch│ ... │batch│batch│batch│                │  │
│   │     1     2     3     4           195   196   197                   │  │
│   │                                                                     │  │
│   │  Each batch: 256 examples → ~195 batches per epoch                  │  │
│   │  Update weights after EACH batch (195 updates per epoch)            │  │
│   │                                                                     │  │
│   │  Epoch 2, 3, 4, ... : Repeat with shuffled data                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
#==============================================================================
# EXERCISE 3.1: Gradient Descent Optimizer
#==============================================================================

import numpy as np

class LinearClassifier:
    """
    A linear classifier with various optimization methods.
    """

    def __init__(self, input_dim, num_classes):
        """
        Initialize with small random weights.

        Args:
            input_dim: Dimension of input features (e.g., 3072 for CIFAR-10)
            num_classes: Number of output classes (e.g., 10)
        """
        # Initialize weights with small random values
        self.W = 0.0001 * np.random.randn(input_dim, num_classes)

    def loss(self, X, y, reg):
        """
        Compute loss and gradient. Override in subclass.
        """
        raise NotImplementedError

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train the classifier using mini-batch gradient descent.

        Args:
            X: Training data of shape (N, D)
            y: Training labels of shape (N,)
            learning_rate: Step size for gradient descent
            reg: Regularization strength
            num_iters: Number of iterations
            batch_size: Number of examples per mini-batch
            verbose: Print progress every 100 iterations

        Returns:
            loss_history: List of loss values at each iteration
        """
        num_train, dim = X.shape
        loss_history = []

        for it in range(num_iters):
            #===================================================================
            # TODO: Sample a mini-batch of training data and labels
            #===================================================================
            batch_indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            #===================================================================

            # Compute loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            #===================================================================
            # TODO: Update weights using gradient descent
            #===================================================================
            self.W -= learning_rate * grad
            #===================================================================

            if verbose and it % 100 == 0:
                print(f'Iteration {it}/{num_iters}: loss = {loss:.4f}')

        return loss_history

    def predict(self, X):
        """
        Predict class labels for test data.

        Args:
            X: Test data of shape (N, D)

        Returns:
            y_pred: Predicted labels of shape (N,)
        """
        scores = X @ self.W
        return np.argmax(scores, axis=1)


class SVM(LinearClassifier):
    """Linear classifier with SVM loss."""

    def loss(self, X, y, reg):
        return svm_loss_vectorized(self.W, X, y, reg)


class Softmax(LinearClassifier):
    """Linear classifier with Softmax loss."""

    def loss(self, X, y, reg):
        return softmax_loss_vectorized(self.W, X, y, reg)


#==============================================================================
# EXERCISE 3.2: Numerical Gradient Checking
#==============================================================================

def gradient_check(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    Compare analytic gradient with numerical gradient.

    This is CRITICAL for debugging. Always do this when implementing
    new gradient computations!

    Args:
        f: Function that takes x and returns scalar loss
        x: Point at which to check gradient
        analytic_grad: The analytic gradient computed at x
        num_checks: Number of random dimensions to check
        h: Step size for numerical gradient
    """
    for i in range(num_checks):
        # Pick a random dimension
        idx = tuple(np.random.randint(0, s) for s in x.shape)

        # Compute numerical gradient using central difference
        old_val = x[idx]

        x[idx] = old_val + h
        f_plus = f(x)

        x[idx] = old_val - h
        f_minus = f(x)

        x[idx] = old_val  # Reset

        numerical_grad = (f_plus - f_minus) / (2 * h)
        analytic = analytic_grad[idx]

        # Relative error
        rel_error = abs(numerical_grad - analytic) / (abs(numerical_grad) + abs(analytic) + 1e-8)

        print(f'Dimension {idx}: numerical={numerical_grad:.6f}, '
              f'analytic={analytic:.6f}, relative error={rel_error:.2e}')

        if rel_error > 1e-4:
            print('  WARNING: Large relative error!')


#==============================================================================
# EXERCISE 3.3: Learning Rate Search
#==============================================================================

def learning_rate_search(classifier_class, X_train, y_train, X_val, y_val,
                        learning_rates, reg=1e-5, num_iters=1500):
    """
    Find the best learning rate using validation set.

    Args:
        classifier_class: SVM or Softmax class
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        learning_rates: List of learning rates to try
        reg: Regularization strength
        num_iters: Number of training iterations

    Returns:
        best_lr: Best learning rate found
        results: Dictionary with accuracy for each learning rate
    """
    results = {}
    best_val_acc = 0
    best_lr = None

    for lr in learning_rates:
        print(f'\nTraining with learning_rate = {lr}')

        # Create and train classifier
        classifier = classifier_class(X_train.shape[1], 10)
        loss_history = classifier.train(
            X_train, y_train,
            learning_rate=lr,
            reg=reg,
            num_iters=num_iters,
            verbose=False
        )

        # Evaluate
        train_acc = np.mean(classifier.predict(X_train) == y_train)
        val_acc = np.mean(classifier.predict(X_val) == y_val)

        results[lr] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'final_loss': loss_history[-1]
        }

        print(f'  Train accuracy: {train_acc:.4f}')
        print(f'  Val accuracy: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr

    print(f'\nBest learning rate: {best_lr} with val accuracy {best_val_acc:.4f}')

    return best_lr, results
```

---

## Neural Networks

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEURAL NETWORK ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TWO-LAYER NEURAL NETWORK:                                                │
│                                                                             │
│   Input Layer      Hidden Layer          Output Layer                       │
│   (3072 neurons)   (100 neurons)         (10 neurons)                       │
│                                                                             │
│      x₁  ────┐                                                              │
│              │    ┌──────────┐                                              │
│      x₂  ────┼───►│  ReLU    │───┐                                         │
│              │    │  neuron  │   │      ┌──────────┐                       │
│      x₃  ────┼───►│    h₁    │───┼─────►│          │                       │
│              │    └──────────┘   │      │ Softmax  │───► P(cat)            │
│       .      │                   │      │          │───► P(dog)            │
│       .      │    ┌──────────┐   │      │  scores  │───► P(car)            │
│       .      │    │  ReLU    │───┼─────►│          │───►  ...              │
│              │    │  neuron  │   │      │          │───► P(truck)          │
│    x₃₀₇₂────┴───►│   h₁₀₀   │───┘      └──────────┘                       │
│                   └──────────┘                                              │
│                                                                             │
│   MATH:                                                                     │
│   h = ReLU(x @ W₁ + b₁)     # Hidden layer: (N, 3072) → (N, 100)          │
│   s = h @ W₂ + b₂           # Output layer: (N, 100) → (N, 10)            │
│   p = softmax(s)            # Probabilities                                │
│                                                                             │
│                                                                             │
│   PARAMETER COUNT:                                                          │
│   W₁: 3072 × 100 = 307,200 parameters                                      │
│   b₁: 100 parameters                                                       │
│   W₂: 100 × 10 = 1,000 parameters                                          │
│   b₂: 10 parameters                                                        │
│   Total: ~308,310 parameters                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Activation Functions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ACTIVATION FUNCTIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHY ACTIVATION FUNCTIONS?                                                 │
│   Without them: f(x) = W₂(W₁x) = (W₂W₁)x = Wx (still linear!)              │
│   With them: Non-linearity allows learning complex patterns                 │
│                                                                             │
│                                                                             │
│   SIGMOID                           TANH                                    │
│   σ(x) = 1/(1+e^(-x))              tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))     │
│                                                                             │
│        │     ___________                 │     ___________                  │
│      1 │    /                          1 │    /                             │
│        │   /                             │   /                              │
│    0.5 │──/──────────               0 │──/──────────                       │
│        │ /                               │ /                                │
│      0 │/                             -1 │/                                 │
│        └──────────────►                  └──────────────►                   │
│               x                                 x                           │
│                                                                             │
│   ✗ Saturates (gradient → 0)        ✗ Saturates                            │
│   ✗ Not zero-centered               ✓ Zero-centered                        │
│   ✗ exp() is slow                   ✗ exp() is slow                        │
│                                                                             │
│                                                                             │
│   ReLU                              LEAKY ReLU                              │
│   f(x) = max(0, x)                  f(x) = max(0.01x, x)                   │
│                                                                             │
│        │       /                          │       /                         │
│        │      /                           │      /                          │
│        │     /                            │     /                           │
│        │    /                             │    /                            │
│      0 │___/                            0 │__/                              │
│        │                                  │/                                │
│        └──────────────►                   └──────────────►                  │
│               x                                  x                          │
│                                                                             │
│   ✓ No saturation for x > 0         ✓ No saturation                        │
│   ✓ Very fast to compute            ✓ No dead neurons                      │
│   ✗ Dead neurons (gradient=0        ✓ Mostly used now                      │
│      for x < 0 forever)                                                    │
│                                                                             │
│   RECOMMENDATION: Start with ReLU, use Leaky ReLU if you see dead neurons │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Backpropagation

### Computational Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTATIONAL GRAPH                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Forward Pass: Compute the output                                         │
│   Backward Pass: Compute gradients using chain rule                        │
│                                                                             │
│   EXAMPLE: f(x,y,z) = (x + y) * z                                          │
│                                                                             │
│   FORWARD PASS:                                                             │
│                                                                             │
│        x = -2 ──────┐                                                       │
│                     │                                                       │
│                     ▼                                                       │
│                   ┌───┐      q = 3                                         │
│                   │ + │ ──────────────┐                                    │
│                   └───┘               │                                    │
│                     ▲                 ▼                                    │
│        y = 5  ──────┘               ┌───┐      f = -12                     │
│                                     │ × │ ──────────────►                  │
│                                     └───┘                                  │
│                                       ▲                                    │
│        z = -4 ────────────────────────┘                                    │
│                                                                             │
│                                                                             │
│   BACKWARD PASS (Chain Rule):                                               │
│                                                                             │
│   Start with ∂f/∂f = 1                                                     │
│                                                                             │
│        ∂f/∂x = -4 ◄────┐                                                   │
│                        │                                                   │
│                        │                                                   │
│                      ┌───┐     ∂f/∂q = z = -4                              │
│                      │ + │ ◄──────────────┐                                │
│                      └───┘                │                                │
│                        │                  │                                │
│        ∂f/∂y = -4 ◄────┘                ┌───┐     ∂f/∂f = 1                │
│                                         │ × │ ◄──────────────              │
│                                         └───┘                              │
│                                           │                                │
│        ∂f/∂z = q = 3 ◄────────────────────┘                                │
│                                                                             │
│                                                                             │
│   LOCAL GRADIENTS × UPSTREAM GRADIENT = DOWNSTREAM GRADIENT                │
│                                                                             │
│   For multiplication:  ∂(ab)/∂a = b,  ∂(ab)/∂b = a                         │
│   For addition:        ∂(a+b)/∂a = 1,  ∂(a+b)/∂b = 1                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Backprop Through Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              BACKPROPAGATION THROUGH NEURAL NETWORK                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FORWARD PASS:                                                             │
│                                                                             │
│   x ──► [Linear: W₁x+b₁] ──► [ReLU] ──► [Linear: W₂h+b₂] ──► [Softmax+Loss]│
│          z₁ = W₁x+b₁         h=max(0,z₁)    s = W₂h+b₂          L          │
│                                                                             │
│                                                                             │
│   BACKWARD PASS (right to left):                                            │
│                                                                             │
│   Step 1: ∂L/∂s (gradient of loss w.r.t. scores)                           │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  For softmax + cross-entropy:                                      │   │
│   │  ∂L/∂s_j = p_j - 1  if j = correct class                          │   │
│   │  ∂L/∂s_j = p_j      otherwise                                     │   │
│   │  (where p = softmax(s))                                            │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Step 2: ∂L/∂W₂ and ∂L/∂h (gradients for second linear layer)             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  s = W₂h + b₂                                                      │   │
│   │  ∂L/∂W₂ = h.T @ (∂L/∂s)    # Outer product                        │   │
│   │  ∂L/∂b₂ = sum(∂L/∂s)       # Sum over batch                       │   │
│   │  ∂L/∂h = (∂L/∂s) @ W₂.T    # Backprop to input                    │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Step 3: ∂L/∂z₁ (gradient through ReLU)                                   │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  h = max(0, z₁)                                                    │   │
│   │  ∂L/∂z₁ = ∂L/∂h * (z₁ > 0)  # Gradient is 0 where z₁ ≤ 0         │   │
│   │                                                                    │   │
│   │       ∂h/∂z₁                                                       │   │
│   │          │  1 _______________                                      │   │
│   │          │  │               (gradient = 1 for positive inputs)    │   │
│   │          │  │                                                      │   │
│   │    ──────┴──┴───────────►                                         │   │
│   │          0                (gradient = 0 for negative inputs)      │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Step 4: ∂L/∂W₁ (gradients for first linear layer)                        │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  z₁ = W₁x + b₁                                                     │   │
│   │  ∂L/∂W₁ = x.T @ (∂L/∂z₁)                                          │   │
│   │  ∂L/∂b₁ = sum(∂L/∂z₁)                                             │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
#==============================================================================
# EXERCISE 4.1: Two-Layer Neural Network
#==============================================================================

import numpy as np

class TwoLayerNet:
    """
    A two-layer fully-connected neural network with ReLU activation.

    Architecture: input - fully connected layer - ReLU - fully connected layer - softmax
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model with small random weights and zero biases.

        Args:
            input_size: Dimension of input (e.g., 3072 for CIFAR-10)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes (e.g., 10)
            std: Standard deviation for weight initialization
        """
        self.params = {
            'W1': std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a minibatch.

        Args:
            X: Input data of shape (N, D)
            y: Training labels of shape (N,). If None, return only scores.
            reg: Regularization strength

        Returns:
            If y is None: scores of shape (N, C)
            If y is not None: tuple of (loss, grads) where grads is a dictionary
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        #=======================================================================
        # FORWARD PASS
        #=======================================================================

        # First layer: linear + ReLU
        z1 = X @ W1 + b1              # (N, H)
        h1 = np.maximum(0, z1)         # ReLU activation

        # Second layer: linear
        scores = h1 @ W2 + b2          # (N, C)

        # If no labels provided, return scores
        if y is None:
            return scores

        #=======================================================================
        # COMPUTE LOSS
        #=======================================================================

        # Numerical stability: shift scores
        scores -= np.max(scores, axis=1, keepdims=True)

        # Softmax probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)

        # Cross-entropy loss
        correct_log_probs = -np.log(probs[np.arange(N), y])
        data_loss = np.sum(correct_log_probs) / N

        # L2 regularization
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        loss = data_loss + reg_loss

        #=======================================================================
        # BACKWARD PASS
        #=======================================================================

        grads = {}

        # Gradient of softmax + cross-entropy loss
        dscores = probs.copy()                    # (N, C)
        dscores[np.arange(N), y] -= 1
        dscores /= N

        # Gradient for W2 and b2
        grads['W2'] = h1.T @ dscores + reg * W2   # (H, C)
        grads['b2'] = np.sum(dscores, axis=0)     # (C,)

        # Backprop to hidden layer
        dh1 = dscores @ W2.T                      # (N, H)

        # Gradient through ReLU
        dz1 = dh1 * (z1 > 0)                      # (N, H)

        # Gradient for W1 and b1
        grads['W1'] = X.T @ dz1 + reg * W1        # (D, H)
        grads['b1'] = np.sum(dz1, axis=0)         # (H,)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100, batch_size=200, verbose=False):
        """
        Train the neural network using SGD.

        Args:
            X: Training data of shape (N, D)
            y: Training labels of shape (N,)
            X_val: Validation data
            y_val: Validation labels
            learning_rate: Initial learning rate
            learning_rate_decay: Multiplicative decay after each epoch
            reg: Regularization strength
            num_iters: Number of iterations
            batch_size: Size of each mini-batch
            verbose: Print progress

        Returns:
            stats: Dictionary with training history
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # Sample mini-batch
            batch_idx = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # Compute loss and gradients
            loss, grads = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            #===================================================================
            # TODO: Update parameters using gradient descent
            #===================================================================
            for param_name in self.params:
                self.params[param_name] -= learning_rate * grads[param_name]
            #===================================================================

            if verbose and it % 100 == 0:
                print(f'Iteration {it}/{num_iters}: loss = {loss:.4f}')

            # Decay learning rate at end of each epoch
            if it % iterations_per_epoch == 0:
                learning_rate *= learning_rate_decay

                # Track accuracy
                train_acc = np.mean(self.predict(X) == y)
                val_acc = np.mean(self.predict(X_val) == y_val)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, X):
        """
        Use trained weights to predict labels for input data.

        Args:
            X: Input data of shape (N, D)

        Returns:
            y_pred: Predicted labels of shape (N,)
        """
        scores = self.loss(X)  # Returns scores when y=None
        return np.argmax(scores, axis=1)


#==============================================================================
# EXERCISE 4.2: Hyperparameter Tuning
#==============================================================================

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Tune hyperparameters for the two-layer network.

    EXERCISE: Try different combinations and find the best accuracy.

    Target: > 50% validation accuracy
    """
    best_val_acc = 0
    best_params = {}

    # Hyperparameter ranges to search
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    hidden_sizes = [50, 100, 200, 500]
    regularization_strengths = [1e-5, 5e-5, 1e-4, 5e-4]

    results = []

    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for reg in regularization_strengths:

                print(f'\nTrying: lr={lr}, hidden={hidden_size}, reg={reg}')

                # Create and train network
                net = TwoLayerNet(
                    input_size=X_train.shape[1],
                    hidden_size=hidden_size,
                    output_size=10
                )

                stats = net.train(
                    X_train, y_train,
                    X_val, y_val,
                    learning_rate=lr,
                    reg=reg,
                    num_iters=1000,
                    batch_size=200,
                    verbose=False
                )

                # Evaluate
                train_acc = np.mean(net.predict(X_train) == y_train)
                val_acc = np.mean(net.predict(X_val) == y_val)

                results.append({
                    'lr': lr,
                    'hidden_size': hidden_size,
                    'reg': reg,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                })

                print(f'  Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}')

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {
                        'learning_rate': lr,
                        'hidden_size': hidden_size,
                        'reg': reg
                    }

    print(f'\nBest validation accuracy: {best_val_acc:.4f}')
    print(f'Best parameters: {best_params}')

    return best_params, results
```

---

## Coding Exercises

### Exercise Summary

| Exercise | Topic | Difficulty | Skills Practiced |
|----------|-------|------------|------------------|
| 1.1 | k-NN Implementation | Beginner | NumPy, vectorization |
| 1.2 | Cross-Validation | Beginner | Model selection |
| 2.1 | SVM/Softmax Loss | Intermediate | Loss functions, gradients |
| 3.1 | Gradient Descent | Intermediate | Optimization |
| 3.2 | Gradient Checking | Intermediate | Debugging |
| 3.3 | Learning Rate Search | Intermediate | Hyperparameters |
| 4.1 | Two-Layer Network | Advanced | Backpropagation |
| 4.2 | Hyperparameter Tuning | Advanced | Model optimization |

### Complete Exercise: End-to-End Pipeline

```python
#==============================================================================
# EXERCISE 5: Complete Image Classification Pipeline
#==============================================================================

import numpy as np
import pickle
import os

def load_cifar10(root_path):
    """
    Load CIFAR-10 dataset.

    Args:
        root_path: Path to cifar-10-batches-py directory

    Returns:
        X_train, y_train, X_test, y_test
    """
    # Load training batches
    X_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(root_path, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        X_train.append(batch[b'data'])
        y_train.extend(batch[b'labels'])

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    # Load test batch
    test_file = os.path.join(root_path, 'test_batch')
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X_test = batch[b'data']
    y_test = np.array(batch[b'labels'])

    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, X_test):
    """
    Preprocess data: normalize and reshape.

    EXERCISE: Implement proper preprocessing.
    """
    # Convert to float
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Normalize to [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    # Subtract mean (computed on training set only!)
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return X_train, X_test, mean_image


def create_validation_split(X_train, y_train, val_ratio=0.1):
    """
    Create training/validation split.

    EXERCISE: Implement proper splitting with shuffling.
    """
    num_train = X_train.shape[0]
    num_val = int(num_train * val_ratio)

    # Shuffle indices
    indices = np.random.permutation(num_train)

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    return X_train, y_train, X_val, y_val


def run_pipeline():
    """
    Complete training pipeline.

    EXERCISE: Run this end-to-end and achieve > 50% test accuracy.
    """
    print("=" * 60)
    print("CIFAR-10 Classification Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading CIFAR-10...")
    # X_train, y_train, X_test, y_test = load_cifar10('path/to/cifar-10')

    # For testing without real data, create dummy data
    np.random.seed(42)
    X_train = np.random.randn(5000, 3072)
    y_train = np.random.randint(0, 10, 5000)
    X_test = np.random.randn(1000, 3072)
    y_test = np.random.randint(0, 10, 1000)

    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")

    # 2. Preprocess
    print("\n2. Preprocessing data...")
    X_train, X_test, mean_image = preprocess_data(X_train, X_test)

    # 3. Create validation split
    print("\n3. Creating validation split...")
    X_train, y_train, X_val, y_val = create_validation_split(X_train, y_train)
    print(f"   Training: {X_train.shape[0]} examples")
    print(f"   Validation: {X_val.shape[0]} examples")

    # 4. Train model
    print("\n4. Training neural network...")
    net = TwoLayerNet(
        input_size=3072,
        hidden_size=200,
        output_size=10
    )

    stats = net.train(
        X_train, y_train,
        X_val, y_val,
        learning_rate=1e-3,
        learning_rate_decay=0.95,
        reg=5e-5,
        num_iters=2000,
        batch_size=200,
        verbose=True
    )

    # 5. Evaluate
    print("\n5. Evaluating model...")
    train_acc = np.mean(net.predict(X_train) == y_train)
    val_acc = np.mean(net.predict(X_val) == y_val)
    test_acc = np.mean(net.predict(X_test) == y_test)

    print(f"   Training accuracy: {train_acc:.4f}")
    print(f"   Validation accuracy: {val_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")

    # 6. Visualize training
    print("\n6. Training complete!")
    print(f"   Final loss: {stats['loss_history'][-1]:.4f}")

    return net, stats


if __name__ == '__main__':
    run_pipeline()
```

---

## Business Applications

### Real-World Use Cases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BUSINESS APPLICATIONS OF WEEK 1-2 CONCEPTS               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. IMAGE CLASSIFICATION                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  • Product categorization in e-commerce (clothes, electronics...)  │  │
│  │  • Quality control in manufacturing (defect detection)             │  │
│  │  • Medical imaging (tumor classification, X-ray analysis)          │  │
│  │  • Document classification (invoices, receipts, forms)             │  │
│  │  • Brand logo detection in social media monitoring                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  2. CONTENT MODERATION                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  • Classify images as safe/unsafe                                   │  │
│  │  • Detect inappropriate content at scale                            │  │
│  │  • Automate review queues (high confidence auto-approve)            │  │
│  │  • Reduce manual moderation costs                                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  3. RECOMMENDATION SYSTEMS                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  • Visual similarity search (k-NN on image features)                │  │
│  │  • "Customers who viewed this also viewed..."                       │  │
│  │  • Fashion/furniture style matching                                 │  │
│  │  • Reverse image search                                             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  4. PRICING & VALUATION                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  • Real estate: estimate property value from photos                 │  │
│  │  • Used cars: assess condition from images                          │  │
│  │  • Art/collectibles: authenticity and value estimation              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Production Code Example

```python
#==============================================================================
# BUSINESS APPLICATION: Product Image Classifier
#==============================================================================

import numpy as np
from typing import List, Tuple, Dict
import json

class ProductClassifier:
    """
    Production-ready product image classifier for e-commerce.

    Use Cases:
    - Automatic product categorization on upload
    - Inventory management
    - Search improvement
    """

    CATEGORIES = [
        'electronics', 'clothing', 'home_garden', 'sports',
        'toys', 'books', 'automotive', 'beauty', 'food', 'other'
    ]

    def __init__(self, model_path: str = None):
        """
        Initialize classifier.

        Args:
            model_path: Path to saved model weights (optional)
        """
        self.model = TwoLayerNet(
            input_size=3072,  # 32x32x3 images
            hidden_size=500,
            output_size=len(self.CATEGORIES)
        )

        if model_path:
            self.load_model(model_path)

        self.mean_image = None
        self.is_trained = False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image for classification.

        Args:
            image: RGB image of shape (H, W, 3) or (32, 32, 3)

        Returns:
            Preprocessed feature vector of shape (3072,)
        """
        # Resize if needed (in production, use proper resizing library)
        if image.shape != (32, 32, 3):
            # Simple resize by sampling (use cv2.resize in production)
            h, w = image.shape[:2]
            image = image[::h//32, ::w//32][:32, :32]

        # Flatten to vector
        features = image.reshape(-1).astype(np.float32)

        # Normalize
        features /= 255.0

        # Subtract mean if available
        if self.mean_image is not None:
            features -= self.mean_image

        return features

    def classify(self, image: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Classify a product image.

        Args:
            image: Input image
            top_k: Number of top predictions to return

        Returns:
            List of dictionaries with 'category' and 'confidence'
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Preprocess
        features = self.preprocess_image(image)

        # Get scores
        scores = self.model.loss(features.reshape(1, -1))

        # Convert to probabilities
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        probs = probs.flatten()

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'category': self.CATEGORIES[idx],
                'confidence': float(probs[idx]),
                'category_id': int(idx)
            })

        return results

    def classify_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Classify multiple images efficiently.

        Args:
            images: List of input images

        Returns:
            List of classification results for each image
        """
        # Preprocess all images
        features = np.array([self.preprocess_image(img) for img in images])

        # Get all scores at once
        scores = self.model.loss(features)

        # Convert to probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Get predictions
        results = []
        for i in range(len(images)):
            pred_idx = np.argmax(probs[i])
            results.append([{
                'category': self.CATEGORIES[pred_idx],
                'confidence': float(probs[i, pred_idx]),
                'category_id': int(pred_idx)
            }])

        return results

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              **kwargs) -> Dict:
        """
        Train the classifier.

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            **kwargs: Training hyperparameters

        Returns:
            Training statistics
        """
        # Compute and store mean
        self.mean_image = np.mean(X_train, axis=0)

        # Preprocess
        X_train = X_train - self.mean_image
        X_val = X_val - self.mean_image

        # Train
        stats = self.model.train(
            X_train, y_train,
            X_val, y_val,
            **kwargs
        )

        self.is_trained = True
        return stats

    def save_model(self, path: str):
        """Save model weights and preprocessing parameters."""
        data = {
            'params': {k: v.tolist() for k, v in self.model.params.items()},
            'mean_image': self.mean_image.tolist() if self.mean_image is not None else None,
            'categories': self.CATEGORIES
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_model(self, path: str):
        """Load model weights and preprocessing parameters."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.model.params = {k: np.array(v) for k, v in data['params'].items()}
        self.mean_image = np.array(data['mean_image']) if data['mean_image'] else None
        self.is_trained = True

    def get_model_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Get detailed model metrics for business reporting.

        Returns:
            Dictionary with accuracy, per-class metrics, etc.
        """
        predictions = self.model.predict(X_test - self.mean_image)

        # Overall accuracy
        accuracy = np.mean(predictions == y_test)

        # Per-class metrics
        per_class = {}
        for i, category in enumerate(self.CATEGORIES):
            mask = y_test == i
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == i)
                per_class[category] = {
                    'accuracy': float(class_acc),
                    'num_samples': int(np.sum(mask))
                }

        return {
            'overall_accuracy': float(accuracy),
            'per_class_metrics': per_class,
            'total_test_samples': len(y_test)
        }


# Example usage
def demo_product_classifier():
    """Demonstrate the product classifier."""

    # Create classifier
    classifier = ProductClassifier()

    # Generate dummy training data
    np.random.seed(42)
    X_train = np.random.rand(1000, 32, 32, 3) * 255
    y_train = np.random.randint(0, 10, 1000)
    X_val = np.random.rand(200, 32, 32, 3) * 255
    y_val = np.random.randint(0, 10, 200)

    # Flatten for training
    X_train_flat = X_train.reshape(1000, -1).astype(np.float32) / 255
    X_val_flat = X_val.reshape(200, -1).astype(np.float32) / 255

    # Train
    print("Training product classifier...")
    stats = classifier.train(
        X_train_flat, y_train,
        X_val_flat, y_val,
        learning_rate=1e-3,
        num_iters=500,
        verbose=True
    )

    # Classify a new product image
    new_image = np.random.rand(32, 32, 3) * 255
    results = classifier.classify(new_image, top_k=3)

    print("\nClassification results:")
    for r in results:
        print(f"  {r['category']}: {r['confidence']:.2%}")

    return classifier


if __name__ == '__main__':
    demo_product_classifier()
```

---

## Summary: Week 1-2 Checklist

### Concepts You Should Understand
- [ ] Data-driven approach vs. hand-coded rules
- [ ] Train/validation/test split
- [ ] k-NN algorithm and distance metrics
- [ ] Linear classifiers as template matching
- [ ] SVM loss (hinge loss) and margin
- [ ] Softmax loss (cross-entropy)
- [ ] Regularization and overfitting
- [ ] Gradient descent optimization
- [ ] Numerical vs. analytic gradients
- [ ] Mini-batch SGD
- [ ] Neural network architecture
- [ ] Activation functions (ReLU, sigmoid, tanh)
- [ ] Backpropagation and chain rule
- [ ] Hyperparameter tuning

### Skills You Should Have
- [ ] Implement k-NN with vectorized distance computation
- [ ] Implement SVM and Softmax loss with gradients
- [ ] Perform gradient checking
- [ ] Train a two-layer neural network from scratch
- [ ] Use cross-validation for model selection
- [ ] Search for optimal hyperparameters

### Next Steps
After completing Week 1-2, you're ready for:
- **Week 3-4**: Convolutional Neural Networks (CNNs)
- Understanding how to work with image spatial structure
- Building deeper networks with proper training techniques
