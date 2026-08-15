# From-Scratch Machine Learning

A collection of machine-learning algorithms implemented from scratch in Python.

The goal of this project is to understand what is happening underneath higher-level libraries such as NumPy, scikit-learn, and PyTorch by implementing the core mathematics and training procedures manually.

## Current Progress

### Implemented

- Perceptron
  - Weighted sum
  - Step activation
  - Bias
  - Random weight initialization
  - Perceptron learning rule
  - Multiple epochs
  - Prediction
  - Accuracy evaluation

- Logistic Regression
  - Weighted sum
  - Sigmoid activation
  - Probability prediction
  - Binary cross-entropy / log loss
  - Gradient descent
  - Weight and bias updates
  - Accuracy evaluation
  - Multiple epochs

- Linear Algebra Utilities
  - Dot product
  - Matrix multiplication
  - Matrix validation
  - Matrix initialization
  - Column extraction

### In Progress

- Multilayer Perceptron (MLP)
  - Forward propagation
  - Multiple layers
  - Matrix-based computation
  - Activation functions
  - Loss calculation
  - Backpropagation
  - Gradient descent
  - Dynamic architecture

## Project Philosophy

Each implementation is built incrementally.

I first implement the underlying mathematical operation manually, verify it with small examples, and then use it as a building block for more complex models.

For example:

```text
dot product
    ↓
matrix multiplication
    ↓
linear transformation
    ↓
activation function
    ↓
perceptron / logistic regression
    ↓
multiple layers
    ↓
multilayer perceptron
    ↓
backpropagation
```

The emphasis is on understanding the relationship between the mathematics and the implementation rather than hiding the operations behind library calls.

## Repository Structure

A suggested structure for the repository is:

```text
from-scratch-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── linear_algebra/
│   ├── __init__.py
│   ├── matrix_operations.py
│   └── test_matrix_operations.py
│
├── models/
│   ├── __init__.py
│   ├── perceptron.py
│   ├── logistic_regression.py
│   └── mlp.py
│
├── experiments/
│   ├── perceptron_experiments.py
│   ├── logistic_regression_experiments.py
│   └── mlp_experiments.py
│
├── examples/
│   ├── and_gate.py
│   ├── or_gate.py
│   └── mlp_dry_run.py
│
└── tests/
    ├── test_perceptron.py
    ├── test_logistic_regression.py
    └── test_mlp.py
```

You do **not** need to create every directory immediately. Start small and expand the repository as the project grows.

## Recommended Separation of Responsibilities

### `linear_algebra/`

Contains general mathematical utilities that are not specific to a particular ML model.

Examples:

- Matrix multiplication
- Matrix addition
- Dot products
- Matrix validation

These functions should ideally know nothing about perceptrons or neural networks.

### `models/`

Contains the actual machine-learning models.

For example:

```text
perceptron.py
logistic_regression.py
mlp.py
```

Each model should contain its parameters, forward computation, training procedure, and prediction/evaluation functionality.

### `examples/`

Small, readable demonstrations.

This is where you can keep things such as:

- AND gate
- OR gate
- Small logistic-regression datasets
- Hand-calculated MLP examples

These examples are especially useful for verifying that your implementation agrees with your manual calculations.

### `experiments/`

Larger experiments used to investigate how the models behave.

For example:

- Changing learning rate
- Changing number of epochs
- Comparing initialization methods
- Testing different datasets
- Observing loss over time

### `tests/`

Automated checks that verify individual components.

For example, your matrix multiplication implementation should be tested against several manually verified cases before it becomes a dependency for the MLP.

## MLP Development Roadmap

The MLP should be developed in stages rather than implemented all at once.

### Stage 1 — Forward Propagation

Represent each layer using:

```text
W
b
activation
```

For a layer:

```text
z = W x + b
a = activation(z)
```

The output activation becomes the input to the next layer.

### Stage 2 — Loss

For binary classification, calculate binary cross-entropy between the target and predicted probability.

### Stage 3 — Output-Layer Gradient

Calculate how the loss changes with respect to the output layer's parameters.

### Stage 4 — Backpropagation

Propagate the gradient backwards through the network using the chain rule.

### Stage 5 — Parameter Updates

Update every weight matrix and bias vector using gradient descent.

### Stage 6 — Dynamic Architecture

Move from a hardcoded network to an MLP that can represent different architectures, for example:

```text
2 → 3 → 1
2 → 4 → 3 → 1
4 → 5 → 5 → 2
```

The architecture should determine the dimensions of the weight matrices and bias vectors automatically.

## Mathematical View

For a simple network:

```text
Input
  ↓
W₁x + b₁
  ↓
activation
  ↓
W₂a₁ + b₂
  ↓
activation
  ↓
prediction
  ↓
loss
```

The forward pass is therefore a sequence of matrix operations.

For example:

```text
z₁ = W₁x + b₁
a₁ = σ(z₁)

z₂ = W₂a₁ + b₂
ŷ = σ(z₂)
```

Training then works backwards:

```text
loss
  ↓
output gradient
  ↓
gradient through W₂ / b₂
  ↓
gradient through activation
  ↓
gradient through W₁ / b₁
  ↓
parameter updates
```

## Goals

The long-term goal is to build an increasingly complete understanding of machine learning from the mathematical foundations upward.

Planned areas include:

- Linear algebra
- Binary classification
- Gradient descent
- Neural networks
- Backpropagation
- Multiclass classification
- Regularization
- Optimization
- Model evaluation
- Eventually, comparison with established ML frameworks

## Disclaimer

This project is primarily educational.

The implementations intentionally favor transparency and readability over performance. In particular, some operations are implemented using Python lists and explicit loops instead of optimized numerical libraries such as Numpy or SciPy.

The purpose is to understand the machinery before relying on abstractions.
